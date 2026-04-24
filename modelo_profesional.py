"""
modelo_profesional.py - BiLSTM/GRU con atencion y multi-pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuracion import Config


class Attention(nn.Module):
    """Atencion aditiva con dropout y re-normalizacion post-mask."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        rnn_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(
            self.dropout(torch.tanh(self.proj(rnn_output)))
        ).squeeze(-1)

        if mask is not None:
            mask = mask[:, : energy.size(1)]
            energy = energy.masked_fill(~mask, -1e4)

        weights = F.softmax(energy, dim=-1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        context = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)
        return context, weights


class ResidualBlock(nn.Module):
    """Bloque lineal con conexion residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.drop(F.gelu(self.linear(x)))
        return self.norm(hidden + self.residual(x))


class MejorRNN(nn.Module):
    """
    BiLSTM/GRU con:
      - Embedding con LayerNorm y escala aprendida
      - Multi-pooling: atencion + mean + max + ultimo hidden
      - Clasificador con residuales
      - Inicializacion robusta
    """

    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.num_directions = 2 if cfg.bidirectional else 1

        rnn_cls = nn.LSTM if cfg.rnn_type == "lstm" else nn.GRU

        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(cfg.embed_dim)
        self.embedding_drop = nn.Dropout(cfg.dropout * 0.4)
        self.embed_scale = nn.Parameter(torch.ones(1))

        self.rnn = rnn_cls(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        self.sequence_norm = nn.LayerNorm(cfg.hidden_total)

        self.attention = Attention(cfg.hidden_total, dropout=cfg.attention_dropout)
        self.feature_norm = nn.LayerNorm(cfg.feature_dim)

        cls_hidden = max(cfg.classifier_dim, cfg.hidden_total)
        cls_mid = max(cls_hidden // 2, cfg.num_classes * 8)
        self.classifier = nn.Sequential(
            ResidualBlock(cfg.feature_dim, cls_hidden, cfg.dropout),
            nn.Dropout(cfg.dropout),
            ResidualBlock(cls_hidden, cls_mid, cfg.dropout * 0.8),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(cls_mid, cfg.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        nn.init.xavier_uniform_(self.attention.proj.weight)
        nn.init.zeros_(self.attention.proj.bias)
        nn.init.xavier_uniform_(self.attention.score.weight)

        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if self.cfg.rnn_type == "lstm":
                    chunk = param.numel() // 4
                    param.data[chunk : 2 * chunk].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_embedding(self):
        self.embedding.weight.requires_grad_(False)
        self.embed_scale.requires_grad_(False)

    def unfreeze_embedding(self):
        self.embedding.weight.requires_grad_(True)
        self.embed_scale.requires_grad_(True)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def parameter_summary(self) -> str:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        frozen = total - trainable
        lines = [
            f"  Total params    : {total:>12,}",
            f"  Entrenables     : {trainable:>12,}",
        ]
        if frozen:
            lines.append(f"  Congelados      : {frozen:>12,}")
        return "\n".join(lines)

    def _extract_last_hidden(self, hidden_state: torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]
        batch_size = hidden_state.size(1)
        hidden_state = hidden_state.view(
            self.num_layers,
            self.num_directions,
            batch_size,
            self.hidden_dim,
        )
        return hidden_state[-1].transpose(0, 1).contiguous().view(batch_size, -1)

    def _embed_inputs(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x) * self.embed_scale
        emb = self.embedding_norm(emb)
        emb = self.embedding_drop(emb)
        return emb

    def _gather_last_hidden_from_outputs(
        self,
        rnn_out: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(rnn_out.device).clamp(min=1, max=rnn_out.size(1))
        batch_idx = torch.arange(rnn_out.size(0), device=rnn_out.device)
        if self.cfg.bidirectional:
            forward_last = rnn_out[batch_idx, lengths - 1, : self.hidden_dim]
            backward_last = rnn_out[:, 0, self.hidden_dim :]
            return torch.cat([forward_last, backward_last], dim=-1)
        return rnn_out[batch_idx, lengths - 1, :]

    def _classify_sequence(
        self,
        rnn_out: torch.Tensor,
        mask: torch.Tensor,
        last_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask[:, : rnn_out.size(1)]
        mask_f = mask.unsqueeze(-1).to(rnn_out.dtype)

        attn_pool, weights = self.attention(rnn_out, mask=mask)
        mean_pool = (rnn_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        max_pool = rnn_out.masked_fill(~mask.unsqueeze(-1), -1e4).amax(1)

        features = torch.cat([attn_pool, mean_pool, max_pool, last_h], dim=-1)
        features = self.feature_norm(features)
        logits = self.classifier(features)
        return logits, weights

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x != 0
        emb = self._embed_inputs(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        rnn_out = self.sequence_norm(rnn_out)

        last_h = self._extract_last_hidden(hidden)
        return self._classify_sequence(rnn_out, mask, last_h)

    def forward_exportable(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Camino alternativo para exportacion ONNX sin packed sequences.
        """
        emb = self._embed_inputs(x)
        rnn_out, _ = self.rnn(emb)
        rnn_out = self.sequence_norm(rnn_out)

        steps = torch.arange(rnn_out.size(1), device=x.device).unsqueeze(0)
        mask = steps < lengths.to(x.device).clamp(min=1).unsqueeze(1)
        last_h = self._gather_last_hidden_from_outputs(rnn_out, lengths)
        return self._classify_sequence(rnn_out, mask, last_h)

    def __repr__(self) -> str:
        direction = "bi" if self.cfg.bidirectional else "uni"
        return (
            "MejorRNN("
            f"{self.cfg.rnn_type.upper()} {direction} {self.cfg.num_layers}L | "
            f"emb={self.cfg.embed_dim} hidden={self.cfg.hidden_dim} -> feat={self.cfg.feature_dim} | "
            f"vocab=? classes={self.cfg.num_classes})"
        )


if __name__ == "__main__":
    from configuracion import get_config

    cfg = get_config("clasificacion_texto")
    model = MejorRNN(cfg, vocab_size=30_000)

    print(model)
    print(model.parameter_summary())

    x = torch.randint(1, 30_000, (4, 64))
    lengths = torch.tensor([64, 50, 32, 10])
    logits, weights = model(x, lengths)

    print(f"logits : {tuple(logits.shape)}")
    print(f"weights: {tuple(weights.shape)}")
