"""
entrenamiento.py — Bucle de entrenamiento, evaluacion y checkpointing.
"""

import time
import warnings
import os 
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configuracion import Config
from modelo_profesional import MejorRNN


# ── Metricas ──────────────────────────────────────────────────────

def _update_cm(matrix: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor, n: int):
    flat    = targets.detach().cpu() * n + preds.detach().cpu()
    matrix += torch.bincount(flat, minlength=n * n).view(n, n)


def _metrics(matrix: torch.Tensor) -> tuple[float, float]:
    total    = matrix.sum().item()
    accuracy = matrix.diag().sum().item() / max(total, 1)
    m        = matrix.float()
    prec     = m.diag() / m.sum(0).clamp(min=1.0)
    rec      = m.diag() / m.sum(1).clamp(min=1.0)
    f1       = (2 * prec * rec / (prec + rec).clamp(min=1e-8)).mean().item()
    return accuracy, f1


def print_confusion_matrix(matrix: torch.Tensor, class_names: list[str]):
    n    = len(class_names)
    col  = max(len(c) for c in class_names) + 2
    head = " " * (col + 2) + "".join(f"{c:>{col}}" for c in class_names)
    print(head)
    for i, name in enumerate(class_names):
        row = f"  {name:<{col}}" + "".join(f"{matrix[i,j].item():>{col}}" for j in range(n))
        print(row)


def classwise_metrics(
    matrix: torch.Tensor,
    class_names: list[str],
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    scores = matrix.float()
    precision = scores.diag() / scores.sum(0).clamp(min=1.0)
    recall = scores.diag() / scores.sum(1).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    for idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class_name": class_name,
                "precision": float(precision[idx].item()),
                "recall": float(recall[idx].item()),
                "f1": float(f1[idx].item()),
                "support": int(matrix[idx].sum().item()),
            }
        )
    return rows


def print_classification_report(matrix: torch.Tensor, class_names: list[str]):
    if not class_names:
        return
    header = f"{'clase':<18}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}"
    print(header)
    print("-" * len(header))
    for row in classwise_metrics(matrix, class_names):
        print(
            f"{str(row['class_name']):<18}"
            f"{row['precision']:>11.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
            f"{row['support']:>10d}"
        )


def _non_blocking(cfg: Config) -> bool:
    return cfg.device_type == "cuda" and cfg.pin_memory


def _create_progress_bar(
    iterable,
    *,
    total: int | None,
    desc: str,
    enabled: bool,
    leave: bool = False,
):
    if not enabled:
        return iterable, None
    progress = tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        ascii=True,
    )
    return progress, progress


def _set_progress_postfix(
    progress,
    *,
    loss: float,
    acc: float,
    lr: float | None = None,
    gnorm: float | None = None,
):
    if progress is None:
        return
    postfix = {
        "loss": f"{loss:.4f}",
        "acc": f"{acc:.4f}",
    }
    if lr is not None:
        postfix["lr"] = f"{lr:.2e}"
    if gnorm is not None:
        postfix["gnorm"] = f"{gnorm:.2f}"
    progress.set_postfix(postfix, refresh=False)


# ── Train epoch ───────────────────────────────────────────────────

def train_epoch(
    modelo:    MejorRNN,
    loader:    DataLoader,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    scaler:    torch.amp.GradScaler,
    cfg:       Config,
    epoch:     int | None = None,
) -> tuple[float, float, float, float, float]:
    """Devuelve (loss, accuracy, f1, grad_norm_avg, lr_final)."""
    modelo.train()
    total_loss  = 0.0
    total_items = 0
    total_gnorm = 0.0
    n_steps     = 0
    confusion   = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.long)
    amp_enabled = cfg.use_amp and cfg.device_type == "cuda"
    total_steps = len(loader)
    iterator, progress = _create_progress_bar(
        loader,
        total=total_steps,
        desc=f"Epoca {epoch}/{cfg.epochs} [train]" if epoch is not None else "Train",
        enabled=cfg.show_progress,
        leave=False,
    )
    non_blocking = _non_blocking(cfg)

    for step, (x, lengths, y) in enumerate(iterator, 1):
        x       = x.to(cfg.device, non_blocking=non_blocking)
        lengths = lengths.to(cfg.device, non_blocking=non_blocking)
        y       = y.to(cfg.device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=cfg.device_type, enabled=amp_enabled):
            logits, _ = modelo(x, lengths)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
            optimizer.step()

        scheduler.step()

        preds = logits.argmax(1)
        _update_cm(confusion, y, preds, cfg.num_classes)
        bs          = x.size(0)
        total_loss  += loss.item() * bs
        total_items += bs
        total_gnorm += gnorm.item()
        n_steps     += 1

        running_loss = total_loss / max(total_items, 1)
        lr_current   = scheduler.get_last_lr()[0]

        if progress is not None and (
            step % cfg.progress_refresh_steps == 0 or step == total_steps
        ):
            acc, _ = _metrics(confusion)
            _set_progress_postfix(
                progress,
                loss=running_loss,
                acc=acc,
                lr=lr_current,
                gnorm=gnorm.item(),
            )
        elif progress is None and cfg.log_interval > 0 and step % cfg.log_interval == 0:
            acc, _ = _metrics(confusion)
            print(
                f"    step {step:>4d}/{total_steps} | "
                f"loss={running_loss:.4f} acc={acc:.4f} lr={lr_current:.2e}"
            )

    acc, f1  = _metrics(confusion)
    avg_gnorm = total_gnorm / max(n_steps, 1)
    lr_final  = scheduler.get_last_lr()[0]
    _set_progress_postfix(
        progress,
        loss=total_loss / max(total_items, 1),
        acc=acc,
        lr=lr_final,
        gnorm=avg_gnorm,
    )
    if progress is not None:
        progress.close()
    return total_loss / max(total_items, 1), acc, f1, avg_gnorm, lr_final


# ── Evaluate ──────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    modelo: MejorRNN,
    loader: DataLoader,
    cfg:    Config,
    epoch:  int | None = None,
) -> tuple[float, float, float]:
    """Devuelve (loss, accuracy, f1)."""
    modelo.eval()
    total_loss  = 0.0
    total_items = 0
    confusion   = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.long)
    amp_enabled = cfg.use_amp and cfg.device_type == "cuda"
    total_steps = len(loader)
    iterator, progress = _create_progress_bar(
        loader,
        total=total_steps,
        desc=f"Epoca {epoch}/{cfg.epochs} [val]" if epoch is not None else "Validacion",
        enabled=cfg.show_eval_progress,
        leave=False,
    )
    non_blocking = _non_blocking(cfg)

    for step, (x, lengths, y) in enumerate(iterator, 1):
        x       = x.to(cfg.device, non_blocking=non_blocking)
        lengths = lengths.to(cfg.device, non_blocking=non_blocking)
        y       = y.to(cfg.device, non_blocking=non_blocking)

        with torch.amp.autocast(device_type=cfg.device_type, enabled=amp_enabled):
            logits, _ = modelo(x, lengths)
            loss = F.cross_entropy(logits, y)

        _update_cm(confusion, y, logits.argmax(1), cfg.num_classes)
        total_loss  += loss.item() * x.size(0)
        total_items += x.size(0)

        if progress is not None and (
            step % cfg.progress_refresh_steps == 0 or step == total_steps
        ):
            acc, _ = _metrics(confusion)
            _set_progress_postfix(
                progress,
                loss=total_loss / max(total_items, 1),
                acc=acc,
            )

    acc, f1 = _metrics(confusion)
    _set_progress_postfix(
        progress,
        loss=total_loss / max(total_items, 1),
        acc=acc,
    )
    if progress is not None:
        progress.close()
    return total_loss / max(total_items, 1), acc, f1


@torch.inference_mode()
def evaluate_detailed(
    modelo: MejorRNN,
    loader: DataLoader,
    cfg: Config,
    class_names: list[str] | None = None,
    epoch: int | None = None,
) -> dict:
    modelo.eval()
    total_loss  = 0.0
    total_items = 0
    confusion   = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.long)
    amp_enabled = cfg.use_amp and cfg.device_type == "cuda"
    total_steps = len(loader)
    iterator, progress = _create_progress_bar(
        loader,
        total=total_steps,
        desc=f"Epoca {epoch}/{cfg.epochs} [eval]" if epoch is not None else "Evaluacion",
        enabled=cfg.show_eval_progress,
        leave=False,
    )
    non_blocking = _non_blocking(cfg)

    for step, (x, lengths, y) in enumerate(iterator, 1):
        x       = x.to(cfg.device, non_blocking=non_blocking)
        lengths = lengths.to(cfg.device, non_blocking=non_blocking)
        y       = y.to(cfg.device, non_blocking=non_blocking)

        with torch.amp.autocast(device_type=cfg.device_type, enabled=amp_enabled):
            logits, _ = modelo(x, lengths)
            loss = F.cross_entropy(logits, y)

        _update_cm(confusion, y, logits.argmax(1), cfg.num_classes)
        total_loss  += loss.item() * x.size(0)
        total_items += x.size(0)

        if progress is not None and (
            step % cfg.progress_refresh_steps == 0 or step == total_steps
        ):
            acc, _ = _metrics(confusion)
            _set_progress_postfix(
                progress,
                loss=total_loss / max(total_items, 1),
                acc=acc,
            )

    loss = total_loss / max(total_items, 1)
    acc, f1 = _metrics(confusion)
    if progress is not None:
        _set_progress_postfix(progress, loss=loss, acc=acc)
        progress.close()

    names = class_names or [str(idx) for idx in range(cfg.num_classes)]
    return {
        "loss": loss,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": confusion,
        "class_names": names,
        "classwise": classwise_metrics(confusion, names),
    }


# ── Checkpoint ────────────────────────────────────────────────────

def save_checkpoint(
    modelo:       MejorRNN,
    optimizer:    AdamW,
    epoch:        int,
    best_val_acc: float,
    cfg:          Config,
    vocab_state:  dict | None = None,
    label_state:  dict | None = None,
    scheduler:    OneCycleLR | None = None,
    scaler:       torch.amp.GradScaler | None = None,
    best_val_loss: float | None = None,
):
    """
    Guarda modelo, optimizer, config y vocab en un solo archivo.
    Incluir vocab permite hacer inferencia sin recargar el dataset.
    """
    payload = {
        "epoch":        epoch,
        "best_val_acc": best_val_acc,
        "model":        modelo.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "config":       {k: v for k, v in cfg.__dict__.items()},
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if vocab_state is not None:
        payload["vocab"] = vocab_state
    if label_state is not None:
        payload["label_encoder"] = label_state
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, cfg.checkpoint_path)
    print(f"  checkpoint guardado -> {cfg.checkpoint_path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    modelo: 'MejorRNN', # Asegúrate de que MejorRNN esté definido o usa 'Any'
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    """
    Carga el checkpoint de forma segura. 
    Verifica tamaños antes de cargar para evitar el error de 'size mismatch'.
    """
    if not os.path.exists(path):
        print(f"-> Archivo no encontrado en '{path}'. Se ignorará la carga y se empezará de cero.")
        return {}

    try:
        ckpt = torch.load(path, map_location=map_location)
        
        # 1. VERIFICACIÓN DE SEGURIDAD: Comparar tamaños de Embedding
        # Esto evita el crash por size mismatch que recibiste
        ckpt_emb_shape = ckpt["model"]["embedding.weight"].shape
        curr_emb_shape = modelo.embedding.weight.shape
        
        if ckpt_emb_shape != curr_emb_shape:
            print(f"-> AVISO: El vocabulario no coincide. Checkpoint: {ckpt_emb_shape}, Modelo: {curr_emb_shape}")
            print("-> Se omitirá la carga de pesos para evitar errores.")
            return ckpt

        # 2. CARGA SEGURA
        modelo.load_state_dict(ckpt["model"])
        
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        # 3. LOGS Y RETURN
        epoch = ckpt.get("epoch", "?")
        acc   = ckpt.get("best_val_acc", 0.0)
        loss  = ckpt.get("best_val_loss")
        extra = f" | best_val_loss={loss:.4f}" if isinstance(loss, (float, int)) else ""
        
        print(f"Checkpoint cargado exitosamente desde '{path}' | epoch={epoch} | best_val_acc={acc:.4f}{extra}")
        return ckpt

    except Exception as e:
        print(f"-> Error inesperado al cargar el checkpoint: {e}")
        return {}

# ── Loop principal ────────────────────────────────────────────────

def _is_better(val_acc, val_loss, best_acc, best_loss) -> bool:
    if val_acc > best_acc + 1e-4:
        return True
    if abs(val_acc - best_acc) <= 1e-4 and val_loss < best_loss - 1e-4:
        return True
    return False


def entrenar(
    modelo:       MejorRNN,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          Config,
    vocab_state:  dict | None = None,
    label_state:  dict | None = None,
    start_epoch:  int = 1,
    resume_checkpoint: str | None = None,
) -> MejorRNN:
    """
    Entrena el modelo con early stopping y devuelve los mejores pesos.

    Args:
        vocab_state: word2idx del Vocabulary para guardar en el checkpoint
        label_state: estado del codificador de etiquetas
        start_epoch: epoch desde la que retomar (util al resumir entrenamiento)
        resume_checkpoint: checkpoint para reanudar optimizer/scheduler/scaler
    """
    modelo = modelo.to(cfg.device)

    # torch.compile acelera ~10-20% en GPU (PyTorch >= 2.0)
    if cfg.compile_model and cfg.device_type == "cuda" and hasattr(torch, "compile"):
        print("Compilando modelo con torch.compile()...")
        modelo = torch.compile(modelo)

    # ── Optimizador ───────────────────────────────────────────────
    decay, no_decay = [], []
    for name, param in modelo.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )

    # ── Scheduler ─────────────────────────────────────────────────
    scheduler = OneCycleLR(
        optimizer,
        max_lr            = cfg.lr,
        steps_per_epoch   = max(len(train_loader), 1),
        epochs            = cfg.epochs,
        pct_start         = cfg.pct_warmup,
        anneal_strategy   = "cos",
        div_factor        = 25.0,
        final_div_factor  = 100.0,
    )

    # FIX: no pasar device= para compatibilidad con PyTorch < 2.3
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and cfg.device_type == "cuda")

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    best_epoch    = start_epoch
    patience      = 0

    if resume_checkpoint:
        ckpt = load_checkpoint(
            resume_checkpoint,
            modelo,
            optimizer=optimizer,
            scaler=scaler,
            map_location=cfg.device,
        )
        ckpt_epoch = int(ckpt.get("epoch", 0))
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = ckpt_epoch if ckpt_epoch > 0 else start_epoch
        start_epoch = max(start_epoch, ckpt_epoch + 1)
        completed_steps = ckpt_epoch * max(len(train_loader), 1)
        if completed_steps > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                )
                for _ in range(completed_steps):
                    scheduler.step()
        print(f"Reanudando entrenamiento desde epoch {start_epoch}")

        if start_epoch > cfg.epochs:
            print("El checkpoint ya alcanzo o supero la ultima epoca configurada.")
            return modelo

    header = (f"\n{'epoca':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'tr_f1':>7}  "
              f"{'val_loss':>8}  {'val_acc':>7}  {'val_f1':>7}  "
              f"{'gnorm':>6}  {'lr':>8}  {'t':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1, gnorm, lr = train_epoch(
            modelo, train_loader, optimizer, scheduler, scaler, cfg, epoch=epoch
        )
        val_loss, val_acc, val_f1 = evaluate(modelo, val_loader, cfg, epoch=epoch)

        elapsed  = time.time() - t0
        improved = _is_better(val_acc, val_loss, best_val_acc, best_val_loss)
        marker   = " *" if improved else ""

        print(
            f"  {epoch:>3d}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_f1:>7.4f}  "
            f"{val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}  "
            f"{gnorm:>6.2f}  {lr:>8.2e}  {elapsed:>5.0f}s{marker}"
        )

        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            patience      = 0
            save_checkpoint(
                modelo,
                optimizer,
                epoch,
                best_val_acc,
                cfg,
                vocab_state,
                label_state,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"\nEarly stopping en epoch {epoch} "
                      f"(sin mejora por {cfg.early_stopping_patience} epocas).")
                break

    print(f"\nMejor validacion -> epoch={best_epoch} "
          f"acc={best_val_acc:.4f} loss={best_val_loss:.4f}")

    load_checkpoint(cfg.checkpoint_path, modelo)
    return modelo
