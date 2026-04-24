"""
augmentacion_texto.py - Augmentacion ligera para clasificacion de texto.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "amazing": ["great", "fantastic", "excellent"],
    "awesome": ["great", "fantastic", "excellent"],
    "awful": ["terrible", "horrible", "bad"],
    "bad": ["poor", "terrible", "awful"],
    "boring": ["dull", "tedious", "slow"],
    "excellent": ["great", "amazing", "fantastic"],
    "fantastic": ["great", "amazing", "excellent"],
    "good": ["nice", "solid", "great"],
    "great": ["excellent", "amazing", "solid"],
    "hate": ["dislike", "loathe", "despise"],
    "horrible": ["terrible", "awful", "bad"],
    "love": ["adore", "like", "enjoy"],
    "nice": ["good", "pleasant", "solid"],
    "poor": ["bad", "weak", "rough"],
    "terrible": ["awful", "horrible", "bad"],
}


def load_synonym_map(path: str | Path | None) -> dict[str, list[str]]:
    if not path:
        return dict(DEFAULT_SYNONYMS)

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("El archivo de sinonimos debe ser un objeto JSON")

    result: dict[str, list[str]] = {}
    for key, values in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(values, str):
            cleaned = [values.strip().lower()] if values.strip() else []
        elif isinstance(values, list):
            cleaned = [
                str(item).strip().lower()
                for item in values
                if str(item).strip()
            ]
        else:
            continue
        if cleaned:
            result[key.strip().lower()] = cleaned
    merged = dict(DEFAULT_SYNONYMS)
    merged.update(result)
    return merged


class TextAugmenter:
    """Aplica reemplazo por sinonimos, swap y borrado aleatorio."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        synonym_prob: float = 0.15,
        swap_prob: float = 0.10,
        delete_prob: float = 0.05,
        max_ops: int = 2,
        synonyms: dict[str, list[str]] | None = None,
        seed: int = 42,
    ):
        self.enabled = enabled
        self.synonym_prob = max(0.0, synonym_prob)
        self.swap_prob = max(0.0, swap_prob)
        self.delete_prob = max(0.0, delete_prob)
        self.max_ops = max(0, int(max_ops))
        self.synonyms = synonyms or dict(DEFAULT_SYNONYMS)
        self.rng = random.Random(seed)

    @property
    def is_active(self) -> bool:
        return self.enabled and self.max_ops > 0

    def augment(self, text: str) -> str:
        if not self.is_active:
            return str(text or "")

        tokens = TOKEN_PATTERN.findall(str(text or ""))
        word_positions = [idx for idx, token in enumerate(tokens) if WORD_PATTERN.fullmatch(token)]
        if not word_positions:
            return str(text or "")

        ops_done = 0

        if self.synonym_prob > 0.0 and ops_done < self.max_ops and self.rng.random() < self.synonym_prob:
            replaced = self._replace_synonym(tokens, word_positions)
            ops_done += int(replaced)

        if self.swap_prob > 0.0 and ops_done < self.max_ops and len(word_positions) > 1 and self.rng.random() < self.swap_prob:
            swapped = self._swap_words(tokens, word_positions)
            ops_done += int(swapped)

        if self.delete_prob > 0.0 and ops_done < self.max_ops and len(word_positions) > 1 and self.rng.random() < self.delete_prob:
            deleted = self._delete_word(tokens, word_positions)
            ops_done += int(deleted)

        if ops_done == 0:
            return str(text or "")
        return self._join_tokens(tokens)

    def _replace_synonym(self, tokens: list[str], word_positions: list[int]) -> bool:
        candidates = []
        for pos in word_positions:
            key = tokens[pos].lower()
            values = self.synonyms.get(key)
            if values:
                candidates.append((pos, values))
        if not candidates:
            return False
        position, values = self.rng.choice(candidates)
        replacement = self.rng.choice(values)
        original = tokens[position]
        if original[:1].isupper():
            replacement = replacement.capitalize()
        tokens[position] = replacement
        return True

    def _swap_words(self, tokens: list[str], word_positions: list[int]) -> bool:
        first, second = sorted(self.rng.sample(word_positions, 2))
        tokens[first], tokens[second] = tokens[second], tokens[first]
        return True

    def _delete_word(self, tokens: list[str], word_positions: list[int]) -> bool:
        if len(word_positions) <= 1:
            return False
        position = self.rng.choice(word_positions)
        del tokens[position]
        return True

    def _join_tokens(self, tokens: list[str]) -> str:
        if not tokens:
            return ""
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
