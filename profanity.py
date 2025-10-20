"""Utilities for detecting profane words in Whisper transcriptions."""

from __future__ import annotations

from typing import Iterable, Sequence

PROFANE = {
    "ass",
    "asshole",
    "bastard",
    "bitch",
    "bloody",
    "bullshit",
    "cock",
    "crap",
    "cunt",
    "damn",
    "dick",
    "douche",
    "fucker",
    "fucking",
    "fuck",
    "goddamn",
    "hell",
    "motherfucker",
    "nigger",
    "piss",
    "shit",
    "shithead",
    "slut",
    "twat",
    "wanker",
}

_PROFANE_PHRASES = (
    ("bull", "shit"),
    ("god", "damn"),
    ("mother", "fucker"),
)


_BASIC_SUFFIXES = (
    "s",
    "es",
    "ed",
    "er",
    "ers",
    "est",
    "ing",
    "ly",
)

_SUBSTITUTIONS = (
    ("ies", "y"),
    ("ied", "y"),
    ("iest", "y"),
)

_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
})


def normalize_token(text: str) -> str:
    """Return a lowercase token containing only alphabetic characters."""

    translated = text.lower().translate(_LEET_MAP)
    return "".join(ch for ch in translated if ch.isalpha())


def _iter_candidates(token: str) -> Iterable[str]:
    """Yield plausible base forms for a profanity candidate."""

    if token:
        yield token

    # Handle truncated gerunds such as "fuckin'".
    if token.endswith("in"):
        yield token + "g"

    for suffix, replacement in _SUBSTITUTIONS:
        if token.endswith(suffix) and len(token) > len(suffix):
            base = token[: -len(suffix)] + replacement
            if base:
                yield base

    for suffix in _BASIC_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            base = token[: -len(suffix)]
            if base:
                yield base


def is_profane_token(text: str) -> bool:
    """Return True when the supplied word is considered profane."""

    token = normalize_token(text)
    if not token:
        return False

    return any(candidate in PROFANE for candidate in _iter_candidates(token))


def merge_spans(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []

    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_profane_spans(
    words: Sequence[dict[str, float | str | None]],
    pad_before: float = 0.05,
    pad_after: float = 0.1,
) -> list[tuple[float, float]]:
    """Return timestamp spans that contain profane content."""

    spans: list[tuple[float, float]] = []
    normalized = [normalize_token(str(word.get("text", ""))) for word in words]
    n_words = len(words)
    i = 0
    while i < n_words:
        token = normalized[i]
        matched = False

        for phrase in _PROFANE_PHRASES:
            length = len(phrase)
            if i + length <= n_words and normalized[i : i + length] == list(phrase):
                start = min((words[i + offset].get("start") or 0.0) for offset in range(length))
                end = max((words[i + offset].get("end") or 0.0) for offset in range(length))
                spans.append((max(0.0, start - pad_before), end + pad_after))
                i += length
                matched = True
                break

        if matched:
            continue

        if is_profane_token(token):
            start = words[i].get("start") or 0.0
            end = words[i].get("end") or 0.0
            spans.append((max(0.0, start - pad_before), end + pad_after))

        i += 1

    return merge_spans(spans)

