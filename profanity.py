"""Utilities for detecting profane words in Whisper transcriptions."""

from __future__ import annotations

from typing import Iterable

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


def _normalize(text: str) -> str:
    """Return a lowercase token containing only alphabetic characters."""

    return "".join(ch for ch in text.lower() if ch.isalpha())


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


def is_profane_word(text: str) -> bool:
    """Return True when the supplied word is considered profane."""

    token = _normalize(text)
    if not token:
        return False

    return any(candidate in PROFANE for candidate in _iter_candidates(token))

