from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from profanity import (
    detect_profane_spans,
    is_profane_token,
    merge_spans,
    normalize_token,
)


def test_normalize_token_handles_punctuation_and_leetspeak():
    assert normalize_token("Sh1t!!") == "shit"


@pytest.mark.parametrize(
    "word",
    [
        "class",
        "passing",
        "assignment",
        "hello",
        "sunny",
    ],
)
def test_is_profane_token_ignores_clean_words(word):
    assert not is_profane_token(word)


@pytest.mark.parametrize(
    "word",
    [
        "fuck",
        "Fuck!",
        "fuckers",
        "fuckin'",
        "sh1thead",
        "asshole",
        "Bloody",
    ],
)
def test_is_profane_token_detects_variants(word):
    assert is_profane_token(word)


def test_merge_spans_coalesces_overlapping_ranges():
    spans = [(0.0, 0.5), (0.4, 0.8), (1.2, 1.3), (1.25, 1.4)]
    assert merge_spans(spans) == [(0.0, 0.8), (1.2, 1.4)]


def test_detect_profane_spans_captures_split_phrase():
    words = [
        {"text": "that's", "start": 0.0, "end": 0.2},
        {"text": "bull", "start": 0.2, "end": 0.4},
        {"text": "shit", "start": 0.4, "end": 0.6},
        {"text": "friend", "start": 0.6, "end": 0.8},
    ]

    spans = detect_profane_spans(words, pad_before=0.0, pad_after=0.0)
    assert spans == [(0.2, 0.6)]


def test_detect_profane_spans_skips_similar_safe_words():
    words = [
        {"text": "class", "start": 0.0, "end": 0.1},
        {"text": "assignment", "start": 0.1, "end": 0.2},
    ]

    assert detect_profane_spans(words) == []

