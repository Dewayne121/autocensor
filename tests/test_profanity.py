from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from profanity import is_profane_word


@pytest.mark.parametrize(
    "word",
    [
        "fuck",
        "Fuck!",
        "fuckers",
        "fuckin'",
        "shitheads",
        "asshole",
        "Bloody",
    ],
)
def test_is_profane_word_detects_variants(word):
    assert is_profane_word(word)


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
def test_is_profane_word_ignores_clean_words(word):
    assert not is_profane_word(word)

