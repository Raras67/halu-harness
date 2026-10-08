import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parsing import parse_yes_no


def test_yes_variants():
    assert parse_yes_no("YES") == 1
    assert parse_yes_no("yes") == 1
    assert parse_yes_no("Yes, it is a hallucination.") == 1


def test_no_variants():
    assert parse_yes_no("NO") == 0
    assert parse_yes_no("no") == 0
    assert parse_yes_no("No hallucination detected.") == 0


def test_unparseable():
    assert parse_yes_no("") == -1
    assert parse_yes_no(None) == -1
    assert parse_yes_no("maybe") == -1
    assert parse_yes_no("YES and NO") == -1


def test_embedded():
    assert parse_yes_no("The answer is YES.") == 1
    assert parse_yes_no("I think NO.") == 0
