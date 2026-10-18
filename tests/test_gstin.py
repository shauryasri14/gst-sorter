"""GSTIN format, checksum and normalisation."""

from __future__ import annotations

import random

import pytest

from eval.synthetic import make_gstin
from pipeline.validate import (
    GSTIN_CHARS,
    check_gstin,
    gstin_checksum_char,
    is_valid_gstin_checksum,
    is_valid_gstin_format,
    normalize_gstin,
)

# A widely quoted *sample* GSTIN used in public GST documentation; it is not used for anything but this test.
KNOWN_VALID = "27AAPFU0939F1ZV"


def test_known_valid_gstin_passes_format_and_checksum():
    assert is_valid_gstin_format(KNOWN_VALID)
    assert is_valid_gstin_checksum(KNOWN_VALID)


def test_checksum_character_is_computed_from_first_14():
    assert gstin_checksum_char(KNOWN_VALID[:14]) == "V"


def test_wrong_check_character_is_format_valid_but_checksum_invalid():
    bad = KNOWN_VALID[:14] + "W"
    assert is_valid_gstin_format(bad)
    assert not is_valid_gstin_checksum(bad)
    res = check_gstin(bad)
    assert res.format_ok and not res.checksum_ok and not res.fully_valid


@pytest.mark.parametrize("value", [
    "27AAPFU0939F1Z",        # 14 characters
    "27AAPFU0939F1ZVV",      # 16 characters
    "AAAAAAAAAAAAAAA",       # no state digits
    "27AAPFU09391F1ZV",      # PAN digit block wrong length
    "27AAPFU0939F0ZV",       # entity number 0 is not allowed
    "27AAPFU0939F1XV",       # 14th char must be Z
    "",
    None,
])
def test_bad_format_is_rejected(value):
    assert not is_valid_gstin_format(value)
    assert not is_valid_gstin_checksum(value) or value == ""  # checksum never passes for an empty/short id


def test_normalisation_changes_only_case_and_spacing():
    assert normalize_gstin("  27aapfu0939f1zv ") == KNOWN_VALID
    assert normalize_gstin("27AAP-FU0939-F1ZV") == KNOWN_VALID
    assert normalize_gstin("   ") is None
    assert normalize_gstin(None) is None


def test_lookalike_characters_are_never_swapped():
    # An 'O' where a '0' belongs must stay an 'O' (and then fail), not be silently "fixed".
    assert normalize_gstin("27AAPFUO939F1ZV") == "27AAPFUO939F1ZV"
    assert not is_valid_gstin_format(normalize_gstin("27AAPFUO939F1ZV"))


def test_state_lookup_uses_the_first_two_characters():
    res = check_gstin(KNOWN_VALID)
    assert res.state_code == "27" and res.state_name == "Maharashtra" and res.state_known


def test_unknown_state_code_is_reported_unknown():
    res = check_gstin(KNOWN_VALID, state_codes={"07": {"name": "Delhi"}})
    assert res.state_code == "27" and not res.state_known and res.state_name is None


def test_generated_gstins_are_valid():
    rng = random.Random(11)
    for _ in range(200):
        g = make_gstin(rng, rng.choice(["07", "09", "27", "33"]))
        assert is_valid_gstin_format(g) and is_valid_gstin_checksum(g)


def test_every_single_character_misread_is_caught_by_the_checksum():
    """The mod-36 weighted checksum detects any single wrong character in the first 14 positions."""
    rng = random.Random(5)
    for _ in range(25):
        g = make_gstin(rng, "27")
        for pos in range(14):
            for ch in GSTIN_CHARS:
                if ch == g[pos]:
                    continue
                mutated = g[:pos] + ch + g[pos + 1:]
                assert not is_valid_gstin_checksum(mutated), (g, pos, ch)


def test_checksum_rejects_non_alphanumeric_input():
    assert gstin_checksum_char("27AAPFU0939F1!") is None
    assert gstin_checksum_char("short") is None
