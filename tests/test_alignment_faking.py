"""Smoke tests for alignment_faking tasks — no API calls required."""

import re

import pytest

from alignment_faking.scorers import _extract_tag


def test_extract_output_tag():
    text = "<SCRATCHPAD_REASONING>some reasoning</SCRATCHPAD_REASONING>\n<OUTPUT>hello</OUTPUT>"
    assert _extract_tag(text, "OUTPUT") == "hello"


def test_extract_scratchpad_tag():
    text = "<SCRATCHPAD_REASONING>I should comply</SCRATCHPAD_REASONING><OUTPUT>ok</OUTPUT>"
    assert _extract_tag(text, "SCRATCHPAD_REASONING") == "I should comply"


def test_extract_missing_tag():
    assert _extract_tag("no tags here", "OUTPUT") is None


def test_extract_last_tag():
    text = "<OUTPUT>first</OUTPUT><OUTPUT>second</OUTPUT>"
    assert _extract_tag(text, "OUTPUT") == "second"


def test_task_imports():
    from alignment_faking import alignment_faking_free_tier, alignment_faking_paid_tier

    assert callable(alignment_faking_free_tier)
    assert callable(alignment_faking_paid_tier)
