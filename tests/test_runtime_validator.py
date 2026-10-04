"""Structural release validation without unsafe model deserialization."""

from pathlib import Path

from tools.validate_release_runtime import validate


def test_complete_release_structure() -> None:
    report = validate(Path("artifacts"), deserialize=False)
    assert report["models"] == 360
    assert report["databases"] == 72
    assert report["trial_counts"] == {"20": 36, "100": 36}
