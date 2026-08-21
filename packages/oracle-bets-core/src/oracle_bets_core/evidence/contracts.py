"""Small evidence enums shared by recording and performance code."""

from enum import StrEnum


class DecisionMode(StrEnum):
    """Evidence cohorts that must never be blended silently."""

    PREMATCH = "prematch"
