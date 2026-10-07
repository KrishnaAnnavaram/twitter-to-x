"""Moderation-talk detection: word-boundary patterns by category, and validation against a labelled set."""

from .detector import CATEGORIES, ModerationDetector, substring_flag

__all__ = ["CATEGORIES", "ModerationDetector", "substring_flag"]
