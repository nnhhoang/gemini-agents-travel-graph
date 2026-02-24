"""Tests for prompt A/B testing."""

from unittest.mock import MagicMock

from travel_planner.services.ab_testing import ABTestService


def test_assign_variant_deterministic():
    service = ABTestService(db=MagicMock())
    v1 = service.assign_variant("user123", "test1", ["A", "B"])
    v2 = service.assign_variant("user123", "test1", ["A", "B"])
    assert v1 == v2


def test_assign_variant_distribution():
    service = ABTestService(db=MagicMock())
    variants = set()
    for i in range(100):
        v = service.assign_variant(f"user{i}", "test1", ["A", "B"])
        variants.add(v)
    assert len(variants) == 2


def test_record_outcome():
    mock_db = MagicMock()
    service = ABTestService(db=mock_db)
    service.record_outcome("test1", "A", score=0.85)
    mock_db.put_item.assert_called_once()
