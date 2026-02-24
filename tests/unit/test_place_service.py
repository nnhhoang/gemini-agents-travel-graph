"""Tests for place service."""

from unittest.mock import MagicMock

from travel_planner.services.place_service import PlaceService


def test_encode_geohash():
    service = PlaceService(repo=MagicMock())
    h = service.encode_geohash(35.6812, 139.7671)
    assert isinstance(h, str)
    assert len(h) == 7


def test_get_neighbor_hashes():
    service = PlaceService(repo=MagicMock())
    neighbors = service.get_neighbor_hashes(35.6812, 139.7671)
    assert len(neighbors) == 9  # center + 8 neighbors


def test_find_nearby_places():
    mock_repo = MagicMock()
    mock_repo.get_places_by_geohash.return_value = []
    service = PlaceService(repo=mock_repo)
    places = service.find_nearby(35.6812, 139.7671)
    assert isinstance(places, list)
    assert mock_repo.get_places_by_geohash.call_count >= 1
