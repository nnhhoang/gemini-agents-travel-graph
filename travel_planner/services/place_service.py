"""
Place enrichment service.

Handles geohash encoding and nearby place lookups.
"""

from typing import Any

import pygeohash as gh

from travel_planner.data.conversation_models import Place
from travel_planner.data.repository import DynamoDBRepository


class PlaceService:
    """Service for place-related operations."""

    def __init__(self, repo: DynamoDBRepository):
        self.repo = repo

    def encode_geohash(
        self, lat: float, lng: float, precision: int = 7
    ) -> str:
        """Encode lat/lng to geohash string."""
        return gh.encode(lat, lng, precision=precision)

    def get_neighbor_hashes(
        self, lat: float, lng: float, precision: int = 7
    ) -> list[str]:
        """Get center + 8 neighboring geohash cells."""
        center = gh.encode(lat, lng, precision=precision)
        top = gh.get_adjacent(center, "top")
        bottom = gh.get_adjacent(center, "bottom")
        left = gh.get_adjacent(center, "left")
        right = gh.get_adjacent(center, "right")
        top_left = gh.get_adjacent(top, "left")
        top_right = gh.get_adjacent(top, "right")
        bottom_left = gh.get_adjacent(bottom, "left")
        bottom_right = gh.get_adjacent(bottom, "right")
        return [center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right]

    def find_nearby(
        self, lat: float, lng: float, precision: int = 7
    ) -> list[Place]:
        """Find places near a location by querying neighboring geohashes."""
        hashes = self.get_neighbor_hashes(lat, lng, precision)
        places = []
        seen_ids: set[str] = set()
        for h in hashes:
            for place in self.repo.get_places_by_geohash(h):
                if place.place_id not in seen_ids:
                    places.append(place)
                    seen_ids.add(place.place_id)
        return places
