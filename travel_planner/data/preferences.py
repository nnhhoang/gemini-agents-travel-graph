"""
User preference models for the travel planner conversation engine.

Maps TypeScript preference union types to Python StrEnum + Pydantic model.
These preferences drive personalized recommendations via Gemini prompts.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class TravelFrequency(StrEnum):
    YEAR_ONCE = "YEAR_ONCE"
    YEAR_MULTI = "YEAR_MULTI"
    MONTH_ONCE = "MONTH_ONCE"
    WEEK_ONCE = "WEEK_ONCE"
    CASUAL = "CASUAL"


class TravelStyle(StrEnum):
    SIGHTSEEING = "SIGHTSEEING"
    GOURMET = "GOURMET"
    NATURE = "NATURE"
    WORKATION = "WORKATION"
    SOLO = "SOLO"
    FAMILY = "FAMILY"
    COUPLE = "COUPLE"


class TravelPurpose(StrEnum):
    HEALING = "HEALING"
    LEARNING = "LEARNING"
    WORK = "WORK"
    STUDY = "STUDY"
    SOCIAL = "SOCIAL"
    OSHI = "OSHI"


class ActivityStyle(StrEnum):
    ACTIVE = "ACTIVE"
    QUIET = "QUIET"
    PLANNED = "PLANNED"
    SPONTANEOUS = "SPONTANEOUS"


class CuisineType(StrEnum):
    JAPANESE = "JAPANESE"
    WESTERN = "WESTERN"
    CHINESE = "CHINESE"
    ASIAN = "ASIAN"
    ETHNIC = "ETHNIC"
    ITALIAN = "ITALIAN"
    FRENCH = "FRENCH"
    BBQ = "BBQ"
    SEAFOOD = "SEAFOOD"
    NOODLE = "NOODLE"
    SWEETS = "SWEETS"
    LOCAL = "LOCAL"
    B_GRADE = "B_GRADE"
    WILD_GAME = "WILD_GAME"
    STREET_FOOD = "STREET_FOOD"


class DiningStyle(StrEnum):
    FAMOUS = "FAMOUS"
    LOCAL = "LOCAL"
    HIDDEN = "HIDDEN"
    STREET_FOOD = "STREET_FOOD"
    COURSE = "COURSE"
    BUFFET = "BUFFET"
    BREAKFAST = "BREAKFAST"
    NIGHT_DRINK = "NIGHT_DRINK"
    LUNCH_DRINK = "LUNCH_DRINK"


class BudgetPreference(StrEnum):
    REASONABLE = "REASONABLE"
    BALANCED = "BALANCED"
    LUXURY = "LUXURY"
    UNLIMITED = "UNLIMITED"


class DietaryRestriction(StrEnum):
    NO_SPICY = "NO_SPICY"
    NO_RAW = "NO_RAW"
    ALLERGIES = "ALLERGIES"
    VEGETARIAN = "VEGETARIAN"
    VEGAN = "VEGAN"
    HALAL = "HALAL"
    GLUTEN_FREE = "GLUTEN_FREE"


class BeverageType(StrEnum):
    ALCOHOL_LOVER = "ALCOHOL_LOVER"
    SAKE = "SAKE"
    SHOCHU = "SHOCHU"
    WINE = "WINE"
    BEER = "BEER"
    CAFE = "CAFE"
    COFFEE = "COFFEE"
    TEA = "TEA"
    NON_ALCOHOL = "NON_ALCOHOL"


class ActivityInterest(StrEnum):
    GOURMET = "GOURMET"
    ALCOHOL = "ALCOHOL"
    PHOTO = "PHOTO"
    HISTORY = "HISTORY"
    THEME_PARK = "THEME_PARK"
    CAFE = "CAFE"
    SHOPPING = "SHOPPING"
    NATURE = "NATURE"
    PARK = "PARK"
    ONSEN = "ONSEN"
    INSTAGRAMMABLE = "INSTAGRAMMABLE"
    CRAFTS = "CRAFTS"
    SPORTS = "SPORTS"
    MUSIC = "MUSIC"


# English labels for each enum value, used in prompt injection.
# Keyed by (EnumClass, value) to handle overlapping values like LOCAL.
_LABELS: dict[tuple[type, str], str] = {
    # TravelFrequency
    (TravelFrequency, "YEAR_ONCE"): "once a year",
    (TravelFrequency, "YEAR_MULTI"): "several times a year",
    (TravelFrequency, "MONTH_ONCE"): "once a month",
    (TravelFrequency, "WEEK_ONCE"): "once a week",
    (TravelFrequency, "CASUAL"): "whenever I feel like it",
    # TravelStyle
    (TravelStyle, "SIGHTSEEING"): "sightseeing",
    (TravelStyle, "GOURMET"): "gourmet",
    (TravelStyle, "NATURE"): "nature & outdoors",
    (TravelStyle, "WORKATION"): "workation",
    (TravelStyle, "SOLO"): "solo travel",
    (TravelStyle, "FAMILY"): "family trip",
    (TravelStyle, "COUPLE"): "couple trip",
    # TravelPurpose
    (TravelPurpose, "HEALING"): "relaxation & healing",
    (TravelPurpose, "LEARNING"): "learning",
    (TravelPurpose, "WORK"): "work",
    (TravelPurpose, "STUDY"): "study",
    (TravelPurpose, "SOCIAL"): "socializing",
    (TravelPurpose, "OSHI"): "fandom travel",
    # ActivityStyle
    (ActivityStyle, "ACTIVE"): "active",
    (ActivityStyle, "QUIET"): "relaxed & easygoing",
    (ActivityStyle, "PLANNED"): "well-planned",
    (ActivityStyle, "SPONTANEOUS"): "spontaneous",
    # CuisineType
    (CuisineType, "JAPANESE"): "Japanese",
    (CuisineType, "WESTERN"): "Western",
    (CuisineType, "CHINESE"): "Chinese",
    (CuisineType, "ASIAN"): "Asian",
    (CuisineType, "ETHNIC"): "ethnic",
    (CuisineType, "ITALIAN"): "Italian",
    (CuisineType, "FRENCH"): "French",
    (CuisineType, "BBQ"): "BBQ & grilled meat",
    (CuisineType, "SEAFOOD"): "seafood",
    (CuisineType, "NOODLE"): "noodles",
    (CuisineType, "SWEETS"): "sweets & desserts",
    (CuisineType, "LOCAL"): "regional specialties",
    (CuisineType, "B_GRADE"): "casual local eats",
    (CuisineType, "WILD_GAME"): "wild game",
    (CuisineType, "STREET_FOOD"): "street food",
    # DiningStyle
    (DiningStyle, "FAMOUS"): "famous restaurants",
    (DiningStyle, "LOCAL"): "local favorites",
    (DiningStyle, "HIDDEN"): "hidden gems",
    (DiningStyle, "STREET_FOOD"): "street food stalls",
    (DiningStyle, "COURSE"): "course meals",
    (DiningStyle, "BUFFET"): "buffet",
    (DiningStyle, "BREAKFAST"): "great breakfast spots",
    (DiningStyle, "NIGHT_DRINK"): "evening drinks",
    (DiningStyle, "LUNCH_DRINK"): "daytime drinks",
    # BudgetPreference
    (BudgetPreference, "REASONABLE"): "budget-friendly",
    (BudgetPreference, "BALANCED"): "balanced",
    (BudgetPreference, "LUXURY"): "luxury",
    (BudgetPreference, "UNLIMITED"): "no budget limit",
    # DietaryRestriction
    (DietaryRestriction, "NO_SPICY"): "no spicy food",
    (DietaryRestriction, "NO_RAW"): "no raw food",
    (DietaryRestriction, "ALLERGIES"): "has allergies",
    (DietaryRestriction, "VEGETARIAN"): "vegetarian",
    (DietaryRestriction, "VEGAN"): "vegan",
    (DietaryRestriction, "HALAL"): "halal",
    (DietaryRestriction, "GLUTEN_FREE"): "gluten-free",
    # BeverageType
    (BeverageType, "ALCOHOL_LOVER"): "loves alcohol",
    (BeverageType, "SAKE"): "sake",
    (BeverageType, "SHOCHU"): "shochu",
    (BeverageType, "WINE"): "wine",
    (BeverageType, "BEER"): "beer",
    (BeverageType, "CAFE"): "cafe",
    (BeverageType, "COFFEE"): "coffee",
    (BeverageType, "TEA"): "tea",
    (BeverageType, "NON_ALCOHOL"): "non-alcoholic",
    # ActivityInterest
    (ActivityInterest, "GOURMET"): "food & gourmet",
    (ActivityInterest, "ALCOHOL"): "drinks & bars",
    (ActivityInterest, "PHOTO"): "photography",
    (ActivityInterest, "HISTORY"): "history",
    (ActivityInterest, "THEME_PARK"): "theme parks",
    (ActivityInterest, "CAFE"): "cafe hopping",
    (ActivityInterest, "SHOPPING"): "shopping",
    (ActivityInterest, "NATURE"): "nature",
    (ActivityInterest, "PARK"): "parks",
    (ActivityInterest, "ONSEN"): "hot springs (onsen)",
    (ActivityInterest, "INSTAGRAMMABLE"): "Instagrammable spots",
    (ActivityInterest, "CRAFTS"): "crafts & workshops",
    (ActivityInterest, "SPORTS"): "sports",
    (ActivityInterest, "MUSIC"): "music",
}


def _ja(val: StrEnum) -> str:
    """Return label for an enum value, falling back to the raw value."""
    return _LABELS.get((type(val), val.value), val.value)


class UserPreferences(BaseModel):
    """User preferences for personalized travel recommendations."""

    # Travel style
    travel_frequency: TravelFrequency | None = None
    travel_styles: list[TravelStyle] = Field(default_factory=list)
    travel_purposes: list[TravelPurpose] = Field(default_factory=list)
    activity_style: ActivityStyle | None = None

    # Food & dining
    cuisine_types: list[CuisineType] = Field(default_factory=list)
    dining_styles: list[DiningStyle] = Field(default_factory=list)
    budget_preference: BudgetPreference | None = None
    dietary_restrictions: list[DietaryRestriction] = Field(default_factory=list)
    beverage_types: list[BeverageType] = Field(default_factory=list)

    # Activities & interests
    activity_interests: list[ActivityInterest] = Field(default_factory=list)

    # Free text (user-inputted)
    custom_notes: str | None = None

    def to_prompt_context(self) -> str:
        """Convert preferences to labeled string for prompt injection."""
        parts: list[str] = []
        if self.travel_frequency:
            parts.append(f"Travel frequency: {_ja(self.travel_frequency)}")
        if self.travel_styles:
            parts.append(
                f"Travel style: {', '.join(_ja(s) for s in self.travel_styles)}"
            )
        if self.travel_purposes:
            parts.append(
                f"Travel purpose: {', '.join(_ja(p) for p in self.travel_purposes)}"
            )
        if self.activity_style:
            parts.append(f"Activity style: {_ja(self.activity_style)}")
        if self.cuisine_types:
            parts.append(
                f"Cuisine: {', '.join(_ja(c) for c in self.cuisine_types)}"
            )
        if self.dining_styles:
            parts.append(
                f"Dining style: {', '.join(_ja(d) for d in self.dining_styles)}"
            )
        if self.budget_preference:
            parts.append(f"Budget: {_ja(self.budget_preference)}")
        if self.dietary_restrictions:
            parts.append(
                f"Dietary restrictions: {', '.join(_ja(d) for d in self.dietary_restrictions)}"
            )
        if self.beverage_types:
            parts.append(
                f"Beverages: {', '.join(_ja(b) for b in self.beverage_types)}"
            )
        if self.activity_interests:
            parts.append(
                f"Interests: {', '.join(_ja(a) for a in self.activity_interests)}"
            )
        if self.custom_notes:
            parts.append(f"Notes: {self.custom_notes}")
        return "\n".join(parts) if parts else "No preferences set"
