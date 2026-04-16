"""
Tool handler implementations for the ResearchAgent.

Each function matches the OpenAI function-calling schema defined in research_agent.py.
API keys are read from environment variables (set in .env).

Backends:
  search_places   → Google Places Text Search API
  get_reviews     → Google Places Details API
  get_travel_time → Google Directions API
  get_weather     → Open-Meteo (free, no API key — forecast up to 16 days, climate API beyond)
  search_hotels   → Google Places (type=lodging)
"""

import os
import datetime
import requests
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_DIRECTIONS_API_KEY")
# Note: Open-Meteo needs no API key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _google_get(url: str, params: dict) -> dict:
    """GET wrapper — raises on HTTP error, returns parsed JSON."""
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_place(place: dict, fields: list[str] | None = None) -> dict:
    """Normalise a Google Places result into our schema dict."""
    hours_raw = place.get("opening_hours", {})
    hours = {}
    if hours_raw.get("weekday_text"):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, text in enumerate(hours_raw["weekday_text"]):
            hours[days[i]] = text
    return {
        "name":        place.get("name", ""),
        "rating":      place.get("rating"),
        "price_level": place.get("price_level"),
        "address":     place.get("vicinity") or place.get("formatted_address", ""),
        "place_id":    place.get("place_id", ""),
        "hours":       hours,
        "types":       place.get("types", []),
    }


# ---------------------------------------------------------------------------
# search_places
# ---------------------------------------------------------------------------

@traceable(name="search_places", run_type="tool")
def search_places(query: str, location: str, type: str) -> list[dict]:
    """
    Google Places Text Search.
    Returns up to 5 results: name, rating, price_level, address, place_id, hours, types.
    """
    data = _google_get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        {
            "query":  f"{query} in {location}",
            "type":   type,
            "key":    GOOGLE_KEY,
        },
    )
    results = data.get("results", [])[:5]
    parsed = []
    for place in results:
        p = _parse_place(place)
        # Text Search doesn't return opening_hours.weekday_text — need Details for that
        parsed.append(p)
    return parsed


# ---------------------------------------------------------------------------
# get_reviews
# ---------------------------------------------------------------------------

@traceable(name="get_reviews", run_type="tool")
def get_reviews(place_id: str) -> dict:
    """
    Google Places Details — fetches reviews, rating, user_ratings_total, and hours.
    Returns: {rating, user_ratings_total, reviews: [{text, rating}], hours: {weekday: str}}
    """
    data = _google_get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        {
            "place_id": place_id,
            "fields":   "name,rating,user_ratings_total,reviews,opening_hours",
            "key":      GOOGLE_KEY,
        },
    )
    result = data.get("result", {})

    reviews = [
        {
            "text":   r.get("text", ""),
            "rating": r.get("rating"),
        }
        for r in result.get("reviews", [])[:5]
    ]

    hours_raw = result.get("opening_hours", {})
    hours = {}
    if hours_raw.get("weekday_text"):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, text in enumerate(hours_raw["weekday_text"]):
            hours[days[i]] = text

    return {
        "rating":             result.get("rating"),
        "user_ratings_total": result.get("user_ratings_total"),
        "reviews":            reviews,
        "hours":              hours,
    }


# ---------------------------------------------------------------------------
# get_travel_time
# ---------------------------------------------------------------------------

@traceable(name="get_travel_time", run_type="tool")
def get_travel_time(origin: str, destination: str, mode: str) -> dict:
    """
    Google Directions API.
    Returns: {duration, distance, route_summary}
    mode: driving | transit | walking
    """
    data = _google_get(
        "https://maps.googleapis.com/maps/api/directions/json",
        {
            "origin":      origin,
            "destination": destination,
            "mode":        mode,
            "key":         GOOGLE_KEY,
        },
    )
    routes = data.get("routes", [])
    if not routes:
        return {"error": f"No route found from '{origin}' to '{destination}' by {mode}."}

    leg = routes[0]["legs"][0]
    return {
        "origin":        origin,
        "destination":   destination,
        "mode":          mode,
        "duration":      leg["duration"]["text"],
        "duration_secs": leg["duration"]["value"],
        "distance":      leg["distance"]["text"],
        "route_summary": routes[0].get("summary", ""),
    }


# ---------------------------------------------------------------------------
# get_weather  (Open-Meteo — free, no API key)
# ---------------------------------------------------------------------------

# WMO weather interpretation codes → human description
_WMO_CODES: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _geocode(location: str) -> tuple[float, float]:
    """
    Convert any location string to (lat, lon) using Google Geocoding API.
    Handles parks, neighborhoods, landmarks, cities — anything Google Maps knows.
    """
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": location, "key": GOOGLE_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Could not geocode location: {location!r} (status: {data.get('status')})")
    loc = results[0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


@traceable(name="get_weather", run_type="tool")
def get_weather(location: str, date: str) -> dict:
    """
    Open-Meteo weather for any date.
    - Within 16 days: uses the free forecast API (no key needed).
    - Beyond 16 days: uses the Open-Meteo climate API (CMIP6 ensemble mean).
    Returns: {location, date, temp_high, temp_low, conditions, precipitation_prob, source}
    """
    target = datetime.date.fromisoformat(date)
    today  = datetime.date.today()
    delta  = (target - today).days

    try:
        lat, lon = _geocode(location)
    except Exception as e:
        return {"location": location, "date": date, "error": f"Geocoding failed: {e}",
                "source": "error"}

    if delta <= 16:
        # ── Forecast API ──────────────────────────────────────────────────
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  lat,
                    "longitude": lon,
                    "daily":     "temperature_2m_max,temperature_2m_min,"
                                 "precipitation_probability_max,weather_code",
                    "temperature_unit": "fahrenheit",
                    "timezone":  "auto",
                    "start_date": date,
                    "end_date":   date,
                },
                timeout=10,
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            wmo   = int(daily["weather_code"][0])
            return {
                "location":           location,
                "date":               date,
                "temp_high":          round(daily["temperature_2m_max"][0]),
                "temp_low":           round(daily["temperature_2m_min"][0]),
                "conditions":         _WMO_CODES.get(wmo, f"weather code {wmo}"),
                "precipitation_prob": daily["precipitation_probability_max"][0],
                "source":             "forecast",
            }
        except Exception:
            pass   # fall through to climate API

    # ── Climate API (CMIP6 — works for any date 1950-2050) ───────────────
    try:
        resp = requests.get(
            "https://climate-api.open-meteo.com/v1/climate",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": date,
                "end_date":   date,
                "models":     "MRI_AGCM3_2_S",   # high-res global model
                "daily":      "temperature_2m_max,temperature_2m_min,"
                              "precipitation_sum",
                "temperature_unit": "fahrenheit",
            },
            timeout=15,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        precip_mm = daily.get("precipitation_sum", [0])[0] or 0
        return {
            "location":           location,
            "date":               date,
            "temp_high":          round(daily["temperature_2m_max"][0]),
            "temp_low":           round(daily["temperature_2m_min"][0]),
            "conditions":         "rainy" if precip_mm > 5 else "partly cloudy",
            "precipitation_prob": min(100, round(precip_mm * 10)),
            "source":             "climate_model",
        }
    except Exception as e:
        return {
            "location":           location,
            "date":               date,
            "temp_high":          None,
            "temp_low":           None,
            "conditions":         "unavailable",
            "precipitation_prob": None,
            "source":             "error",
            "error":              str(e),
        }


# ---------------------------------------------------------------------------
# search_hotels
# ---------------------------------------------------------------------------

@traceable(name="search_hotels", run_type="tool")
def search_hotels(location: str, checkin: str, checkout: str, max_price: float) -> list[dict]:
    """
    Google Places Text Search filtered to lodging type.
    (Amadeus test env not used — Google Places is sufficient for MVP.)
    Returns up to 5 hotels: name, rating, price_per_night (estimated), address, amenities.
    """
    data = _google_get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        {
            "query": f"hotels in {location}",
            "type":  "lodging",
            "key":   GOOGLE_KEY,
        },
    )
    results = data.get("results", [])

    hotels = []
    for place in results:
        # Google Places doesn't return price_per_night; map price_level to estimate
        price_level = place.get("price_level")
        estimated_price = {0: 60, 1: 100, 2: 160, 3: 250, 4: 400}.get(price_level, 150)
        if estimated_price > max_price:
            continue

        hotels.append({
            "name":            place.get("name", ""),
            "rating":          place.get("rating"),
            "price_per_night": estimated_price,
            "address":         place.get("vicinity") or place.get("formatted_address", ""),
            "place_id":        place.get("place_id", ""),
            "amenities":       [],   # requires Places Details call — done by get_reviews if needed
        })

        if len(hotels) >= 5:
            break

    return hotels


# ---------------------------------------------------------------------------
# Handler registry — passed to ResearchAgent
# ---------------------------------------------------------------------------

ALL_TOOL_HANDLERS = {
    "search_places":   search_places,
    "get_reviews":     get_reviews,
    "get_travel_time": get_travel_time,
    "get_weather":     get_weather,
    "search_hotels":   search_hotels,
}
