"""
Rule-based itinerary validator. No LLM calls.

Hard errors  → caller should retry planning (up to MAX_RETRIES times).
Soft warnings → appended to itinerary["warnings"], no retry triggered.
"""

import re
from datetime import datetime, time
from dataclasses import dataclass, field

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> datetime | None:
    """Parse wall-clock strings like '10:00 AM', '9:30 PM' → datetime (date portion ignored)."""
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            return datetime.strptime(t.strip().upper(), fmt)
        except ValueError:
            continue
    return None


def _parse_duration(travel_str: str) -> float:
    """
    Parse travel_from_previous strings like '20min drive', '1h 15min walk' → hours (float).
    Returns 0.0 if unparseable.
    """
    if not travel_str:
        return 0.0
    hours = 0.0
    h_match = re.search(r"(\d+)\s*h", travel_str, re.IGNORECASE)
    m_match = re.search(r"(\d+)\s*min", travel_str, re.IGNORECASE)
    if h_match:
        hours += int(h_match.group(1))
    if m_match:
        hours += int(m_match.group(1)) / 60
    return hours


def _all_place_ids(brief: dict) -> set[str]:
    """Collect every place_id present in the research brief."""
    ids = set()
    for place in brief.get("places", []):
        if place.get("place_id"):
            ids.add(place["place_id"])
    for hotel in brief.get("hotels", []):
        if hotel.get("place_id"):
            ids.add(hotel["place_id"])
    return ids


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    hard_errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.hard_errors) == 0

    def error_message(self) -> str:
        """Formatted string fed back to the planning agent on retry."""
        return "VALIDATION FAILED:\n" + "\n".join(f"- {e}" for e in self.hard_errors)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ItineraryValidator:

    def validate(self, itinerary: dict, req, brief: dict) -> ValidationResult:
        result = ValidationResult()

        # ── 0. Parse guard ──────────────────────────────────────────────────
        if not itinerary or itinerary.get("parse_error"):
            result.hard_errors.append(
                "Output is not valid JSON or could not be parsed. Re-output the complete itinerary JSON."
            )
            return result

        days = itinerary.get("days", [])
        budget = req.budget
        brief_ids = _all_place_ids(brief)

        # ── 1. Budget check ──────────────────────────────────────────────────
        breakdown = itinerary.get("budget_breakdown", {})
        total_cost = breakdown.get("total", itinerary.get("trip_summary", {}).get("total_cost", 0))
        if total_cost > budget:
            over = total_cost - budget
            result.hard_errors.append(
                f"Total cost ${total_cost:.0f} exceeds budget ${budget:.0f} by ${over:.0f}. "
                f"Reduce costs across days to fit within budget."
            )

        # ── 2. Per-day checks ────────────────────────────────────────────────
        cumulative_day_cost = 0.0
        seen_restaurants: dict[str, int] = {}  # name → first day_number

        for day in days:
            day_num = day.get("day_number", "?")
            activities = day.get("activities", [])

            # 2a. Every day has ≥1 activity (exclude pure logistics)
            non_transit = [a for a in activities if a.get("type") not in ("transit", "checkin", "checkout")]
            if len(non_transit) < 1:
                result.hard_errors.append(
                    f"Day {day_num} has no activities. Add at least one attraction or meal."
                )

            # 2b. Every day has ≥2 meals (meal, dining, restaurant, cafe, breakfast, lunch, dinner)
            MEAL_TYPES = {"meal", "dining", "restaurant", "cafe", "breakfast", "lunch", "dinner", "food"}
            meals = [a for a in activities if a.get("type", "").lower() in MEAL_TYPES]
            real_meals = [m for m in meals if m.get("source") != "insufficient_data"]
            if len(meals) < 2:
                result.hard_errors.append(
                    f"Day {day_num} has only {len(meals)} meal(s). Each day requires at least 2."
                )
            elif len(real_meals) < 1:
                result.soft_warnings.append(
                    f"Day {day_num} has no meals from the research brief — all meals are placeholders."
                )

            # 2c. place_id validity (soft — model may use names instead of Google IDs)
            for act in activities:
                pid = act.get("place_id", "")
                if pid and pid not in ("none", "") and pid not in brief_ids and act.get("source") != "insufficient_data":
                    result.soft_warnings.append(
                        f"Day {day_num} activity '{act.get('name')}' has unrecognised place_id '{pid}'."
                    )

            # 2d. Time overlap + feasibility check
            timed = []
            for act in activities:
                t = _parse_time(act.get("time", ""))
                if t:
                    timed.append((t, act))

            timed.sort(key=lambda x: x[0])

            for i in range(1, len(timed)):
                prev_t, prev_act = timed[i - 1]
                curr_t, curr_act = timed[i]

                prev_end_hours = prev_t.hour + prev_t.minute / 60 + (prev_act.get("duration_hours") or 0)
                travel_hours = _parse_duration(curr_act.get("travel_from_previous", ""))
                required_start = prev_end_hours + travel_hours
                actual_start = curr_t.hour + curr_t.minute / 60

                if actual_start < prev_end_hours:
                    # Overlap
                    result.hard_errors.append(
                        f"Day {day_num}: '{curr_act.get('name')}' at {curr_act.get('time')} "
                        f"overlaps with '{prev_act.get('name')}' which ends at "
                        f"{prev_end_hours:.1f}h. Adjust start times."
                    )
                elif actual_start < required_start:
                    # Not enough travel time
                    result.hard_errors.append(
                        f"Day {day_num}: '{curr_act.get('name')}' at {curr_act.get('time')} "
                        f"doesn't allow {travel_hours * 60:.0f}min travel from "
                        f"'{prev_act.get('name')}'. Push start time to at least "
                        f"{int(required_start)}:{int((required_start % 1) * 60):02d}."
                    )

            # 2e. Operating hours check (soft — only if hours data available from brief)
            brief_hours: dict[str, dict] = {}
            for p in brief.get("places", []) + brief.get("hotels", []):
                if p.get("place_id") and p.get("hours"):
                    brief_hours[p["place_id"]] = p["hours"]

            weekday = day.get("weekday", "")
            for act in activities:
                pid = act.get("place_id", "")
                act_time = _parse_time(act.get("time", ""))
                hours_info = brief_hours.get(pid, {})
                if not hours_info or not act_time or not weekday:
                    continue
                day_hours = hours_info.get(weekday) or hours_info.get("weekday_text", [])
                if isinstance(day_hours, str) and "Closed" in day_hours:
                    result.hard_errors.append(
                        f"Day {day_num}: '{act.get('name')}' is closed on {weekday}. "
                        f"Replace with an open alternative."
                    )

            # 2f. Soft: idle gap > 2h between consecutive activities
            for i in range(1, len(timed)):
                prev_t, prev_act = timed[i - 1]
                curr_t, curr_act = timed[i]
                prev_end = prev_t.hour + prev_t.minute / 60 + (prev_act.get("duration_hours") or 0)
                curr_start = curr_t.hour + curr_t.minute / 60
                gap = curr_start - prev_end
                if gap > 2:
                    result.soft_warnings.append(
                        f"Day {day_num}: {gap:.1f}h idle gap between "
                        f"'{prev_act.get('name')}' and '{curr_act.get('name')}'. "
                        f"Consider adding an activity."
                    )

            # 2g. Soft: same restaurant on multiple days
            for act in activities:
                if act.get("type") == "meal":
                    name = act.get("name", "")
                    if name in seen_restaurants:
                        result.soft_warnings.append(
                            f"'{name}' appears on both Day {seen_restaurants[name]} and Day {day_num}. "
                            f"Consider varying dining options."
                        )
                    else:
                        seen_restaurants[name] = day_num

            cumulative_day_cost += day.get("day_total", 0)

        # ── 3. Day count ─────────────────────────────────────────────────────
        if len(days) != req.num_days:
            result.hard_errors.append(
                f"Itinerary has {len(days)} day(s) but trip is {req.num_days} day(s). "
                f"Add or remove days accordingly."
            )

        # ── 4. Budget breakdown consistency ──────────────────────────────────
        declared_total = breakdown.get("total", 0)
        component_sum = sum(
            breakdown.get(k, 0)
            for k in ("accommodation", "food", "activities", "transport")
        )
        if abs(declared_total - component_sum) > 1:
            result.hard_errors.append(
                f"budget_breakdown.total ({declared_total}) does not equal "
                f"accommodation + food + activities + transport ({component_sum:.2f}). Fix the breakdown."
            )

        # ── 5. Soft: low budget utilization ──────────────────────────────────
        remaining = itinerary.get("trip_summary", {}).get("budget_remaining", budget - total_cost)
        if remaining > budget * 0.5:
            result.soft_warnings.append(
                f"Budget utilization is low — ${remaining:.0f} remaining. "
                f"Consider adding more activities or upgrading accommodation."
            )

        # ── 6. Soft: transport = $0 on a driving trip ─────────────────────────
        travel_times = brief.get("travel_times", [])
        has_driving = any(t.get("mode") == "driving" for t in travel_times)
        if has_driving and breakdown.get("transport", 0) == 0:
            result.soft_warnings.append(
                "Transport cost is $0 but trip includes driving legs. "
                "Add estimated fuel/toll costs (~$0.20/mile)."
            )

        # ── 7. Soft: food budget = $0 ─────────────────────────────────────────
        if breakdown.get("food", 0) == 0:
            result.soft_warnings.append(
                "Food budget is $0. Assign realistic meal costs ($15–60 per meal)."
            )

        return result
