import json
import logging
from openai import OpenAI
from tracer import PipelineTracer
from langsmith import get_current_run_tree

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Itinerary output schema
# ---------------------------------------------------------------------------

ITINERARY_SCHEMA = {
    "type": "object",
    "required": [
        "trip_summary", "days", "budget_breakdown", "warnings",
        "alternatives", "insufficient_data",
    ],
    "properties": {
        "trip_summary": {
            "type": "object",
            "required": ["destination", "from_location", "dates", "total_cost", "budget_remaining", "party_size"],
            "properties": {
                "destination":      {"type": "string"},
                "from_location":    {"type": "string"},
                "dates":            {"type": "string", "description": "Human-readable date range, e.g. 'June 15-17, 2026'"},
                "total_cost":       {"type": "number"},
                "budget_remaining": {"type": "number"},
                "party_size":       {"type": "integer"},
            },
        },
        "days": {
            "type": "array",
            "description": "One entry per trip day, in order",
            "items": {
                "type": "object",
                "required": ["day_number", "date", "weekday", "theme", "weather", "activities", "accommodation", "day_total"],
                "properties": {
                    "day_number":  {"type": "integer"},
                    "date":        {"type": "string"},
                    "weekday":     {"type": "string"},
                    "theme":       {"type": "string"},
                    "weather":     {"type": "string", "description": "e.g. '72°F, partly cloudy'"},
                    "day_total":   {"type": "number"},
                    "accommodation": {
                        "type": "object",
                        "required": ["name", "place_id", "cost_per_night"],
                        "properties": {
                            "name":           {"type": "string"},
                            "place_id":       {"type": "string"},
                            "cost_per_night": {"type": "number"},
                        },
                    },
                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["time", "type", "name", "place_id", "duration_hours", "cost_total"],
                            "properties": {
                                "time":                 {"type": "string", "description": "Wall-clock time, e.g. '10:00 AM'"},
                                "type":                 {"type": "string", "description": "meal | activity | transit | checkin"},
                                "name":                 {"type": "string"},
                                "place_id":             {"type": "string"},
                                "source_url":           {"type": "string", "description": "Google Maps URL: https://www.google.com/maps/place/?q=place_id:{place_id}. Use empty string if place_id is 'none'."},
                                "duration_hours":       {"type": "number"},
                                "cost_per_person":      {"type": "number"},
                                "cost_total":           {"type": "number"},
                                "travel_from_previous": {"type": "string", "description": "e.g. '20min drive'"},
                                "notes":                {"type": "string"},
                                "source":               {"type": "string", "enum": ["research_brief", "wikivoyage", "insufficient_data"]},
                            },
                        },
                    },
                },
            },
        },
        "budget_breakdown": {
            "type": "object",
            "required": ["accommodation", "food", "activities", "transport", "total"],
            "properties": {
                "accommodation": {"type": "number"},
                "food":          {"type": "number"},
                "activities":    {"type": "number"},
                "transport":     {"type": "number"},
                "total":         {"type": "number"},
            },
        },
        "reasoning": {
            "type": "object",
            "description": "Natural language explanation of key planning decisions",
            "properties": {
                "overview":        {"type": "string", "description": "2-3 sentence summary of the trip plan"},
                "budget_logic":    {"type": "string", "description": "How budget was allocated and why"},
                "hotel_choice":    {"type": "string", "description": "Why this hotel was chosen"},
                "day_by_day":      {"type": "array", "items": {"type": "string"}, "description": "One sentence per day explaining the theme and key choices"},
                "weather_impact":  {"type": "string", "description": "How weather influenced activity sequencing"},
                "constraints_met": {"type": "string", "description": "How user constraints and interests were satisfied"},
            },
        },
        "warnings":  {"type": "array", "items": {"type": "string"}},
        "alternatives": {
            "type": "object",
            "properties": {
                "rain_day_swaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "outdoor_activity_name":   {"type": "string"},
                            "outdoor_place_id":        {"type": "string"},
                            "indoor_alternative_name": {"type": "string"},
                            "indoor_place_id":         {"type": "string"},
                            "notes":                   {"type": "string"},
                        },
                    },
                },
            },
        },
        "insufficient_data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day_number":  {"type": "integer"},
                    "time":        {"type": "string"},
                    "reason":      {"type": "string"},
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Travel Planning Agent. Your job is to build a complete, realistic, cost-accurate \
day-by-day itinerary using ONLY the information in the research brief provided to you. \
You have no tools. Do not invent places, restaurants, or attractions that are not in the brief.

## Reasoning approach (think before you write)
Before producing the JSON output, reason through the following in a private scratchpad:

1. BUDGET ALLOCATION
   - Accommodation total = cost_per_night × number of nights. Subtract from total budget first.
   - Transport: estimate gas/tolls for driving legs at $0.20/mile. Add to budget_breakdown.transport.
     e.g. 480 miles driving = ~$96 transport cost. Never leave transport = $0 for road trips.
   - Meals: budget $15–25 per person per breakfast, $20–35 per lunch, $30–60 per dinner.
     Assign realistic cost_total to every meal activity. Never leave food cost_total = 0.
   - Activities: use any entry fees from the brief. National parks = $35/vehicle.
   - Remaining budget after the above = discretionary spend.

2. MANDATORY STRUCTURE
   - Day 1 MUST start with a "transit" activity for the origin→destination drive/flight:
     type="transit", name="Drive from [origin] to [destination]", cost_total = transport cost estimate.
   - Last day MUST end with a "transit" activity for the return leg.
   - Every day MUST have at least 2 meals (breakfast + dinner minimum, add lunch if time allows).
   - Every meal MUST have a non-zero cost_total reflecting realistic restaurant prices.

3. GEOGRAPHIC CLUSTERING
   - Group nearby places on the same day to minimize travel time.
   - Use travel_times from the brief to sequence activities logically.

4. TIME SEQUENCING
   - Day 1: depart origin early (6–7 AM), arrive destination mid-afternoon, check in, dinner.
   - Middle days: full day — breakfast, activities, lunch, more activities, dinner.
   - Last day: breakfast, morning activity, lunch, check-out, drive home (depart by 1–2 PM).
   - Respect typical opening hours. Account for weather — move outdoor activities to clear days.

5. CONSTRAINT SATISFACTION
   - Dietary restrictions: only include restaurants that meet them.
   - Mobility constraints: check place types for accessibility.
   - No-fly / transport constraints: use travel_times mode data.

6. INSUFFICIENT DATA
   - If you cannot fill a slot from the brief, do NOT invent a place.
   - Add an entry to `insufficient_data` with the day, slot, and reason.
   - Set that activity's source to "insufficient_data".

## Output rules
- Output ONLY a single valid JSON object. No prose, no markdown fences.
- Use EXACTLY these top-level keys: trip_summary, days, budget_breakdown, reasoning, warnings, alternatives, insufficient_data.
- Use wall-clock times for activities (e.g. "10:00 AM").
- Every activity must include: time, type, name, place_id, source_url, duration_hours, cost_total, source.
- type must be one of: transit, checkin, checkout, meal, activity, hiking.
- Every place_id must come from the research brief. Use "none" if unknown — never omit the field.
- source_url: if place_id is not "none", set to "https://www.google.com/maps/place/?q=place_id:{place_id}" (substitute the actual place_id). If place_id is "none", set to "".
- source must be one of: "research_brief", "wikivoyage", "insufficient_data".
- budget_breakdown.total = accommodation + food + activities + transport.
- trip_summary.total_cost = budget_breakdown.total.
- trip_summary.budget_remaining = budget_usd - trip_summary.total_cost.
- day_total per day = sum of that day's activity cost_total values + accommodation cost_per_night.
- If budget_remaining would be negative, reduce costs until it is >= 0.
- days[] must use field "activities" (NOT "itinerary" or any other name).
- Each day must have fields: day_number (int), date, weekday, theme, weather, activities, accommodation, day_total.
- Populate alternatives.rain_day_swaps with indoor alternatives for outdoor activities.
- Populate reasoning with natural language explanations: overview (2-3 sentences), budget_logic, hotel_choice, day_by_day (one sentence per day), weather_impact, constraints_met.

## Exact output structure (follow field names precisely):
{
  "trip_summary": {"destination": "...", "from_location": "...", "dates": "...", "total_cost": 0, "budget_remaining": 0, "party_size": 1},
  "days": [
    {
      "day_number": 1, "date": "YYYY-MM-DD", "weekday": "Friday", "theme": "...", "weather": "...", "day_total": 0,
      "accommodation": {"name": "...", "place_id": "...", "cost_per_night": 150},
      "activities": [
        {"time": "6:00 AM", "type": "transit", "name": "Drive from [origin] to [destination]", "place_id": "none", "source_url": "", "duration_hours": 8, "cost_total": 96, "travel_from_previous": "", "notes": "~480 miles driving", "source": "research_brief"},
        {"time": "3:00 PM", "type": "checkin", "name": "Hotel Name", "place_id": "ChIJXXXXX", "source_url": "https://www.google.com/maps/place/?q=place_id:ChIJXXXXX", "duration_hours": 1, "cost_total": 0, "travel_from_previous": "", "notes": "", "source": "research_brief"},
        {"time": "5:00 PM", "type": "meal", "name": "Dinner Restaurant", "place_id": "ChIJYYYYY", "source_url": "https://www.google.com/maps/place/?q=place_id:ChIJYYYYY", "duration_hours": 1.5, "cost_total": 45, "travel_from_previous": "10min drive", "notes": "Seafood dinner", "source": "research_brief"}
      ]
    }
  ],
  "budget_breakdown": {"accommodation": 0, "food": 0, "activities": 0, "transport": 0, "total": 0},
  "reasoning": {
    "overview": "2-3 sentence summary of the trip plan and key choices made.",
    "budget_logic": "How the budget was allocated across accommodation, food, transport, and activities.",
    "hotel_choice": "Why this hotel was chosen over alternatives.",
    "day_by_day": ["Day 1: Travel day — drive from X to Y, check in, dinner near hotel.", "Day 2: ..."],
    "weather_impact": "How weather forecasts influenced activity ordering and day themes.",
    "constraints_met": "How the user's interests and constraints shaped the plan."
  },
  "warnings": [],
  "alternatives": {"rain_day_swaps": [{"outdoor_activity_name": "...", "outdoor_place_id": "...", "indoor_alternative_name": "...", "indoor_place_id": "...", "notes": "..."}]},
  "insufficient_data": []
}

## On retry (validation errors provided)
- Read each validation error carefully.
- Fix only what is wrong. Do not restructure the whole itinerary.
- Re-output the complete corrected JSON.
"""

# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------

def _salvage_truncated_json(text: str) -> dict | None:
    """
    Try to parse truncated JSON by closing any unclosed brackets/braces.
    Returns a dict if successful, None otherwise.
    """
    import re
    # Find where the last complete top-level value ends by progressively trimming
    # trailing incomplete tokens and closing open structures.
    stack = []
    in_string = False
    escape = False
    last_good = 0

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    last_good = i + 1
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
                if not stack:
                    last_good = i + 1

    if not stack:
        # Nothing was truncated — standard parse already failed for other reasons
        return None

    # Close all unclosed structures
    closing = ""
    for ch in reversed(stack):
        closing += '}' if ch == '{' else ']'

    candidate = text[:last_good] if last_good else text.rstrip().rstrip(',')
    # Try appending closers
    for trim in [text, text.rstrip(), text.rstrip().rstrip(',')]:
        for suffix in [closing, closing.lstrip(']').lstrip('}')]:
            try:
                return json.loads(trim + suffix)
            except json.JSONDecodeError:
                continue
    return None


class PlanningAgent:
    def __init__(self, client: OpenAI, tracer: PipelineTracer | None = None, model: str = "gpt-5.2"):
        self.client = client
        self.model = model
        self.tracer: PipelineTracer | None = tracer

    def _slim_brief(self, brief: dict) -> dict:
        """
        Strip high-cardinality fields that aren't needed for planning,
        to keep the prompt within the model's context window.
        """
        import copy
        b = copy.deepcopy(brief)

        # Truncate Wikivoyage passages to 200 chars each, keep only top 2
        for d in b.get("destinations", [])[:2]:
            if isinstance(d.get("passage"), str):
                d["passage"] = d["passage"][:200]
        b["destinations"] = b.get("destinations", [])[:2]

        # Drop reviews from places — keep place_id so links work
        for p in b.get("places", []):
            p.pop("reviews", None)
            for key in list(p.keys()):
                if key not in ("name", "type", "rating", "price_level", "address", "place_id", "hours"):
                    p.pop(key, None)

        # Keep top 8 places by rating (more = better place_id coverage for links)
        b["places"] = sorted(
            b.get("places", []), key=lambda x: x.get("rating") or 0, reverse=True
        )[:8]

        # Keep top 3 hotels, strip hours (not needed for hotels)
        b["hotels"] = b.get("hotels", [])[:3]
        for h in b["hotels"]:
            for key in list(h.keys()):
                if key not in ("name", "rating", "price_per_night", "address", "place_id"):
                    h.pop(key, None)

        # Drop cache_report, partial flag noise
        b.pop("cache_report", None)

        return b

    def _build_user_message(
        self,
        req,
        brief: dict,
        validation_errors: list[str] | None = None,
    ) -> str:
        parts = [
            "## User Request",
            f"- From: {req.from_location}",
            f"- To: {req.location}",
            f"- Dates: {req.start_date.isoformat()} → {req.end_date.isoformat()} ({req.num_days} days)",
            f"- Budget: ${req.budget} USD total",
            f"- Interests: {req.interests or 'general sightseeing'}",
            f"- Constraints: {req.constraints or 'none'}",
            "",
            "## Trip Days (with weekdays for open/close awareness)",
        ]
        for d in req.trip_days:
            parts.append(f"  Day {d['day_number']}: {d['weekday']} {d['date'].isoformat()}")

        parts += [
            "",
            "## Research Brief",
            json.dumps(self._slim_brief(brief), indent=2, default=str),
        ]

        if validation_errors:
            parts += [
                "",
                "## Validation Errors (fix these)",
                *[f"- {e}" for e in validation_errors],
            ]

        parts.append(
            "\nNow reason through budget, geography, timing, and constraints, "
            "then output the itinerary JSON."
        )
        return "\n".join(parts)

    def _call_and_trace(self, stage: str, messages: list, retry_count: int = 0) -> tuple[dict, dict]:
        """Make one LLM call, emit a span, return (parsed_result, span_entry)."""
        span = self.tracer.new_span(stage) if self.tracer else None

        parent = get_current_run_tree()
        if parent is not None:
            child = parent.create_child(
                name="PlanningAgent.llm_call",
                run_type="llm",
                inputs={"stage": stage, "model": self.model, "retry_count": retry_count},
            )
            child.post()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            if parent is not None:
                child.end(error=str(e))
                child.patch()
            raise
        raw = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            log.warning("PlanningAgent: output truncated by max_tokens (finish_reason=length)")
        log.debug("PlanningAgent raw output (%d chars): %s", len(raw), raw[:200])

        if parent is not None:
            child.end(outputs={"finish_reason": finish_reason,
                               "usage": response.usage.model_dump() if response.usage else {}})
            child.patch()

        if span:
            if response.usage:
                span.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
            span.retry_count = retry_count
            entry = span.emit()
            if self.tracer:
                self.tracer._spans.append(entry)

        return self._parse(raw)

    def plan(self, req, brief: dict, validation_errors: list[str] | None = None) -> dict:
        """
        Generate an itinerary from the research brief.
        On retry, pass validation_errors to guide the model's corrections.
        Returns a parsed itinerary dict.
        """
        retry_count = 1 if validation_errors else 0
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(req, brief, validation_errors)},
        ]
        return self._call_and_trace("planning_agent", messages, retry_count=retry_count)

    def refine(self, existing: dict, feedback: str) -> dict:
        """Refine an existing itinerary based on user follow-up feedback."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "## Existing Itinerary\n"
                    f"{json.dumps(existing, indent=2, default=str)}\n\n"
                    "## User Feedback\n"
                    f"{feedback}\n\n"
                    "Apply the feedback and re-output the complete corrected itinerary JSON."
                ),
            },
        ]
        return self._call_and_trace("planning_agent_refine", messages)

    def _parse(self, raw: str) -> dict:
        """Parse JSON from model output, stripping think blocks and markdown fences."""
        import re
        import logging
        log = logging.getLogger(__name__)

        if not raw.strip():
            log.error("PlanningAgent: model returned empty response")
            return {"raw_output": "", "parse_error": True}

        # Strip <think>...</think> blocks emitted by Hermes/Qwen reasoning models
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` fence
        match = re.search(r"```json\s*([\s\S]+?)\s*```", cleaned)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding the outermost { ... } block
        brace_match = re.search(r"\{[\s\S]+\}", cleaned)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: output was truncated mid-stream — try salvaging with json5 or
        # by closing open brackets so the known-good fields are preserved
        salvaged = _salvage_truncated_json(cleaned)
        if salvaged:
            salvaged["_truncated"] = True
            log.warning("PlanningAgent: JSON was truncated; salvaged partial output")
            return salvaged

        log.error("PlanningAgent: failed to parse JSON. First 500 chars: %s", raw[:500])
        return {"raw_output": raw, "parse_error": True}
