import json
import logging
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from api_cache import APICache
from tracer import PipelineTracer, Span
from langsmith import get_current_run_tree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema (research brief)
# ---------------------------------------------------------------------------

RESEARCH_BRIEF_SCHEMA = {
    "type": "object",
    "required": [
        "destination",
        "from_location",
        "travel_dates",
        "budget_usd",
        "destinations",
        "places",
        "hotels",
        "weather",
        "travel_times",
        "warnings",
        "partial",
    ],
    "properties": {
        "destination":    {"type": "string"},
        "from_location":  {"type": "string"},
        "travel_dates":   {"type": "object", "properties": {
            "start": {"type": "string"},
            "end":   {"type": "string"},
            "days":  {"type": "integer"},
        }},
        "budget_usd":     {"type": "number"},
        "destinations":   {
            "type": "array",
            "description": "Wikivoyage passages relevant to the trip",
            "items": {
                "type": "object",
                "properties": {
                    "passage":     {"type": "string"},
                    "destination": {"type": "string"},
                    "section":     {"type": "string"},
                    "source":      {"type": "string"},
                },
            },
        },
        "places": {
            "type": "array",
            "description": "Attractions, restaurants, and activities found via Places API",
            "items": {
                "type": "object",
                "properties": {
                    "name":        {"type": "string"},
                    "type":        {"type": "string"},
                    "rating":      {"type": "number"},
                    "price_level": {"type": "integer"},
                    "address":     {"type": "string"},
                    "place_id":    {"type": "string"},
                    "hours":       {"type": "object"},
                    "reviews":     {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "hotels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":            {"type": "string"},
                    "rating":          {"type": "number"},
                    "price_per_night": {"type": "number"},
                    "address":         {"type": "string"},
                    "amenities":       {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "weather": {
            "type": "array",
            "description": "One entry per trip day",
            "items": {
                "type": "object",
                "properties": {
                    "date":                    {"type": "string"},
                    "weekday":                 {"type": "string"},
                    "temp_high":               {"type": "number"},
                    "temp_low":                {"type": "number"},
                    "conditions":              {"type": "string"},
                    "precipitation_prob":      {"type": "number"},
                },
            },
        },
        "travel_times": {
            "type": "array",
            "description": "Origin-to-destination and key in-trip legs",
            "items": {
                "type": "object",
                "properties": {
                    "origin":       {"type": "string"},
                    "destination":  {"type": "string"},
                    "mode":         {"type": "string"},
                    "duration":     {"type": "string"},
                    "distance":     {"type": "string"},
                    "summary":      {"type": "string"},
                },
            },
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any issues encountered during research (failed tools, missing data, etc.)",
        },
        "partial": {
            "type": "boolean",
            "description": "True if max tool calls were hit before research was complete",
        },
    },
}

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_destinations",
            "description": (
                "Search a FAISS index over Wikivoyage chunks for factual travel information "
                "about a destination. Use this to gather background knowledge, must-sees, "
                "local tips, and neighbourhood overviews. Do NOT rely on your own knowledge."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'things to do in Portland Maine'",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": (
                "Search Google Places for real, current businesses: attractions, restaurants, "
                "museums, parks, etc. Returns up to 5 results. Call multiple times with different "
                "types to cover the trip's interests."
            ),
            "parameters": {
                "type": "object",
                "required": ["query", "location", "type"],
                "properties": {
                    "query":    {"type": "string", "description": "Search query, e.g. 'seafood restaurant'"},
                    "location": {"type": "string", "description": "City or area to search within"},
                    "type": {
                        "type": "string",
                        "description": "Place type: restaurant | museum | park | tourist_attraction | bar | cafe | store",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": (
                "Fetch up to 5 user reviews and the overall rating for a specific place. "
                "Call this after search_places for any place the planner is likely to include."
            ),
            "parameters": {
                "type": "object",
                "required": ["place_id"],
                "properties": {
                    "place_id": {"type": "string", "description": "Google Places place_id from search_places result"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_travel_time",
            "description": (
                "Get travel time, distance, and route summary between two locations. "
                "REQUIRED calls: (1) origin→destination outbound, (2) destination→origin return, "
                "(3) between each pair of top attractions/restaurants within the destination "
                "so the planner knows local transfer times. Use driving for inter-city legs, "
                "walking or driving for local legs."
            ),
            "parameters": {
                "type": "object",
                "required": ["origin", "destination", "mode"],
                "properties": {
                    "origin":      {"type": "string"},
                    "destination": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "transit", "walking"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get weather forecast or historical averages for a location on a specific date. "
                "Call once per trip day so the planner can account for rain, heat, etc."
            ),
            "parameters": {
                "type": "object",
                "required": ["location", "date"],
                "properties": {
                    "location": {"type": "string"},
                    "date":     {"type": "string", "description": "ISO date string, e.g. '2026-04-20'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": (
                "Search for hotels at the destination within a price ceiling. "
                "Call once with the full stay dates."
            ),
            "parameters": {
                "type": "object",
                "required": ["location", "checkin", "checkout", "max_price"],
                "properties": {
                    "location":  {"type": "string"},
                    "checkin":   {"type": "string", "description": "ISO date string"},
                    "checkout":  {"type": "string", "description": "ISO date string"},
                    "max_price": {"type": "number", "description": "Max price per night in USD"},
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Travel Research Agent. Your ONLY job is to gather verified facts about a trip \
and return them as a structured JSON research brief. You do NOT plan itineraries, \
make recommendations, or write prose responses.

## Rules
1. Use tools for every factual claim. Never rely on your own knowledge.
2. Do NOT produce an itinerary or suggest a schedule — that is the Planner's job.
3. Call tools in this efficient order to cover the most ground first:
   a. search_destinations (Wikivoyage background)
   b. get_travel_time — call ALL of these legs:
      - origin → destination (outbound, Day 1)
      - destination → origin (return, last day)
      - between each pair of top places within the destination (local transfers)
      Use mode=driving for inter-city, mode=walking or driving for local legs.
   c. search_places for EACH of these separately — do not skip any:
      - user interests (e.g. hiking trails, museums)
      - seafood restaurants
      - breakfast cafes or brunch spots
      - dinner restaurants
   d. search_hotels
   e. get_weather — once per trip day
   f. get_reviews — only for the top 2-3 places, not every result
4. You MUST find at least 3 restaurants covering breakfast, lunch, and dinner options.
5. Limit get_reviews to 3 calls maximum — prioritise breadth over depth.
5. If a tool call fails, record a warning and continue — do not stop.
6. After completing the above OR when approaching the tool call limit, \
output ONLY a valid JSON object matching the research brief schema. No extra text, no markdown.

## Output format
Return a single JSON object. Fields:
- destination, from_location, travel_dates, budget_usd
- destinations: Wikivoyage passages
- places: verified places with reviews merged in
- hotels: lodging options
- weather: one entry per day
- travel_times: origin→destination plus any major legs
- warnings: list of issues encountered
- partial: true if tool limit was reached before research was complete
"""

# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS = 25


def _extract_tool_calls_from_content(content: str) -> list[dict]:
    """
    Qwen3 emits tool calls as <tool_call>{...}</tool_call> in the message content
    instead of the OpenAI tool_calls field. Parse them out.
    Returns list of {name, arguments} dicts.
    """
    import re
    calls = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


def _dispatch_tool(name: str, args: dict, tool_handlers: dict, cache: APICache, span: Span) -> str:
    """
    Call the registered handler through the API cache.
    Records latency and cache-hit status on the active span.
    Creates a LangSmith child run if an active trace exists.
    Returns JSON string.
    """
    handler = tool_handlers.get(name)
    if handler is None:
        span.add_tool_call(name, latency_ms=0, cache_hit=False)
        return json.dumps({"error": f"Tool '{name}' is not implemented yet."})

    # LangSmith child span — context is propagated via contextvars.copy_context()
    parent = get_current_run_tree()
    ls_child = None
    if parent is not None:
        ls_child = parent.create_child(
            name=f"tool:{name}",
            run_type="tool",
            inputs={"args": args},
        )
        ls_child.post()

    try:
        key = cache.make_key(name, args)
        hit, _ = cache.get(key)
        t0 = time.monotonic()
        result = cache.cached_call(api_name=name, params=args, fn=handler)
        latency = (time.monotonic() - t0) * 1000
        span.add_tool_call(name, latency_ms=latency, cache_hit=hit)
        if ls_child is not None:
            ls_child.end(outputs={"cache_hit": hit, "result_keys": list(result.keys()) if isinstance(result, dict) else []})
            ls_child.patch()
        return json.dumps(result)
    except Exception as exc:
        span.add_tool_call(name, latency_ms=0, cache_hit=False)
        if ls_child is not None:
            ls_child.end(error=str(exc))
            ls_child.patch()
        return json.dumps({"error": str(exc)})


class ResearchAgent:
    def __init__(
        self,
        client: OpenAI,
        tool_handlers: dict | None = None,
        cache: APICache | None = None,
        tracer: PipelineTracer | None = None,
        model: str = "gpt-5.2",
    ):
        self.client = client
        self.model = model
        self.tool_handlers: dict = tool_handlers or {}
        self.cache: APICache = cache or APICache()
        self.tracer: PipelineTracer | None = tracer

    def _trim_messages(self, messages: list) -> list:
        """
        Drop older tool result messages to stay within context limits.
        Always keeps: system message, first user message, last 6 tool results.
        Each trim removes the oldest assistant+tool pair.
        """
        system = [m for m in messages if m["role"] == "system"]
        first_user = next((m for m in messages if m["role"] == "user"), None)
        rest = [m for m in messages if m not in system and m is not first_user]

        # Drop oldest pairs of (assistant tool_calls + tool results) until short enough
        # Keep at least the last 6 messages so the model has recent context
        while len(rest) > 6:
            # Find first assistant message with tool_calls and its following tool messages
            for i, m in enumerate(rest):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    # Remove this assistant message and all immediately following tool messages
                    j = i + 1
                    while j < len(rest) and rest[j].get("role") == "tool":
                        j += 1
                    rest = rest[:i] + rest[j:]
                    break
            else:
                break  # no more assistant+tool pairs to trim

        trimmed = system + ([first_user] if first_user else []) + rest
        logger.warning("ResearchAgent: trimmed message history from %d → %d messages", len(messages), len(trimmed))
        return trimmed

    def _llm_call(self, messages, tools, tool_choice):
        """Make one LLM call, with automatic context-length trimming on 400 errors."""
        def _do_call(msgs):
            return self.client.chat.completions.create(
                model=self.model, messages=msgs, tools=tools, tool_choice=tool_choice,
            )

        parent = get_current_run_tree()
        child = None
        if parent is not None:
            child = parent.create_child(
                name="ResearchAgent.llm_call",
                run_type="llm",
                inputs={"messages": [{"role": m["role"], "content": str(m.get("content", ""))[:200]} for m in messages[-3:]],
                        "tool_choice": tool_choice, "model": self.model},
            )
            child.post()

        try:
            try:
                response = _do_call(messages)
            except Exception as e:
                # Retry with trimmed history if context length exceeded
                if "maximum context length" in str(e) or "400" in str(e):
                    logger.warning("ResearchAgent: context too long, trimming and retrying")
                    trimmed = self._trim_messages(messages)
                    response = _do_call(trimmed)
                else:
                    raise

            if child is not None:
                child.end(outputs={"finish_reason": response.choices[0].finish_reason,
                                   "usage": response.usage.model_dump() if response.usage else {}})
                child.patch()
            return response

        except Exception as e:
            if child is not None:
                child.end(error=str(e))
                child.patch()
            raise

    def run(self, req) -> dict:
        """
        Execute the ReAct research loop for a TripRequest.
        Returns a research brief dict (conforming to RESEARCH_BRIEF_SCHEMA).
        """
        span = self.tracer.new_span("research_agent") if self.tracer else Span("research_agent", "untraced")

        user_message = (
            f"Research a {req.num_days}-day trip to {req.location} "
            f"departing from {req.from_location}.\n"
            f"Dates: {req.start_date.isoformat()} to {req.end_date.isoformat()}.\n"
            f"Trip days: {[d['weekday'] + ' ' + d['date'].isoformat() for d in req.trip_days]}.\n"
            f"Budget: ${req.budget} USD total.\n"
            f"Interests: {req.interests or 'general sightseeing'}.\n"
            f"Constraints: {req.constraints or 'none'}.\n\n"
            "Gather all research using the available tools, then return the JSON research brief."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        tool_calls_made = 0
        warnings = []
        tokens_in = tokens_out = 0
        choice = None

        # Mandatory tools that must be called before the model can summarise
        REQUIRED_TOOLS = {"search_destinations", "get_travel_time", "search_places", "search_hotels", "get_weather"}
        called_tools: set[str] = set()

        while tool_calls_made < MAX_TOOL_CALLS:
            # Force tool use until all required tools have been called at least once
            remaining_required = REQUIRED_TOOLS - called_tools
            if remaining_required:
                tool_choice = "required"
            else:
                tool_choice = "auto"

            response = self._llm_call(messages, TOOLS, tool_choice)

            choice = response.choices[0]
            if response.usage:
                tokens_in += response.usage.prompt_tokens
                tokens_out += response.usage.completion_tokens

            # ── Normalise tool calls ────────────────────────────────────────
            # OpenAI-style: tool_calls field populated, finish_reason="tool_calls"
            # Qwen3-style:  tool_calls=[], content contains <tool_call>{...}</tool_call>
            raw_content = choice.message.content or ""
            openai_tcs  = choice.message.tool_calls or []
            qwen_tcs    = _extract_tool_calls_from_content(raw_content) if not openai_tcs else []
            has_tool_calls = bool(openai_tcs) or bool(qwen_tcs)

            # Append assistant message as plain dict (works for both OpenAI and vLLM)
            import re as _re
            clean_content = _re.sub(r"<tool_call>.*?</tool_call>", "", raw_content, flags=_re.DOTALL).strip()
            assistant_msg: dict = {"role": "assistant", "content": clean_content or None}
            if openai_tcs:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in openai_tcs
                ]
            messages.append(assistant_msg)

            if not has_tool_calls:
                break

            # Build a unified list: [{name, arguments, id}]
            unified = []
            for tc in openai_tcs:
                unified.append({
                    "id":        tc.id,
                    "name":      tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
            for i, tc in enumerate(qwen_tcs):
                unified.append({
                    "id":        f"qwen_tc_{tool_calls_made + i}",
                    "name":      tc["name"],
                    "arguments": tc["arguments"],
                })

            # Dispatch all tool calls in this batch concurrently.
            # Each thread gets its own copy of the current contextvars context so
            # get_current_run_tree() works correctly inside the threads.
            import contextvars

            for tc in unified:
                called_tools.add(tc["name"])

            with ThreadPoolExecutor(max_workers=len(unified)) as executor:
                # Submit in order, keep (tc, future) pairs so results stay ordered
                # Each submission gets a fresh context copy — a single Context object
                # cannot be entered by more than one thread simultaneously.
                tc_futures = [
                    (tc, executor.submit(
                        contextvars.copy_context().run,
                        _dispatch_tool,
                        tc["name"], tc["arguments"], self.tool_handlers, self.cache, span,
                    ))
                    for tc in unified
                ]

            # Append results in original submission order (deterministic message history)
            for tc, fut in tc_futures:
                tool_calls_made += 1
                result = fut.result()
                parsed = json.loads(result)
                if "error" in parsed:
                    warnings.append(f"Tool '{tc['name']}' failed: {parsed['error']}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })

            if tool_calls_made >= MAX_TOOL_CALLS:
                    warnings.append(f"Max tool calls ({MAX_TOOL_CALLS}) reached. Brief may be incomplete.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have reached the tool call limit. "
                            "Output the research brief JSON now using whatever data you have collected. "
                            "Set partial=true. Output ONLY valid JSON, no markdown."
                        ),
                    })
                    final_response = self._llm_call(messages, TOOLS, "none")
                    choice = final_response.choices[0]
                    if final_response.usage:
                        tokens_in += final_response.usage.prompt_tokens
                        tokens_out += final_response.usage.completion_tokens
                    break

        # Parse the model's final JSON output
        raw = (choice.message.content or "") if choice else ""
        try:
            brief = json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"```json\s*([\s\S]+?)\s*```", raw)
            if match:
                try:
                    brief = json.loads(match.group(1))
                except json.JSONDecodeError:
                    brief = {"raw_output": raw, "parse_error": True}
            else:
                brief = {"raw_output": raw, "parse_error": True}

        brief.setdefault("warnings", [])
        brief["warnings"].extend(warnings)
        brief.setdefault("partial", tool_calls_made >= MAX_TOOL_CALLS)
        brief["cache_report"] = self.cache.report()

        # Finalise and emit the span
        span.set_tokens(tokens_in, tokens_out)
        span.validation_result = "partial" if brief.get("partial") else "pass"
        entry = span.emit()
        if self.tracer:
            self.tracer._spans.append(entry)

        return brief
