import os
import datetime
from dataclasses import dataclass, field
from openai import OpenAI
from research_agent import ResearchAgent
from planning_agent import PlanningAgent
from validator import ItineraryValidator, MAX_RETRIES
from api_cache import APICache
from tracer import PipelineTracer, configure_logging
from vector_search import search_destinations
from tools import ALL_TOOL_HANDLERS
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()
configure_logging()


def _build_client() -> tuple[OpenAI, str]:
    """
    Returns (client, model_name).
    Uses local vLLM if VLLM_BASE_URL is set and the server is reachable,
    otherwise falls back to the OpenAI API.
    """
    base_url  = os.getenv("VLLM_BASE_URL", "").strip()
    model_name = os.getenv("VLLM_MODEL_NAME", "").strip()

    if base_url:
        import urllib.request
        try:
            urllib.request.urlopen(f"{base_url}/models", timeout=3)
            # Tunnel is up — use vLLM
            if not model_name:
                import json
                with urllib.request.urlopen(f"{base_url}/models", timeout=3) as r:
                    data = json.loads(r.read())
                model_name = data["data"][0]["id"]
            client = OpenAI(base_url=base_url, api_key="dummy")
            print(f"[LLM] Using vLLM at {base_url}  model={model_name}")
            return client, model_name
        except Exception:
            print("[LLM] vLLM tunnel not reachable — falling back to OpenAI API")

    model_name = model_name or "gpt-5.2"
    client = OpenAI()
    print(f"[LLM] Using OpenAI API  model={model_name}")
    return client, model_name

def _default_cache() -> APICache:
    return APICache()


@dataclass
class TripRequest:
    from_location: str
    location: str
    start_date: datetime.date
    end_date: datetime.date
    budget: float
    interests: str
    constraints: str

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def trip_days(self) -> list[dict]:
        """Returns one entry per trip day with date, day number, and weekday name."""
        days = []
        for i in range(self.num_days):
            date = self.start_date + datetime.timedelta(days=i)
            days.append({
                "day_number": i + 1,
                "date": date,
                "weekday": date.strftime("%A"),
            })
        return days


CONVERSATION_WINDOW = 3  # number of user-assistant turn pairs to retain


@dataclass
class SessionState:
    # Latest validated artefacts
    current_itinerary: dict | None = None
    research_brief: dict | None = None
    constraints: str | None = None          # raw constraints string from the latest request
    last_request: TripRequest | None = None
    api_cache: APICache = field(default_factory=_default_cache)

    # Sliding window: stores {"role": ..., "content": ...} dicts
    # Max CONVERSATION_WINDOW user-assistant pairs = 2 * CONVERSATION_WINDOW entries
    _history: list = field(default_factory=list)

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Append a turn and trim to the sliding window."""
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        # Keep only the last N pairs
        max_entries = CONVERSATION_WINDOW * 2
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]

    @property
    def conversation_history(self) -> list[dict]:
        return list(self._history)


class IntentRouter:
    def route(self, user_msg: str) -> str:
        """Return intent label: 'plan', 'refine', or 'research'."""
        msg = user_msg.lower()
        if any(k in msg for k in ("change", "update", "modify", "adjust", "instead")):
            return "refine"
        if any(k in msg for k in ("weather", "visa", "currency", "safety", "info")):
            return "research"
        return "plan"


class TripOrchestrator:
    def __init__(self):
        self.client, self.model_name = _build_client()
        self.router = IntentRouter()
        self.session = SessionState()  # api_cache lives here, persists across refinement calls
        self.tracer: PipelineTracer | None = None  # reset per form submission
        tool_handlers = {
            "search_destinations": search_destinations,
            **ALL_TOOL_HANDLERS,
        }
        self.research_agent = ResearchAgent(self.client, tool_handlers=tool_handlers, cache=self.session.api_cache, model=self.model_name)
        self.planning_agent = PlanningAgent(self.client, model=self.model_name)
        self.validator = ItineraryValidator()
        # Date/duration fields derived from the active request
        self.start_date: datetime.date | None = None
        self.end_date: datetime.date | None = None
        self.num_days: int | None = None
        self.trip_days: list[dict] | None = None

    @traceable(name="TravelPlanner", run_type="chain")
    def handle_form(self, req: TripRequest) -> dict:
        """Called when the user submits the trip planning form."""
        # Fresh tracer per form submission; wire into agents
        self.tracer = PipelineTracer()
        self.research_agent.tracer = self.tracer
        self.planning_agent.tracer = self.tracer

        # Update session state
        self.session.last_request = req
        self.session.constraints = req.constraints
        self.start_date = req.start_date
        self.end_date = req.end_date
        self.num_days = req.num_days
        self.trip_days = req.trip_days

        # Step 1: research
        self.session.research_brief = self.research_agent.run(req)

        # Step 2: plan → validate → retry loop
        validation_errors = None
        itinerary = None
        val_result = None
        for attempt in range(MAX_RETRIES + 1):
            itinerary = self.planning_agent.plan(req, self.session.research_brief, validation_errors)

            with self.tracer.span("validator") as span:
                val_result = self.validator.validate(itinerary, req, self.session.research_brief)
                span.retry_count = attempt
                span.router_intent = "NEW_TRIP"
                span.validation_result = "pass" if val_result.passed else "fail"

            if val_result.passed:
                break

            validation_errors = [val_result.error_message()]
            if attempt == MAX_RETRIES:
                val_result.soft_warnings.extend(
                    [f"[Hard error after {MAX_RETRIES} retries] {e}" for e in val_result.hard_errors]
                )
                break

        # Merge soft warnings into itinerary
        soft = val_result.soft_warnings if itinerary and not itinerary.get("parse_error") else []
        if soft and isinstance(itinerary, dict):
            itinerary.setdefault("warnings", [])
            itinerary["warnings"].extend(soft)

        self.session.current_itinerary = itinerary

        all_warnings = list(soft)
        if self.session.research_brief.get("partial"):
            all_warnings.append(
                "Research was incomplete (tool call limit reached). Itinerary may be missing some details."
            )
        all_warnings.extend(self.session.research_brief.get("warnings", []))

        self.session.add_turn(
            user_msg=f"[form] {req.location} {req.num_days}d ${req.budget}",
            assistant_msg=str(itinerary),
        )
        return {
            "itinerary": itinerary,
            "research_brief": self.session.research_brief,
            "warnings": all_warnings,
            "cache_report": self.session.api_cache.report(),
            "trace_summary": self.tracer.summary(),
        }

    @traceable(name="Orchestrator.handle_message", run_type="chain")
    def handle_message(self, user_msg: str) -> dict:
        """Called for follow-up chat messages after the initial plan is generated."""
        if self.tracer is None:
            self.tracer = PipelineTracer()
            self.research_agent.tracer = self.tracer
            self.planning_agent.tracer = self.tracer

        with self.tracer.span("intent_router") as span:
            intent = self.router.route(user_msg)
            span.router_intent = intent.upper()

        if intent == "research" and self.session.last_request:
            brief = self.research_agent.run(self.session.last_request)
            self.session.research_brief = brief
            response = brief.get("raw_output") or str(brief)
        elif intent == "refine" and self.session.current_itinerary:
            refined = self.planning_agent.refine(self.session.current_itinerary, user_msg)
            self.session.current_itinerary = refined
            response = str(refined)
        else:
            response = str(self.planning_agent.refine(self.session.current_itinerary or {}, user_msg))

        self.session.add_turn(user_msg=user_msg, assistant_msg=response)
        return {
            "response": response,
            "intent": intent,
            "cache_report": self.session.api_cache.report(),
            "trace_summary": self.tracer.summary(),
        }
