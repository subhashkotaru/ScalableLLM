# ScalableLLM — AI Travel Planner

A production-grade multi-agent travel planning system built on self-hosted vLLM (Hermes-3 on RunPod A100 80GB), with end-to-end LangSmith observability, Prometheus telemetry, and concurrent load benchmarking.

---

## Architecture

```
User (Streamlit)
      │
      ▼
TripOrchestrator  ──────────────────────────────────────────┐
      │                                                      │
      ▼                                                      │
ResearchAgent (ReAct loop)                         LangSmith Tracing
  ├── search_destinations  (Wikivoyage FAISS)       Prometheus /metrics
  ├── search_places        (Google Places API)      nvidia-smi via SSH
  ├── search_hotels        (Google Places API)
  ├── get_travel_time      (Google Directions API)
  ├── get_weather          (Open-Meteo + Geocoding)
  └── get_reviews          (Google Places API)
      │  [tool calls dispatched concurrently via ThreadPoolExecutor]
      ▼
PlanningAgent  (structured JSON itinerary, schema-enforced)
      │
      ▼
ItineraryValidator  (rule-based: budget, timing, meals, place_ids)
      │  [retry loop up to 2x with validation errors fed back]
      ▼
Itinerary + Reasoning + Source Links (Google Maps URLs)
```

---

## Stack

| Component | Technology |
|---|---|
| LLM Serving | vLLM on RunPod A100 80GB |
| Model | Hermes-3 (Qwen3-based, OpenAI-compatible) |
| Orchestration | Custom Python agents (no LangChain) |
| Vector Search | FAISS over Wikivoyage chunks |
| External APIs | Google Places, Directions, Geocoding; Open-Meteo |
| Observability | LangSmith tracing, Prometheus, nvidia-smi SSH |
| Frontend | Streamlit |
| API Cache | In-memory keyed cache (deduplicates repeat tool calls) |

---

## Benchmark: Prefix Caching Under Concurrent Agentic Load

**Setup:** 20 diverse trip requests (1–3 day trips, $100–$800 budgets, varied interests), ramped across 3 concurrency stages. Each request runs the full ResearchAgent → PlanningAgent → Validator pipeline (~12–13 tool calls, ~15k prompt tokens).

**vLLM config:** `--enable-prefix-caching`, Hermes-3, 8192 token context, A100 80GB

### Stage-Level Results (averaged across 3 runs)

| Stage | Concurrency | Requests | Mean Latency | p95 Latency | TTFT | TPOT | KV Hit Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 (warm-up) | 2 | 4 | 180.8s | 199.8s | 192ms | 10.4ms | 59.7% |
| 2 (medium)  | 4 | 8 | 261.2s | 277.4s | 172ms | 12.9ms | 62.3% |
| 3 (peak)    | 8 | 8 | 284.5s | 348.4s | 509ms | 15.2ms | 60.5% |

### Key Observations

| Metric | Concurrency 2 → 8 | Change |
|---|---|---|
| TTFT | 192ms → 509ms | **+165%** — scheduler queue buildup |
| TPOT (inter-token) | 10.4ms → 15.2ms | **+46%** — batch contention |
| KV Cache Hit Rate | 59.7% → 60.5% | **stable** — prefix cache effective regardless of load |
| Mean E2E Latency | 180s → 284s | **+57%** — dominated by tool call overhead, not LLM(also 8 requests are processed so on avg 36s per req) |


### Overall (across all runs)

| Metric | Value |
|---|---|
| Total LLM requests served | ~100 per run |
| Mean prompt tokens | ~15,900 |
| Mean generation tokens | ~15,800 |
| KV cache hit rate | **60–63%** consistent |
| Avg TTFT | 171–192ms (concurrency 2–4) |
| Avg TPOT | 10–15ms |
| Avg e2e vLLM latency | ~50s per LLM call sequence |
| Pipeline success rate | 55–70% (failures: context overflow, parse errors) |

---

## Context-Length Resilience

Each agent request accumulates up to **15k prompt tokens** across 12–15 tool call round-trips. With an 8192 total context window this required:

| Problem | Solution |
|---|---|
| Research brief too large (~14k chars) | `_slim_brief()`: strips reviews, truncates Wikivoyage passages to 200 chars, keeps top 8 places — reduces to ~4k chars (**~70% reduction**) |
| Message history overflow mid-loop | `_trim_messages()`: drops oldest assistant+tool pairs, preserves system prompt + last 6 messages, auto-retries on 400 |
| Truncated JSON output | `_salvage_truncated_json()`: closes unclosed brackets to recover partial itineraries |
| `<think>` blocks from Hermes reasoning | Stripped via regex before JSON parse |

---

## Concurrency Design

Tool calls within each ReAct step are dispatched **concurrently**:

```python
# Each tool in a batch fires in parallel; context propagated to threads
with ThreadPoolExecutor(max_workers=len(unified)) as executor:
    tc_futures = [
        (tc, executor.submit(contextvars.copy_context().run, _dispatch_tool, ...))
        for tc in unified
    ]
```

- `contextvars.copy_context()` per thread — preserves LangSmith run tree for correct nested tracing
- Results appended in submission order — deterministic message history
- Typical tool batch: 3–5 concurrent API calls → ~3× wall-clock speedup vs sequential

---

## Observability

**LangSmith** trace hierarchy:
```
TravelPlanner (chain)
├── ResearchAgent.llm_call (llm) ×12–15
│   └── tool:{name} (tool) ×3–5 per call
└── PlanningAgent.llm_call (llm) ×1–2
```

**GPU Metrics tab** (live, via SSH to RunPod):
- nvidia-smi: temp, GPU util, memory used/total, power draw, SM clock, ECC errors
- vLLM Prometheus: KV cache usage/hit rate, TTFT, TPOT, requests running/waiting

---

## Running

```bash
# Start vLLM on RunPod
vllm serve Hermes-3 --enable-prefix-caching --max-model-len 8192

# SSH tunnel
ssh -L 8000:localhost:8000 -L 9400:localhost:9400 runpod

# Web app
streamlit run app.py

# Benchmark (prefix caching ON)
python benchmark.py --experiment prefix_cache_on

# Benchmark (prefix caching OFF — restart vLLM without --enable-prefix-caching)
python benchmark.py --experiment prefix_cache_off
```

### Benchmark output
```
results/
└── prefix_cache_on/
    └── 20260416_063300/
        ├── summary.json   ← stage metrics, vLLM snapshots, GPU state
        └── records.json   ← per-request latency, tokens, tool calls, status
```

---

## Environment

```bash
OPENAI_API_KEY=...
GOOGLE_PLACES_API_KEY=...
GOOGLE_DIRECTIONS_API_KEY=...
OPENWEATHERMAP_API_KEY=...        # unused — switched to Open-Meteo
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=ScalableLLM-TravelPlanner
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=travel-copilot
RUNPOD_SSH_HOST=...
RUNPOD_SSH_PORT=...
RUNPOD_SSH_USER=root
RUNPOD_SSH_KEY=~/.ssh/id_ed25519
DCGM_ENABLED=1
```
##LANGSMITH SAMPLE RUN

https://smith.langchain.com/public/8c140dd6-3718-4978-bca9-f6be1cacdec3/r