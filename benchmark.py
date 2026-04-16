"""
Benchmark the TripOrchestrator pipeline (ResearchAgent → PlanningAgent → Validator)
against a set of fixed trip requests, with concurrent execution.

Usage:
    python benchmark.py --experiment prefix_cache_on --runs 3 --concurrency 5
    python benchmark.py --experiment prefix_cache_off --runs 3 --concurrency 5 --output-dir /workspace/results
"""

import argparse
import json
import os
import subprocess
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from orchestrator import TripOrchestrator, TripRequest

# ---------------------------------------------------------------------------
# Fixed trip requests
# ---------------------------------------------------------------------------

RAW_REQUESTS = [
    dict(from_location="New York, NY",     location="Boston, MA",                   days=2, budget=400,  interests="history",                    constraints=""),
    dict(from_location="Boston, MA",       location="Portland, ME",                 days=2, budget=600,  interests="seafood, craft beer",         constraints=""),
    dict(from_location="New York, NY",     location="Jersey Shore, NJ",             days=1, budget=100,  interests="beach",                       constraints=""),
    dict(from_location="New York, NY",     location="Cape Cod, MA",                 days=3, budget=800,  interests="family activities, beach",    constraints="family with kids"),
    dict(from_location="Boston, MA",       location="Providence, RI",               days=2, budget=300,  interests="food, art",                   constraints=""),
    dict(from_location="New York, NY",     location="Burlington, VT",               days=3, budget=500,  interests="hiking",                      constraints=""),
    dict(from_location="New York, NY",     location="Philadelphia, PA",             days=2, budget=450,  interests="history, food",               constraints=""),
    dict(from_location="New York, NY",     location="Acadia National Park, ME",     days=2, budget=400,  interests="nature, hiking",              constraints="couple"),
    dict(from_location="New York, NY",     location="Newport, RI",                  days=1, budget=150,  interests="architecture",                constraints=""),
    dict(from_location="Boston, MA",       location="White Mountains, NH",          days=3, budget=600,  interests="hiking",                      constraints=""),
    dict(from_location="New York, NY",     location="Mystic, CT",                   days=2, budget=400,  interests="seafood, history",            constraints=""),
    dict(from_location="New York, NY",     location="Hudson Valley, NY",            days=3, budget=500,  interests="fall foliage, wineries",      constraints=""),
    dict(from_location="Boston, MA",       location="Stowe, VT",                    days=2, budget=700,  interests="skiing",                      constraints=""),
    dict(from_location="Boston, MA",       location="Salem, MA",                    days=1, budget=100,  interests="history",                     constraints=""),
    dict(from_location="Boston, MA",       location="Bar Harbor, ME",               days=3, budget=800,  interests="nature, food",                constraints=""),
    dict(from_location="New York, NY",     location="Lake Champlain, VT",           days=2, budget=400,  interests="outdoors",                    constraints=""),
    dict(from_location="New York, NY",     location="Washington, DC",               days=2, budget=600,  interests="museums",                     constraints=""),
    dict(from_location="New York, NY",     location="Baltimore, MD",                days=1, budget=150,  interests="harbor, seafood",             constraints=""),
    dict(from_location="New York, NY",     location="Finger Lakes, NY",             days=3, budget=550,  interests="wine, outdoors",              constraints=""),
    dict(from_location="Manhattan, NY",    location="Brooklyn, NY",                 days=1, budget=200,  interests="food tour",                   constraints=""),
]


def _make_request(raw: dict, start_date: datetime.date) -> TripRequest:
    end_date = start_date + datetime.timedelta(days=raw["days"] - 1)
    return TripRequest(
        from_location=raw["from_location"],
        location=raw["location"],
        start_date=start_date,
        end_date=end_date,
        budget=float(raw["budget"]),
        interests=raw["interests"],
        constraints=raw["constraints"],
    )


# ---------------------------------------------------------------------------
# GPU snapshot
# ---------------------------------------------------------------------------

def _gpu_snapshot() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        gpus = []
        for line in out.splitlines():
            idx, name, mem_used, mem_total, util, power = [v.strip() for v in line.split(",")]
            gpus.append({
                "index": int(idx), "name": name,
                "mem_used_mb": int(mem_used), "mem_total_mb": int(mem_total),
                "util_pct": float(util), "power_w": float(power),
            })
        return {"gpus": gpus}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Single pipeline run
# ---------------------------------------------------------------------------

def _run_one(raw: dict, start_date: datetime.date, idx: int) -> dict:
    """Instantiate a fresh orchestrator and run one trip request."""
    orchestrator = TripOrchestrator()
    req = _make_request(raw, start_date)

    t0 = time.time()
    try:
        result = orchestrator.handle_form(req)
        latency = time.time() - t0

        itin = result.get("itinerary", {})
        trace = result.get("trace_summary", {})
        parse_error = bool(itin.get("parse_error")) if isinstance(itin, dict) else True

        return {
            "idx":            idx,
            "location":       raw["location"],
            "days":           raw["days"],
            "budget":         raw["budget"],
            "status":         "error" if parse_error else "ok",
            "latency_s":      round(latency, 2),
            "tokens_in":      trace.get("total_tokens_in", 0),
            "tokens_out":     trace.get("total_tokens_out", 0),
            "tool_calls":     trace.get("total_tool_calls", 0),
            "cache_hit_rate": trace.get("cache_hit_rate", "N/A"),
            "warnings":       len(result.get("warnings", [])),
            "parse_error":    parse_error,
        }
    except Exception as e:
        return {
            "idx":         idx,
            "location":    raw["location"],
            "days":        raw["days"],
            "budget":      raw["budget"],
            "status":      "exception",
            "latency_s":   round(time.time() - t0, 2),
            "error":       str(e),
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

# Ramp stages: (concurrency, n_requests)
RAMP_STAGES = [
    (2, 4),   # warm-up:  4 requests at concurrency 2
    (4, 8),   # medium:   8 requests at concurrency 4
    (8, 8),   # peak:     8 requests at concurrency 8
]


def _vllm_snapshot() -> dict:
    """Fetch vLLM Prometheus metrics. Returns {} on error."""
    from vllm_metrics import fetch_vllm_metrics
    m = fetch_vllm_metrics()
    if "error" in m:
        return {"error": m["error"]}
    return {
        "kv_cache_usage_pct":  m.get("kv_cache_usage_pct"),
        "kv_cache_hit_rate":   m.get("kv_cache_hit_rate"),
        "kv_cache_hits":       m.get("kv_cache_hits"),
        "kv_cache_queries":    m.get("kv_cache_queries"),
        "avg_ttft_s":          m.get("avg_ttft_s"),
        "avg_tpot_ms":         m.get("avg_tpot_ms"),
        "avg_e2e_latency_s":   m.get("avg_e2e_latency_s"),
        "total_requests":      m.get("total_requests"),
        "requests_running":    m.get("requests_running"),
        "total_prompt_tokens": m.get("total_prompt_tokens"),
        "total_gen_tokens":    m.get("total_gen_tokens"),
    }


def _vllm_delta(before: dict, after: dict) -> dict:
    """Compute per-stage deltas for cumulative counters."""
    if "error" in before or "error" in after:
        return {}

    def _diff(key):
        a, b = before.get(key), after.get(key)
        if a is not None and b is not None:
            return round(b - a, 4)
        return None

    n_req = _diff("total_requests") or 1

    hits_delta    = _diff("kv_cache_hits")
    queries_delta = _diff("kv_cache_queries")
    kv_hit_rate   = round(hits_delta / queries_delta * 100, 1) if queries_delta else None

    return {
        "stage_requests":         _diff("total_requests"),
        "stage_prompt_tokens":    _diff("total_prompt_tokens"),
        "stage_gen_tokens":       _diff("total_gen_tokens"),
        "kv_cache_hit_rate_pct":  kv_hit_rate,
        "kv_cache_hits":          hits_delta,
        "kv_cache_queries":       queries_delta,
        "avg_ttft_s":             after.get("avg_ttft_s"),   # cumulative avg from vLLM
        "avg_tpot_ms":            after.get("avg_tpot_ms"),
        "avg_e2e_latency_s":      after.get("avg_e2e_latency_s"),
        "kv_cache_usage_pct":     after.get("kv_cache_usage_pct"),
    }


def _run_stage(stage_idx: int, concurrency: int, requests: list, start_date: datetime.date, all_records: list):
    print(f"\n  Stage {stage_idx} — concurrency={concurrency}, {len(requests)} requests", flush=True)

    vllm_before = _vllm_snapshot()
    t_stage = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_one, raw, start_date, raw["_idx"]): raw["_idx"]
            for raw in requests
        }
        stage_records = []
        for fut in as_completed(futures):
            rec = fut.result()
            rec["stage"] = stage_idx
            rec["stage_concurrency"] = concurrency
            stage_records.append(rec)
            symbol = "✓" if rec.get("status") == "ok" else "✗"
            print(f"    {symbol} [{rec['idx']:02d}] {rec['location']:<35} {rec['latency_s']:>6.1f}s  {rec.get('status','?')}", flush=True)

    elapsed = time.time() - t_stage
    vllm_after = _vllm_snapshot()
    vllm_delta = _vllm_delta(vllm_before, vllm_after)

    all_records.extend(stage_records)
    ok = [r for r in stage_records if r.get("status") == "ok"]
    print(f"  Stage {stage_idx} done in {elapsed:.1f}s  —  {len(ok)}/{len(stage_records)} succeeded", flush=True)
    if vllm_delta:
        print(f"    vLLM: TTFT={vllm_delta.get('avg_ttft_s')}s  TPOT={vllm_delta.get('avg_tpot_ms')}ms  "
              f"KV hit={vllm_delta.get('kv_cache_hit_rate_pct')}%  KV usage={vllm_delta.get('kv_cache_usage_pct')}%", flush=True)

    return stage_records, vllm_before, vllm_after, vllm_delta


def benchmark(
    experiment: str,
    output_dir: str,
):
    start_date = datetime.date.today() + datetime.timedelta(days=30)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / experiment / ts
    out_path.mkdir(parents=True, exist_ok=True)

    # Assign global index to each request
    requests = [{**r, "_idx": i} for i, r in enumerate(RAW_REQUESTS)]

    # Slice requests across stages
    total_needed = sum(n for _, n in RAMP_STAGES)
    assert total_needed <= len(requests), f"Need {total_needed} requests but only {len(requests)} defined"
    stages = []
    offset = 0
    for concurrency, n in RAMP_STAGES:
        stages.append((concurrency, requests[offset:offset + n]))
        offset += n

    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment}")
    print(f"  Stages     : {[(c, len(r)) for c, r in stages]}")
    print(f"  Output     : {out_path}")
    print(f"{'='*60}", flush=True)

    gpu_before = _gpu_snapshot()
    vllm_start = _vllm_snapshot()
    all_records = []
    stage_metrics = []

    for i, (concurrency, reqs) in enumerate(stages, 1):
        _, vb, va, vd = _run_stage(i, concurrency, reqs, start_date, all_records)
        stage_metrics.append((vb, va, vd))

    gpu_after = _gpu_snapshot()
    vllm_end = _vllm_snapshot()

    # Aggregate
    ok_records = [r for r in all_records if r.get("status") == "ok"]
    lats   = [r["latency_s"]  for r in ok_records]
    tok_in = [r["tokens_in"]  for r in ok_records]
    tok_out= [r["tokens_out"] for r in ok_records]
    tools  = [r["tool_calls"] for r in ok_records]

    # Per-stage breakdown including vLLM metrics
    per_stage = []
    for s_idx, (s_conc, _) in enumerate(RAMP_STAGES, 1):
        s_recs = [r for r in all_records if r.get("stage") == s_idx and r.get("status") == "ok"]
        s_lats = [r["latency_s"] for r in s_recs]
        _, _, vd = stage_metrics[s_idx - 1]
        cache_hits   = sum(1 for r in s_recs if r.get("cache_hit_rate") not in (None, "N/A", "0.0%", "0%"))
        per_stage.append({
            "stage":                s_idx,
            "concurrency":          s_conc,
            "n":                    len(s_recs),
            "mean_latency_s":       round(float(np.mean(s_lats)), 2)           if s_lats else None,
            "p95_latency_s":        round(float(np.percentile(s_lats, 95)), 2) if s_lats else None,
            "tool_cache_hits":      cache_hits,
            "vllm_avg_ttft_s":      vd.get("avg_ttft_s"),
            "vllm_avg_tpot_ms":     vd.get("avg_tpot_ms"),
            "vllm_avg_e2e_s":       vd.get("avg_e2e_latency_s"),
            "vllm_kv_hit_rate_pct": vd.get("kv_cache_hit_rate_pct"),
            "vllm_kv_usage_pct":    vd.get("kv_cache_usage_pct"),
            "vllm_stage_requests":  vd.get("stage_requests"),
            "vllm_prompt_tokens":   vd.get("stage_prompt_tokens"),
            "vllm_gen_tokens":      vd.get("stage_gen_tokens"),
        })

    overall_vllm_delta = _vllm_delta(vllm_start, vllm_end)

    summary = {
        "experiment":             experiment,
        "timestamp":              ts,
        "stages":                 RAMP_STAGES,
        "total_requests":         len(all_records),
        "successful":             len(ok_records),
        "failed":                 len(all_records) - len(ok_records),
        "p50_latency_s":          round(float(np.percentile(lats, 50)), 2) if lats else None,
        "p95_latency_s":          round(float(np.percentile(lats, 95)), 2) if lats else None,
        "mean_latency_s":         round(float(np.mean(lats)), 2)           if lats else None,
        "mean_tokens_in":         round(float(np.mean(tok_in)), 0)         if tok_in else None,
        "mean_tokens_out":        round(float(np.mean(tok_out)), 0)        if tok_out else None,
        "mean_tool_calls":        round(float(np.mean(tools)), 1)          if tools else None,
        "vllm_avg_ttft_s":        overall_vllm_delta.get("avg_ttft_s"),
        "vllm_avg_tpot_ms":       overall_vllm_delta.get("avg_tpot_ms"),
        "vllm_avg_e2e_s":         overall_vllm_delta.get("avg_e2e_latency_s"),
        "vllm_kv_hit_rate_pct":   overall_vllm_delta.get("kv_cache_hit_rate_pct"),
        "vllm_kv_usage_pct":      overall_vllm_delta.get("kv_cache_usage_pct"),
        "per_stage":              per_stage,
        "gpu_before":             gpu_before,
        "gpu_after":              gpu_after,
        "vllm_start":             vllm_start,
        "vllm_end":               vllm_end,
    }

    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_path / "records.json").write_text(json.dumps(all_records, indent=2))

    print(f"\n{'─'*60}")
    print(f"  RESULTS — {experiment}")
    print(f"{'─'*60}")
    skip = ("gpu_before", "gpu_after", "per_stage", "stages", "vllm_start", "vllm_end")
    for k, v in summary.items():
        if k in skip:
            continue
        print(f"  {k:<32} {str(v):>10}")

    print(f"\n  Per-stage breakdown:")
    print(f"  {'Stage':<7} {'Conc':>5} {'N':>3} {'mean_lat':>9} {'p95_lat':>8} {'TTFT':>7} {'TPOT':>8} {'KV hit%':>8} {'KV use%':>8}")
    for s in per_stage:
        print(f"  {s['stage']:<7} {s['concurrency']:>5} {s['n']:>3} "
              f"{str(s['mean_latency_s']):>9} {str(s['p95_latency_s']):>8} "
              f"{str(s.get('vllm_avg_ttft_s')):>7} {str(s.get('vllm_avg_tpot_ms')):>8} "
              f"{str(s.get('vllm_kv_hit_rate_pct')):>8} {str(s.get('vllm_kv_usage_pct')):>8}")
    print(f"\n  Saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TripOrchestrator pipeline")
    parser.add_argument("--experiment",  required=True,       help="e.g. prefix_cache_on, prefix_cache_off, chunked_prefill")
    parser.add_argument("--output-dir",  default="results",   help="Root folder for results")
    args = parser.parse_args()

    benchmark(
        experiment=args.experiment,
        output_dir=args.output_dir,
    )
