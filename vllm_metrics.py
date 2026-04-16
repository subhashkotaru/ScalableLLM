"""
GPU telemetry:
  - nvidia-smi  : hardware metrics via SSH (always available)
  - DCGM        : detailed metrics via dcgm-exporter Prometheus endpoint (optional)
  - vLLM        : serving metrics via Prometheus /metrics endpoint (via tunnel)

History is kept in-process for plotting (rolling 100 samples).
"""

import re
import subprocess
import requests
import os
import time
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

METRICS_URL      = "http://localhost:8000/metrics"
DCGM_METRICS_URL = os.getenv("DCGM_METRICS_URL", "http://localhost:9400/metrics")
DCGM_ENABLED     = os.getenv("DCGM_ENABLED", "0") == "1"

SSH_HOST = os.getenv("RUNPOD_SSH_HOST", "64.247.206.116")
SSH_PORT = os.getenv("RUNPOD_SSH_PORT", "14481")
SSH_USER = os.getenv("RUNPOD_SSH_USER", "root")
SSH_KEY  = os.path.expanduser(os.getenv("RUNPOD_SSH_KEY", "~/.ssh/id_ed25519"))

HISTORY_LEN = 100  # rolling samples kept for plots

# ---------------------------------------------------------------------------
# In-process time-series history (survives Streamlit reruns via session_state)
# ---------------------------------------------------------------------------

_history: dict[str, deque] = {}

def _record(key: str, value):
    if key not in _history:
        _history[key] = deque(maxlen=HISTORY_LEN)
    _history[key].append({"t": datetime.now().strftime("%H:%M:%S"), "v": value})

def get_history(key: str) -> tuple[list, list]:
    """Returns (timestamps, values) lists for plotting."""
    h = list(_history.get(key, []))
    return [p["t"] for p in h], [p["v"] for p in h]


# ---------------------------------------------------------------------------
# nvidia-smi via SSH
# ---------------------------------------------------------------------------

_SMI_FIELDS = [
    "name", "index",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]


def fetch_gpu_metrics() -> dict:
    """SSH nvidia-smi — returns {gpus: [...]} or {error: ...}."""
    query = ",".join(_SMI_FIELDS)
    cmd = [
        "ssh", "-p", SSH_PORT, "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        f"{SSH_USER}@{SSH_HOST}",
        f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if res.returncode != 0:
            return {"error": res.stderr.strip() or "nvidia-smi failed"}
        gpus = []
        for line in res.stdout.strip().splitlines():
            vals = [v.strip() for v in line.split(",")]
            if len(vals) < len(_SMI_FIELDS):
                continue
            def _f(v):
                try: return float(v)
                except: return None
            g = {
                "name":            vals[0],
                "index":           int(vals[1]) if vals[1].isdigit() else 0,
                "temp_c":          _f(vals[2]),
                "gpu_util_pct":    _f(vals[3]),
                "mem_util_pct":    _f(vals[4]),
                "mem_used_mib":    _f(vals[5]),
                "mem_free_mib":    _f(vals[6]),
                "mem_total_mib":   _f(vals[7]),
                "power_draw_w":    _f(vals[8]),
                "power_limit_w":   _f(vals[9]),
                "sm_clock_mhz":    _f(vals[10]),
                "mem_clock_mhz":   _f(vals[11]),
                "ecc_corrected":   _f(vals[12]),
                "ecc_uncorrected": _f(vals[13]),
                "pcie_gen":        _f(vals[14]),
                "pcie_width":      _f(vals[15]),
            }
            gpus.append(g)
            idx = g["index"]
            # Record history for plots
            for key, val in [
                (f"gpu{idx}_temp",     g["temp_c"]),
                (f"gpu{idx}_util",     g["gpu_util_pct"]),
                (f"gpu{idx}_mem_used", g["mem_used_mib"]),
                (f"gpu{idx}_power",    g["power_draw_w"]),
                (f"gpu{idx}_sm_clk",   g["sm_clock_mhz"]),
            ]:
                if val is not None:
                    _record(key, val)
        return {"gpus": gpus}
    except subprocess.TimeoutExpired:
        return {"error": "SSH timeout"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# DCGM via Prometheus endpoint (dcgm-exporter tunnel on port 9400)
# ---------------------------------------------------------------------------

_DCGM_METRICS = {
    "DCGM_FI_DEV_SM_CLOCK":         "sm_clock_mhz",
    "DCGM_FI_DEV_MEM_CLOCK":        "mem_clock_mhz",
    "DCGM_FI_DEV_GPU_TEMP":         "temp_c",
    "DCGM_FI_DEV_POWER_USAGE":      "power_draw_w",
    "DCGM_FI_DEV_GPU_UTIL":         "gpu_util_pct",
    "DCGM_FI_DEV_MEM_COPY_UTIL":    "mem_util_pct",
    "DCGM_FI_DEV_FB_USED":          "mem_used_mib",
    "DCGM_FI_DEV_FB_FREE":          "mem_free_mib",
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "ecc_sbe",
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": "ecc_dbe",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL": "nvlink_bw_mbs",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE": "gr_engine_active",
    "DCGM_FI_PROF_SM_ACTIVE":        "sm_active",
    "DCGM_FI_PROF_SM_OCCUPANCY":     "sm_occupancy",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": "tensor_active",
    "DCGM_FI_PROF_DRAM_ACTIVE":      "dram_active",
    "DCGM_FI_PROF_PCIE_TX_BYTES":    "pcie_tx_bytes",
    "DCGM_FI_PROF_PCIE_RX_BYTES":    "pcie_rx_bytes",
}


def fetch_dcgm_metrics() -> dict:
    """Fetch DCGM exporter Prometheus metrics. Returns {} if disabled/unavailable."""
    if not DCGM_ENABLED:
        return {"disabled": True}
    try:
        resp = requests.get(DCGM_METRICS_URL, timeout=3)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        return {"error": str(e)}

    result = {}
    for prom_name, friendly in _DCGM_METRICS.items():
        # Match: METRIC_NAME{gpu="0",...} value
        matches = re.findall(
            rf'^{re.escape(prom_name)}\{{[^}}]*gpu="(\d+)"[^}}]*\}}\s+([\d.e+\-]+)',
            text, re.MULTILINE
        )
        for gpu_idx, val in matches:
            key = f"gpu{gpu_idx}_{friendly}"
            result[key] = float(val)
            _record(key, float(val))

    return result


# ---------------------------------------------------------------------------
# vLLM Prometheus metrics
# ---------------------------------------------------------------------------

def _parse(text, metric):
    m = re.search(rf'^{re.escape(metric)}\{{[^}}]*\}}\s+([\d.e+\-]+)', text, re.MULTILINE)
    if m: return float(m.group(1))
    m = re.search(rf'^{re.escape(metric)}\s+([\d.e+\-]+)', text, re.MULTILINE)
    return float(m.group(1)) if m else None

def _parse_sum(text, metric):
    return sum(float(v) for v in re.findall(
        rf'^{re.escape(metric)}\{{[^}}]*\}}\s+([\d.e+\-]+)', text, re.MULTILINE))


def fetch_vllm_metrics() -> dict:
    try:
        resp = requests.get(METRICS_URL, timeout=3)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        return {"error": str(e)}

    kv_usage    = _parse(text, "vllm:kv_cache_usage_perc")
    queries     = _parse_sum(text, "vllm:prefix_cache_queries_total")
    hits        = _parse_sum(text, "vllm:prefix_cache_hits_total")
    kv_hit_rate = (hits / queries * 100) if queries > 0 else 0.0

    gpu_mem_util = None
    m = re.search(r'gpu_memory_utilization="([^"]+)"', text)
    if m:
        try: gpu_mem_util = float(m.group(1)) * 100
        except: pass

    num_gpu_blocks = None
    m2 = re.search(r'num_gpu_blocks="(\d+)"', text)
    if m2: num_gpu_blocks = int(m2.group(1))

    running  = _parse(text, "vllm:num_requests_running")
    waiting  = _parse(text, "vllm:num_requests_waiting")
    prompt_t = _parse_sum(text, "vllm:prompt_tokens_total")
    gen_t    = _parse_sum(text, "vllm:generation_tokens_total")

    lat_sum   = _parse_sum(text, "vllm:e2e_request_latency_seconds_sum")
    lat_count = _parse_sum(text, "vllm:e2e_request_latency_seconds_count")
    avg_lat   = (lat_sum / lat_count) if lat_count > 0 else None

    ttft_sum   = _parse_sum(text, "vllm:time_to_first_token_seconds_sum")
    ttft_count = _parse_sum(text, "vllm:time_to_first_token_seconds_count")
    avg_ttft   = (ttft_sum / ttft_count) if ttft_count > 0 else None

    tpot_sum   = _parse_sum(text, "vllm:inter_token_latency_seconds_sum")
    tpot_count = _parse_sum(text, "vllm:inter_token_latency_seconds_count")
    avg_tpot   = (tpot_sum / tpot_count) if tpot_count > 0 else None

    d = {
        "kv_cache_usage_pct":  round(kv_usage * 100, 1) if kv_usage is not None else None,
        "kv_cache_hit_rate":   round(kv_hit_rate, 1),
        "kv_cache_hits":       int(hits),
        "kv_cache_queries":    int(queries),
        "gpu_memory_util_pct": round(gpu_mem_util, 1) if gpu_mem_util is not None else None,
        "num_gpu_blocks":      num_gpu_blocks,
        "requests_running":    int(running) if running is not None else None,
        "requests_waiting":    int(waiting) if waiting is not None else None,
        "total_prompt_tokens": int(prompt_t),
        "total_gen_tokens":    int(gen_t),
        "avg_e2e_latency_s":   round(avg_lat, 2) if avg_lat else None,
        "avg_ttft_s":          round(avg_ttft, 3) if avg_ttft else None,
        "avg_tpot_ms":         round(avg_tpot * 1000, 1) if avg_tpot else None,
        "total_requests":      int(lat_count),
    }
    # Record history
    for key, val in [
        ("vllm_kv_usage",    d["kv_cache_usage_pct"]),
        ("vllm_kv_hit_rate", d["kv_cache_hit_rate"]),
        ("vllm_running",     d["requests_running"]),
        ("vllm_ttft",        d["avg_ttft_s"]),
        ("vllm_itl",         d["avg_tpot_ms"]),
    ]:
        if val is not None:
            _record(key, val)
    return d


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def fetch_metrics() -> dict:
    return {
        "vllm": fetch_vllm_metrics(),
        "gpu":  fetch_gpu_metrics(),
        "dcgm": fetch_dcgm_metrics(),
    }
