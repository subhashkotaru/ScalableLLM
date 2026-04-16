import datetime
import streamlit as st
from orchestrator import TripOrchestrator, TripRequest
from vllm_metrics import fetch_metrics, get_history

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("AI Travel Planner")
st.subheader("Plan your perfect trip with us")
st.divider()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = TripOrchestrator()
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

with st.form("trip_form"):
    today    = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=tomorrow, min_value=tomorrow)
    with col2:
        end_date = st.date_input("End Date", value=tomorrow + datetime.timedelta(days=2), min_value=tomorrow)

    col3, col4 = st.columns(2)
    with col3:
        from_location = st.text_input("From (Origin)", placeholder="e.g. New York, NY")
    with col4:
        location = st.text_input("To (Destination)", placeholder="e.g. Portland, Maine")

    budget = st.number_input("Budget (USD)", min_value=0, value=1500, step=100)

    col5, col6 = st.columns(2)
    with col5:
        interests = st.text_area("Interests", placeholder="e.g. museums, hiking, local food...", height=100)
    with col6:
        constraints = st.text_area("Constraints", placeholder="e.g. vegetarian, no flying...", height=100)

    submitted = st.form_submit_button("Generate Travel Plan", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Validation & orchestration
# ---------------------------------------------------------------------------

if submitted:
    errors = []
    if not from_location.strip():
        errors.append("Please enter your origin location.")
    if not location.strip():
        errors.append("Please enter a destination.")
    if end_date < start_date:
        errors.append("End date must be on or after the start date.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        req = TripRequest(
            from_location=from_location,
            location=location,
            start_date=start_date,
            end_date=end_date,
            budget=float(budget),
            interests=interests,
            constraints=constraints,
        )
        with st.spinner("Researching your destination and building your itinerary…"):
            st.session_state.result = st.session_state.orchestrator.handle_form(req)

# ---------------------------------------------------------------------------
# Render result
# ---------------------------------------------------------------------------

def _sparkline(history_key: str, label: str, color: str = "#4c9be8"):
    """Render a small line chart from history if enough data points exist."""
    ts, vs = get_history(history_key)
    if len(vs) >= 2:
        import pandas as pd
        df = pd.DataFrame({"time": ts, label: vs})
        st.line_chart(df.set_index("time"), height=120, color=color)


def _render_gpu_metrics():
    st.markdown("### GPU Telemetry")
    col_r, col_auto = st.columns([1, 4])
    with col_r:
        if st.button("🔄 Refresh", key="gpu_refresh"):
            st.rerun()
    with col_auto:
        auto = st.checkbox("Auto-refresh every 10s", key="gpu_auto")

    if auto:
        import time
        time.sleep(10)
        st.rerun()

    m     = fetch_metrics()
    vllm  = m.get("vllm", {})
    gpu   = m.get("gpu", {})
    gpus  = gpu.get("gpus", [])

    # ── nvidia-smi hardware metrics ─────────────────────────────────────────
    st.markdown("#### Hardware (nvidia-smi)")
    if "error" in gpu:
        st.warning(f"nvidia-smi unavailable: {gpu['error']}")
    elif gpus:
        for g in gpus:
            idx  = g["index"]
            name = g["name"]
            st.markdown(f"**GPU {idx} — {name}**")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Temp",        f"{g['temp_c']} °C" if g['temp_c'] is not None else "—")
            c2.metric("GPU Util",    f"{g['gpu_util_pct']} %" if g['gpu_util_pct'] is not None else "—")
            c3.metric("Mem Used",    f"{g['mem_used_mib']:,.0f} / {g['mem_total_mib']:,.0f} MiB" if g['mem_used_mib'] else "—")
            c4.metric("Power",       f"{g['power_draw_w']:.0f} / {g['power_limit_w']:.0f} W" if g['power_draw_w'] else "—")
            c5.metric("SM Clock",    f"{g['sm_clock_mhz']:.0f} MHz" if g['sm_clock_mhz'] else "—")
            c6.metric("ECC Errors",  f"{int(g['ecc_uncorrected'] or 0)} DBE / {int(g['ecc_corrected'] or 0)} SBE")

            pc1, pc2 = st.columns(2)
            with pc1:
                st.caption("GPU Utilization %")
                _sparkline(f"gpu{idx}_util", "GPU Util %", "#4c9be8")
            with pc2:
                st.caption("Power Draw (W)")
                _sparkline(f"gpu{idx}_power", "Power W", "#e8824c")

            pc3, pc4 = st.columns(2)
            with pc3:
                st.caption("Temperature (°C)")
                _sparkline(f"gpu{idx}_temp", "Temp °C", "#e84c4c")
            with pc4:
                st.caption("Memory Used (MiB)")
                _sparkline(f"gpu{idx}_mem_used", "Mem MiB", "#4ce8a0")

            st.markdown(f"PCIe Gen {int(g['pcie_gen'] or 0)} × {int(g['pcie_width'] or 0)} &nbsp;|&nbsp; "
                        f"Mem Clock {g['mem_clock_mhz']:.0f} MHz" if g['pcie_gen'] else "")
            st.divider()

    # ── vLLM serving metrics ────────────────────────────────────────────────
    st.markdown("#### vLLM Serving")
    if "error" in vllm:
        st.warning(f"vLLM metrics unavailable: {vllm['error']}")
        st.caption("Make sure SSH tunnel is open: `ssh -L 8000:localhost:8000 runpod`")
    else:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("KV Cache Usage",    f"{vllm.get('kv_cache_usage_pct')} %" if vllm.get('kv_cache_usage_pct') is not None else "—")
        v2.metric("KV Hit Rate",       f"{vllm.get('kv_cache_hit_rate')} %")
        v3.metric("Requests Running",  vllm.get('requests_running', "—"))
        v4.metric("Requests Waiting",  vllm.get('requests_waiting', "—"))

        v5, v6, v7, v8 = st.columns(4)
        v5.metric("Total Requests",    vllm.get('total_requests', "—"))
        v6.metric("Avg E2E Latency",   f"{vllm.get('avg_e2e_latency_s')} s" if vllm.get('avg_e2e_latency_s') else "—")
        v7.metric("Avg TTFT",          f"{vllm.get('avg_ttft_s')} s" if vllm.get('avg_ttft_s') else "—")
        v8.metric("Avg TPOT",          f"{vllm.get('avg_tpot_ms')} ms" if vllm.get('avg_tpot_ms') else "—")

        vc1, vc2 = st.columns(2)
        with vc1:
            st.caption("KV Cache Hit Rate %")
            _sparkline("vllm_kv_hit_rate", "KV Hit %", "#4ce8c4")
        with vc2:
            st.caption("Time to First Token (s)")
            _sparkline("vllm_ttft", "TTFT s", "#e8e84c")


def _cost(n) -> str:
    if n is None:
        return "—"
    return f"${n:,.0f}"

def _render_itinerary(itin: dict):
    summary = itin.get("trip_summary", {})

    # ── Trip summary banner ──────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Destination",   summary.get("destination", "—"))
    s2.metric("Dates",         summary.get("dates", "—"))
    s3.metric("Total Cost",    _cost(summary.get("total_cost")))
    s4.metric("Budget Left",   _cost(summary.get("budget_remaining")))

    st.divider()

    # ── Budget breakdown ─────────────────────────────────────────────────
    bd = itin.get("budget_breakdown", {})
    if bd:
        with st.expander("Budget Breakdown", expanded=False):
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Accommodation", _cost(bd.get("accommodation")))
            bc2.metric("Food",          _cost(bd.get("food")))
            bc3.metric("Activities",    _cost(bd.get("activities")))
            bc4.metric("Transport",     _cost(bd.get("transport")))

    # ── Days ─────────────────────────────────────────────────────────────
    days = itin.get("days", [])
    for day in days:
        day_num  = day.get("day_number", "?")
        date_str = day.get("date", "")
        weekday  = day.get("weekday", "")
        theme    = day.get("theme", "")
        weather  = day.get("weather", "")
        day_cost = day.get("day_total", 0)

        with st.expander(
            f"Day {day_num} — {weekday}, {date_str}  |  {theme}  |  {_cost(day_cost)}",
            expanded=(day_num == 1),
        ):
            if weather:
                st.caption(f"Weather: {weather}")

            activities = day.get("activities", [])
            for act in activities:
                atype    = act.get("type", "activity")
                name     = act.get("name", "")
                time_str = act.get("time", "")
                cost     = act.get("cost_total", 0)
                notes    = act.get("notes", "")
                travel   = act.get("travel_from_previous", "")
                duration = act.get("duration_hours")

                # Icon by type
                icon = {"meal": "🍽️", "activity": "🗺️", "transit": "🚗", "checkin": "🏨"}.get(atype, "📍")

                source_url = act.get("source_url", "")

                col_time, col_detail, col_cost = st.columns([1, 5, 1])
                with col_time:
                    st.markdown(f"**{time_str}**")
                with col_detail:
                    dur_str = f" · {duration}h" if duration else ""
                    if source_url:
                        st.markdown(f"{icon} **[{name}]({source_url})**{dur_str}")
                    else:
                        st.markdown(f"{icon} **{name}**{dur_str}")
                    if travel:
                        st.caption(f"🚗 {travel}")
                    if notes:
                        st.caption(notes)
                with col_cost:
                    if cost:
                        st.markdown(f"*{_cost(cost)}*")

                st.markdown("---")

            # Accommodation
            accom = day.get("accommodation")
            if accom:
                st.markdown(
                    f"🏨 **Stay:** {accom.get('name', '')} &nbsp;·&nbsp; "
                    f"{_cost(accom.get('cost_per_night'))}/night"
                )

    # ── Alternatives ─────────────────────────────────────────────────────
    alts = itin.get("alternatives", {})
    rain_swaps = alts.get("rain_day_swaps", [])
    if rain_swaps:
        with st.expander("☔ Rain Day Alternatives", expanded=False):
            for swap in rain_swaps:
                if isinstance(swap, dict):
                    outdoor  = swap.get("outdoor_activity_name") or swap.get("outdoor_place_id", "")
                    indoor   = swap.get("indoor_alternative_name") or swap.get("indoor_place_id", "")
                    notes    = swap.get("notes", "")
                    line = f"**{outdoor}** → **{indoor}**" if outdoor and indoor else indoor or outdoor or str(swap)
                    st.markdown(f"- {line}")
                    if notes:
                        st.caption(f"  {notes}")
                else:
                    st.markdown(f"- {swap}")

    # ── Insufficient data gaps ────────────────────────────────────────────
    gaps = itin.get("insufficient_data", [])
    if gaps:
        with st.expander("⚠️ Data Gaps", expanded=False):
            for g in gaps:
                st.markdown(f"- Day {g.get('day_number')} {g.get('time', '')}: {g.get('reason', '')}")

    # ── Trip Reasoning ────────────────────────────────────────────────────
    reasoning = itin.get("reasoning", {})
    if reasoning:
        st.divider()
        st.markdown("## Trip Planning Rationale")
        overview = reasoning.get("overview", "")
        if overview:
            st.markdown(overview)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            budget_logic = reasoning.get("budget_logic", "")
            if budget_logic:
                st.markdown("**Budget Allocation**")
                st.markdown(budget_logic)

            hotel_choice = reasoning.get("hotel_choice", "")
            if hotel_choice:
                st.markdown("**Hotel Choice**")
                st.markdown(hotel_choice)

        with col_r2:
            weather_impact = reasoning.get("weather_impact", "")
            if weather_impact:
                st.markdown("**Weather Considerations**")
                st.markdown(weather_impact)

            constraints_met = reasoning.get("constraints_met", "")
            if constraints_met:
                st.markdown("**Interests & Constraints**")
                st.markdown(constraints_met)

        day_by_day = reasoning.get("day_by_day", [])
        if day_by_day:
            st.markdown("**Day-by-Day Decisions**")
            for i, note in enumerate(day_by_day, 1):
                st.markdown(f"- **Day {i}:** {note}")


result = st.session_state.result

# GPU metrics tab is always visible
tab_plan, tab_warnings, tab_debug, tab_gpu = st.tabs(["Itinerary", "⚠️ Warnings (dev)", "🔍 Debug", "🖥️ GPU Metrics"])

with tab_gpu:
    _render_gpu_metrics()

if result:
    itin     = result.get("itinerary", {})
    warnings = result.get("warnings", [])
    itin_warnings = itin.get("warnings", []) if isinstance(itin, dict) else []

    with tab_plan:
        if itin and not itin.get("parse_error"):
            _render_itinerary(itin)
        elif itin and itin.get("parse_error"):
            raw = itin.get("raw_output", "")
            if not raw.strip():
                st.error("The planning agent returned an empty response. This usually means the model hit its token limit mid-generation or the vLLM server returned no content.")
                st.info("Check the Debug tab for token counts. You may need to increase max_tokens or shorten the research brief.")
            else:
                st.error("The planning agent returned malformed output. Raw response below:")
                st.code(raw, language="text")

    with tab_warnings:
        st.markdown("### Warnings")
        all_warnings = list(warnings) + [w for w in itin_warnings if w not in warnings]
        if all_warnings:
            for w in all_warnings:
                st.warning(w)
        else:
            st.success("No warnings.")

    with tab_debug:
        with st.expander("🔍 Pipeline Trace", expanded=True):
            trace = result.get("trace_summary", {})
            if trace:
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Total Latency",  f"{trace.get('total_latency_ms', 0):,.0f} ms")
                tc2.metric("Tokens In",      trace.get("total_tokens_in", 0))
                tc3.metric("Tokens Out",     trace.get("total_tokens_out", 0))
                tc4.metric("Cache Hit Rate", trace.get("cache_hit_rate", "N/A"))
            st.caption(result.get("cache_report", ""))

        with st.expander("📋 Research Brief (raw)", expanded=False):
            st.json(result.get("research_brief", {}))
