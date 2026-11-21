"""
ui/app.py
=========
Single-screen Streamlit "Presidential Command Center" for the Quantitative
20-Agent Geopolitical Simulator.

Run from the project root:
    streamlit run ui/app.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `core.*` resolves cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.engine import (
    ADVISORS,
    BLACKOPS_TYPES,
    CABLES,
    LLM_PROVIDER,
    MARKETWAR_ACTIONS,
    NON_ALIGNED,
    PROVIDERS,
    RIVAL_AXIS,
    US_COALITION,
    Simulation,
    consult_advisors,
    dm_agent,
    execute_turn,
    export_portfolio_csv,
    fog_of_war_probabilities,
    get_sim,
    llm_ready,
    load_history_baseline,
    reset_sim,
    run_blackops,
    run_marketwar,
    run_sync,
    set_provider,
    summit_agents,
)

# =========================================================================== #
# 1. PAGE CONFIG & MILITARY / BLOOMBERG DARK THEME
# =========================================================================== #
st.set_page_config(
    page_title="POTUS Command Center // Quant Geopolitical Simulator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #070b12;
        --bg-panel: #0e141f;
        --bg-card: #121a28;
        --border-dim: #1f2b3e;
        --accent-amber: #f59e0b;
        --accent-cyan: #22d3ee;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --text-main: #e2e8f0;
        --text-muted: #94a3b8;
    }
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-main);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #090e17;
        border-right: 1px solid var(--border-dim);
    }
    .ticker-bar {
        background: linear-gradient(90deg, #0b111c 0%, #101827 50%, #0b111c 100%);
        border: 1px solid #243247;
        border-left: 4px solid var(--accent-amber);
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .ticker-label {
        font-size: 0.70rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .ticker-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-main);
    }
    .defcon-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
    }
    .sigint-box {
        background-color: #090d14;
        border: 1px solid #7f1d1d;
        border-top: 3px solid #dc2626;
        border-radius: 6px;
        padding: 12px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        max-height: 420px;
        overflow-y: auto;
    }
    .red-phone-alert {
        background: linear-gradient(135deg, #2b0909 0%, #160606 100%);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 0 18px rgba(239, 68, 68, 0.25);
    }
    .card-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border-dim);
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================== #
# 2. SESSION STATE INITIALIZATION
# =========================================================================== #
def init_session_state() -> Simulation:
    if "sim" not in st.session_state:
        st.session_state.sim = get_sim()
    if "active_view" not in st.session_state:
        st.session_state.active_view = "Situation Room & Turn Control"
    if "cabinet_chat" not in st.session_state:
        st.session_state.cabinet_chat = []  # List[Dict[str, Any]]
    if "last_consult" not in st.session_state:
        st.session_state.last_consult = None
    if "dm_history" not in st.session_state:
        st.session_state.dm_history = {}  # Dict[agent_name, List[Dict]]
    if "last_summit" not in st.session_state:
        st.session_state.last_summit = None
    if "last_blackops" not in st.session_state:
        st.session_state.last_blackops = None
    if "last_marketwar" not in st.session_state:
        st.session_state.last_marketwar = None
    if "pending_red_phone" not in st.session_state:
        st.session_state.pending_red_phone = None  # {"move": str, "probs": dict}
    if "turn_logs" not in st.session_state:
        st.session_state.turn_logs = []
    return st.session_state.sim


sim: Simulation = init_session_state()


# =========================================================================== #
# 3. HELPER PLOTS & FORMATTERS
# =========================================================================== #
DEFCON_COLORS = {
    5: ("#10b981", "#052e16"),
    4: ("#84cc16", "#1a2e05"),
    3: ("#f59e0b", "#451a03"),
    2: ("#f97316", "#431407"),
    1: ("#ef4444", "#450a0a"),
}


def render_global_ticker(sim: Simulation) -> None:
    ws = sim.ws
    m = ws["market"]
    defcon = int(ws.get("defcon", 5))
    fg, bg = DEFCON_COLORS.get(defcon, ("#10b981", "#052e16"))
    sp_chg = float(m.get("sp500_change_pct", 0.0))
    sp_color = "#10b981" if sp_chg >= 0 else "#ef4444"
    hawkes = float(ws.get("hawkes_escalation_intensity", 0.0))
    hawkes_color = "#ef4444" if hawkes >= 5.0 else "#f59e0b" if hawkes >= 2.5 else "#22d3ee"
    score = float(ws.get("us_strategic_score", 0.0))
    inv_tag = " [INV]" if ws.get("yield_curve_inverted") else ""

    st.markdown(
        f"""
        <div class="ticker-bar">
            <div style="display:grid; grid-template-columns: repeat(7, 1fr); gap: 8px; align-items: center;">
                <div>
                    <div class="ticker-label">Turn</div>
                    <div class="ticker-val" style="color:#22d3ee;">T-{ws.get('turn', 0)}</div>
                </div>
                <div>
                    <div class="ticker-label">DEFCON</div>
                    <span class="defcon-badge" style="color:{fg}; background:{bg}; border:1px solid {fg};">
                        DEFCON {defcon}
                    </span>
                </div>
                <div>
                    <div class="ticker-label">S&amp;P 500</div>
                    <div class="ticker-val">{m['sp500']:,.0f} <span style="font-size:0.8rem; color:{sp_color};">({sp_chg:+.1f}%)</span></div>
                </div>
                <div>
                    <div class="ticker-label">WTI Crude</div>
                    <div class="ticker-val">${m['oil_price']:.2f}<span style="font-size:0.75rem; color:#94a3b8;">/bbl</span></div>
                </div>
                <div>
                    <div class="ticker-label">10Y Treasury</div>
                    <div class="ticker-val">{ws['treasury_yield_10y']:.2f}%<span style="font-size:0.75rem; color:#ef4444;">{inv_tag}</span></div>
                </div>
                <div>
                    <div class="ticker-label">Hawkes Intensity</div>
                    <div class="ticker-val" style="color:{hawkes_color};">&lambda; = {hawkes:.2f}</div>
                </div>
                <div>
                    <div class="ticker-label">US Strategic Score</div>
                    <div class="ticker-val" style="color:#10b981;">{score:.1f}<span style="font-size:0.8rem; color:#94a3b8;">/100</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_nash_heatmap(nash: Dict[str, Any]) -> go.Figure:
    us_strat: Dict[str, float] = nash.get("us_coalition_strategy", {})
    rv_strat: Dict[str, float] = nash.get("rival_axis_strategy", {})
    us_labels = list(us_strat.keys()) or ["De-escalate", "Contain", "Escalate"]
    rv_labels = list(rv_strat.keys()) or ["De-escalate", "Probe", "Escalate"]

    # Use explicit 3x3 payoff matrix if exposed by quant_engine; otherwise compute joint mixed-strategy probability matrix
    payoff = nash.get("payoff_matrix")
    if isinstance(payoff, list) and len(payoff) == len(us_labels):
        z = np.array(payoff, dtype=float)
        title = f"3x3 Superpower Nash Payoff Matrix ({nash.get('method', 'LP')})"
        fmt = ".2f"
    else:
        u_vec = np.array([float(us_strat.get(k, 1.0 / len(us_labels))) for k in us_labels])
        r_vec = np.array([float(rv_strat.get(k, 1.0 / len(rv_labels))) for k in rv_labels])
        z = np.outer(u_vec, r_vec) * 100.0
        title = f"3x3 Mixed-Strategy Nash Joint Probability % ({nash.get('method', 'Nash')})"
        fmt = ".1f"

    text = [[f"{val:{fmt}}" for val in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=rv_labels,
            y=us_labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#f8fafc"},
            colorscale="YlOrRd",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Rival Axis Strategy",
        yaxis_title="US Coalition Strategy",
        template="plotly_dark",
        paper_bgcolor="#0e141f",
        plot_bgcolor="#0e141f",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def build_merton_mc_chart(sim: Simulation, n_paths: int = 1000, steps: int = 21) -> go.Figure:
    """
    Generate a 1,000-path Merton Jump-Diffusion fan chart anchored to the live
    S&P 500, Hawkes intensity, supply-chain stress, and 95% VaR/CVaR.
    """
    ws = sim.ws
    m = ws["market"]
    s0 = float(m["sp500"])
    hawkes = float(ws.get("hawkes_escalation_intensity", 1.0))
    stress = float(ws.get("supply_chain_stress_index", 20.0))
    var_95_pct = float(m.get("sp500_var_95", 0.0))
    cvar_95_pct = float(m.get("sp500_cvar_95", 0.0))

    # Deterministic per-turn seed so switching views keeps the chart stable
    rng = np.random.default_rng(seed=42 + int(ws.get("turn", 0)))
    dt = 1.0 / 252.0
    mu = 0.06 - 0.015 * max(0.0, hawkes - 1.0)
    sigma = 0.16 + 0.002 * stress + 0.025 * hawkes
    jump_lambda = 0.5 + 1.2 * hawkes
    jump_mu = -0.025 - 0.008 * hawkes
    jump_sigma = 0.03 + 0.005 * hawkes

    paths = np.zeros((n_paths, steps + 1), dtype=float)
    paths[:, 0] = s0
    for t in range(1, steps + 1):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(jump_lambda * dt, n_paths)
        jump_sum = rng.normal(jump_mu, jump_sigma, n_paths) * n_jumps
        drift = (mu - 0.5 * sigma**2) * dt
        paths[:, t] = paths[:, t - 1] * np.exp(drift + sigma * np.sqrt(dt) * z + jump_sum)

    x_axis = list(range(steps + 1))
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    p05 = np.percentile(paths, 5, axis=0)
    # CVaR (Expected Shortfall) path: mean of bottom 5% at each step
    cvar_path = np.array([paths[:, t][paths[:, t] <= p05[t]].mean() for t in range(steps + 1)])

    fig = go.Figure()
    # Plot 40 faint sample trajectories for visual texture
    for i in range(40):
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=paths[i],
                mode="lines",
                line=dict(color="rgba(34, 211, 238, 0.08)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 95% upper-to-lower band
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list(p95) + list(p05[::-1]),
            fill="toself",
            fillcolor="rgba(34, 211, 238, 0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5th–95th Percentile Band",
        )
    )

    # Shaded 95% VaR to CVaR tail-risk band
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list(p05) + list(cvar_path[::-1]),
            fill="toself",
            fillcolor="rgba(239, 68, 68, 0.32)",
            line=dict(color="rgba(239, 68, 68, 0.6)", width=1),
            name=f"95% VaR / CVaR Tail ({var_95_pct:.1f}% / {cvar_95_pct:.1f}%)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=p50,
            mode="lines",
            line=dict(color="#22d3ee", width=2.5),
            name="Median Path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=cvar_path,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="CVaR (Expected Shortfall)",
        )
    )

    fig.update_layout(
        title=f"S&P 500 Merton Jump-Diffusion ({n_paths:,} Paths, 21-Day Horizon)",
        xaxis_title="Trading Days Forward",
        yaxis_title="S&P 500 Index Level",
        template="plotly_dark",
        paper_bgcolor="#0e141f",
        plot_bgcolor="#0e141f",
        height=400,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_yield_curve_chart(ws: Dict[str, Any]) -> go.Figure:
    ff = float(ws["fed_funds_rate"])
    y2 = float(ws["treasury_yield_2y"])
    y10 = float(ws["treasury_yield_10y"])
    spread = y10 - y2
    color = "#ef4444" if ws.get("yield_curve_inverted") else "#10b981"

    fig = go.Figure(
        data=go.Scatter(
            x=["Fed Funds", "2Y Treasury", "10Y Treasury"],
            y=[ff, y2, y10],
            mode="lines+markers+text",
            text=[f"{ff:.2f}%", f"{y2:.2f}%", f"{y10:.2f}%"],
            textposition="top center",
            line=dict(color=color, width=3),
            marker=dict(size=10, color=color),
        )
    )
    fig.update_layout(
        title=f"US Yield Curve (2s10s Spread: {spread:+.2f}%)",
        yaxis_title="Yield (%)",
        template="plotly_dark",
        paper_bgcolor="#0e141f",
        plot_bgcolor="#0e141f",
        height=270,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


# =========================================================================== #
# 4. SIDEBAR NAVIGATION & SYSTEM CONTROLS
# =========================================================================== #
with st.sidebar:
    st.markdown("### 🛰️ PRESIDENTIAL COMMAND")
    st.caption("Quantitative 20-Agent Geopolitical Simulator")

    nav_views = [
        "Situation Room & Turn Control",
        "US Cabinet & NSC Board Room",
        "Foreign Diplomacy & Espionage Hub",
        "Quant Finance & Stochastic Market Desk",
    ]
    selected_view = st.radio(
        "Command Views",
        nav_views,
        index=nav_views.index(st.session_state.active_view),
        label_visibility="collapsed",
    )
    st.session_state.active_view = selected_view

    st.divider()
    st.markdown("**LLM Provider & Engine**")
    provider_names = list(PROVIDERS.keys())
    cur_idx = provider_names.index(LLM_PROVIDER) if LLM_PROVIDER in provider_names else 0
    chosen_provider = st.selectbox("Provider", provider_names, index=cur_idx)
    if st.button("Switch Provider", use_container_width=True):
        set_provider(chosen_provider)
        st.rerun()

    ok_llm, reason_llm = llm_ready()
    if ok_llm:
        st.success(f"LLM Online: {chosen_provider} ({PROVIDERS[chosen_provider]['model']})")
    else:
        st.warning(f"Offline Heuristic Mode ({reason_llm})")

    st.divider()
    st.markdown("**Historical Baseline (/history)**")
    hist_date = st.text_input("Date (YYYY-MM-DD)", value=sim.ws.get("scenario_date") or "2022-02-24")
    if st.button("Load Historical Baseline", use_container_width=True):
        with st.spinner("Fetching real market baseline..."):
            res = run_sync(load_history_baseline(hist_date.strip(), sim=sim))
        if res.get("ok"):
            st.success(f"Loaded {res['date']} (Regime: {res['regime'] or 'Standard'})")
            st.rerun()
        else:
            st.error(res.get("error", "Failed to load history."))

    st.divider()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("💾 Save", use_container_width=True):
            sim.save_all()
            st.toast("Saved data/*.json")
    with col_s2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim = reset_sim()
            st.session_state.pending_red_phone = None
            st.session_state.last_consult = None
            st.session_state.cabinet_chat = []
            st.session_state.dm_history = {}
            st.rerun()


# =========================================================================== #
# 5. GLOBAL PERSISTENT TICKER
# =========================================================================== #
render_global_ticker(sim)


# =========================================================================== #
# 6. VIEW 1: SITUATION ROOM & TURN CONTROL
# =========================================================================== #
if selected_view == "Situation Room & Turn Control":
    ws = sim.ws
    dom = ws["us_domestic_politics"]

    # ---- 3:00 AM RED PHONE MODAL / ALERT HANDLER ----
    if st.session_state.pending_red_phone is not None:
        rp = st.session_state.pending_red_phone
        probs = rp["probs"]
        st.markdown(
            f"""
            <div class="red-phone-alert">
                <h3 style="color:#ef4444; margin-top:0;">☎️ 03:00 AM RED PHONE — EARLY WARNING ALARM</h3>
                <p><strong>Source:</strong> Space Force Early-Warning (Radar + Satellite)<br/>
                <strong>Alert:</strong> Multiple launch detections inbound toward CONUS. Confidence: <em>UNVERIFIED</em>.<br/>
                <strong>Alarm Probability:</strong> {probs['p_alarm']:.1%} &nbsp;|&nbsp;
                <strong>Bayesian Posterior P(REAL):</strong> <span style="color:#f59e0b; font-weight:700;">{probs['posterior_real']:.1%}</span></p>
                <p style="margin-bottom:0;"><strong>Pending Presidential Order:</strong> <code>{rp['move']}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_rp1, c_rp2, c_rp3 = st.columns(3)
        chosen_rp: Optional[str] = None
        if c_rp1.button("1️⃣ Stand Down & Verify (Backchannel)", use_container_width=True):
            chosen_rp = "verify"
        if c_rp2.button("2️⃣ Raise Alert & Disperse Forces", use_container_width=True):
            chosen_rp = "alert"
        if c_rp3.button("3️⃣ Launch Retaliatory Strike", type="primary", use_container_width=True):
            chosen_rp = "retaliate"

        if chosen_rp:
            async def _ui_red_phone_handler(_ev: Dict[str, Any]) -> str:
                return chosen_rp  # type: ignore[return-value]

            st.session_state.turn_logs = []
            sim.progress = lambda msg: st.session_state.turn_logs.append(msg)
            with st.spinner("Executing turn with Red Phone command..."):
                rep = run_sync(
                    execute_turn(
                        rp["move"],
                        sim=sim,
                        red_phone_handler=_ui_red_phone_handler,
                        force_red_phone=True,
                    )
                )
            st.session_state.pending_red_phone = None
            st.rerun()

    # ---- OFFICIAL /ACT TURN EXECUTION BOX ----
    st.subheader("Official Presidential Order (/act)")
    with st.form("act_turn_form", clear_on_submit=True):
        move_text = st.text_area(
            "Directive",
            placeholder="e.g., Deploy carrier strike group to the Taiwan Strait and impose secondary sanctions on shadow-fleet oil tankers...",
            height=85,
            label_visibility="collapsed",
        )
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        force_rp = fc1.checkbox("Simulate 3:00 AM Red Phone Crisis (--redphone)", value=False)
        fog_probs = fog_of_war_probabilities(sim)
        fc2.caption(
            f"Fog-of-War Sensor Sweep: P(Alarm) = {fog_probs['p_alarm']:.1%} | "
            f"P(Real | Alarm) = {fog_probs['posterior_real']:.1%}"
        )
        submitted = fc3.form_submit_button("EXECUTE TURN", type="primary", use_container_width=True)

    if submitted and move_text.strip():
        probs = fog_of_war_probabilities(sim)
        # Check whether a Red Phone interrupt triggers before running the turn
        if force_rp or random.random() < probs["p_alarm"]:
            st.session_state.pending_red_phone = {"move": move_text.strip(), "probs": probs}
            st.rerun()
        else:
            st.session_state.turn_logs = []
            sim.progress = lambda msg: st.session_state.turn_logs.append(msg)
            with st.spinner("Running 20-agent parallel deliberation & QuantMacroEngine..."):
                run_sync(execute_turn(move_text.strip(), sim=sim, force_red_phone=False))
            st.rerun()

    # ---- LAST TURN REPORT & FLASHPOINTS ----
    if sim.last_report is not None:
        rep = sim.last_report
        with st.expander(f"📰 Turn {rep.turn} Referee Adjudication & Headlines", expanded=True):
            st.markdown(f"**Presidential Move:** `{rep.move}`")
            st.info(rep.narrative or "No narrative recorded.")
            if rep.red_phone:
                st.error(
                    f"**RED PHONE ({rep.red_phone.get('decision', '').upper()}):** "
                    f"{rep.red_phone.get('outcome', '')}"
                )
            for h in rep.headlines:
                st.markdown(f"- **{h}**")
            if rep.llm_fallbacks:
                st.caption(f"⚠️ {rep.llm_fallbacks} LLM call(s) fell back to deterministic heuristics.")

    # ---- DEEP STATE / 25TH AMENDMENT COUP METERS & STOCKPILES ----
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("12 Global Trade Corridors")
        cor_rows = []
        for k, flow in ws["trade_corridors_status"].items():
            if flow >= 0.8:
                status = "OK"
            elif flow >= 0.6:
                status = "STRESSED"
            elif ws.get("us_secondary_sanctions_executed"):
                status = "SEVERED"
            else:
                status = "SHADOW FLEET"
            cor_rows.append({"Corridor": k, "Flow %": round(flow * 100.0, 1), "Status": status})
        cor_df = pd.DataFrame(cor_rows)
        st.dataframe(
            cor_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Flow %": st.column_config.ProgressColumn("Flow %", min_value=0, max_value=100, format="%.0f%%"),
            },
        )

    with col_right:
        st.subheader("Deep State & 25th Amendment Coup Meters")
        coup_risk = float(dom["deep_state_25th_amendment_risk"])
        if coup_risk >= 30:
            st.error(f"CRITICAL COUP / 25TH AMENDMENT RISK: {coup_risk:.1f}/100")
        elif coup_risk >= 15:
            st.warning(f"ELEVATED DOMESTIC INSTABILITY: {coup_risk:.1f}/100")

        meters = [
            ("25th Amendment / Coup Risk", coup_risk),
            ("War Exhaustion Index", float(dom["war_exhaustion_index"])),
            ("Rally-Around-the-Flag Effect", float(dom["rally_effect"])),
            ("Wall Street Oligarch Loyalty", float(dom["wall_street_oligarch_loyalty"])),
            ("Pentagon Joint Chiefs Obedience", float(dom["pentagon_joint_chiefs_obedience"])),
        ]
        for label, val in meters:
            st.caption(f"{label}: **{val:.1f} / 100**")
            st.progress(min(max(int(round(val)), 0), 100))

        st.subheader("3 Undersea Cables")
        for cable_name, status in ws["undersea_cables"].items():
            badge = "🔴 CUT" if status == "Cut" else "🟢 INTACT"
            st.markdown(f"**{cable_name}** — {badge} ({len(CABLES[cable_name])} nations)")

    # ---- ACTIVE FLASHPOINTS (20-AGENT MATRIX) ----
    st.subheader("Active Flashpoints & 20-Agent Readiness Matrix")
    intel_map = ws["us_profile"].get("intel_on", {})
    ag_rows = []
    for name, a in sim.agents.items():
        cable_cut = ws["undersea_cables"].get(sim.cable_of(name)) == "Cut"
        ag_rows.append(
            {
                "Agent": name,
                "Bloc": a["coalition"],
                "Trust": round(a["trust"], 1),
                "Risk Tol": round(a["risk_tolerance"], 2),
                "Loss Domain": "YES" if a["in_loss_domain"] else "-",
                "Military": round(a["mil"], 0),
                "Stability": round(a.get("regime_stability", 60.0), 0),
                "Cable": "CUT" if cable_cut else "OK",
                "CIA Intel": round(intel_map.get(name, 0.0), 0),
            }
        )
    st.dataframe(pd.DataFrame(ag_rows), use_container_width=True, hide_index=True)


# =========================================================================== #
# 7. VIEW 2: US CABINET & NSC BOARD ROOM
# =========================================================================== #
elif selected_view == "US Cabinet & NSC Board Room":
    st.subheader("US Cabinet & National Security Council (/consult)")
    st.caption("Consult `CIA_Director`, `Fed_Chair`, and `Joint_Chiefs` in parallel, plus Superpower 3x3 Nash Equilibrium & Shapley Betrayal Warnings.")

    # Interactive Cabinet Chat / Consultation Box
    with st.form("cabinet_consult_form", clear_on_submit=True):
        draft_move = st.text_input(
            "Draft Presidential Move or Question for the NSC",
            placeholder="e.g., Should we authorize a naval blockade of Iranian oil terminals?",
        )
        run_consult = st.form_submit_button("🏛️ /consult US Cabinet", type="primary")

    if run_consult and draft_move.strip():
        with st.spinner("CIA Director, Fed Chair, and Joint Chiefs deliberating..."):
            result = run_sync(consult_advisors(draft_move.strip(), sim=sim))
        st.session_state.last_consult = result
        st.session_state.cabinet_chat.append({"draft": draft_move.strip(), "result": result})

    if st.session_state.last_consult:
        res = st.session_state.last_consult
        st.markdown(f"**Evaluated Draft:** `{res['draft']}`")

        # 3 Advisor Cards
        adv_cols = st.columns(3)
        for col, op in zip(adv_cols, res["advisors"]):
            verdict = op["verdict"]
            risk = float(op["risk_score"])
            success_prob = max(5.0, min(95.0, (10.0 - risk) * 10.0))
            v_color = "#10b981" if verdict == "PROCEED" else "#f59e0b" if verdict == "MODIFY" else "#ef4444"
            with col:
                st.markdown(
                    f"""
                    <div class="card-box" style="border-top: 3px solid {v_color};">
                        <div style="font-weight:700; font-size:1.05rem;">{op['advisor']}</div>
                        <div style="color:{v_color}; font-weight:700; margin:4px 0;">
                            {verdict} &nbsp;|&nbsp; Risk: {risk:.1f}/10 &nbsp;|&nbsp; Est. Success: {success_prob:.0f}%
                        </div>
                        <p style="font-size:0.88rem; color:#cbd5e1;">{op['rationale']}</p>
                        <div style="font-size:0.82rem; color:#f59e0b;">
                            <strong>Warnings:</strong><br/>{"<br/>".join("• " + w for w in op['warnings']) or "None"}
                        </div>
                        <div style="font-size:0.82rem; color:#22d3ee; margin-top:6px;">
                            <strong>Suggested Modification:</strong> {op['suggested_modification']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3x3 Nash Equilibrium Heatmap + Shapley Betrayal Warnings
        c_nash, c_shap = st.columns([3, 2])
        with c_nash:
            st.plotly_chart(build_nash_heatmap(res["nash"]), use_container_width=True)
            st.caption(
                f"Coalition Power Share: **{res['nash'].get('coalition_power_share', 0):.0%}** | "
                f"Solver: `{res['nash'].get('method', 'LP')}`"
            )
        with c_shap:
            st.markdown("#### Shapley Coalition Betrayal Warnings")
            for w in res["shapley_warnings"]:
                if w["flag"]:
                    st.error(f"🚨 {w['text']}")
                else:
                    st.warning(f"⚠️ {w['text']}")

    # Persistent Cabinet Chat History
    if len(st.session_state.cabinet_chat) > 1:
        with st.expander(f"Previous NSC Consultations ({len(st.session_state.cabinet_chat) - 1})"):
            for item in reversed(st.session_state.cabinet_chat[:-1]):
                st.markdown(f"**Draft:** `{item['draft']}`")
                summary = " | ".join(
                    f"{o['advisor']}: **{o['verdict']}** ({o['risk_score']:.1f}/10)"
                    for o in item["result"]["advisors"]
                )
                st.caption(summary)


# =========================================================================== #
# 8. VIEW 3: FOREIGN DIPLOMACY & ESPIONAGE HUB
# =========================================================================== #
elif selected_view == "Foreign Diplomacy & Espionage Hub":
    col_dip, col_intel = st.columns([3, 2])

    with col_dip:
        tab_dm, tab_summit, tab_ops = st.tabs(["💬 Private /dm", "🌐 Multilateral /summit", "🕶️ Black Ops & Market War"])

        # ---- TAB 1: PRIVATE /DM ----
        with tab_dm:
            agent_list = sorted(sim.agents.keys())
            sel_agent = st.selectbox("Select Foreign Leader (20 Agents)", agent_list)
            cable_name = sim.cable_of(sel_agent)
            cable_cut = sim.cable_status().get(cable_name) == "Cut"
            dossier = sim.find_kompromat(sel_agent)
            ag_info = sim.agents[sel_agent]

            st.caption(
                f"Bloc: **{ag_info['coalition']}** | Trust: **{ag_info['trust']:.0f}/100** | "
                f"Cable: **{cable_name}** ({'🔴 CUT' if cable_cut else '🟢 OK'})"
            )
            if cable_cut:
                st.error(f"Secure channel to {sel_agent} is BLOCKED ({cable_name} is severed).")

            use_blackmail = False
            if dossier:
                use_blackmail = st.toggle(
                    f"🔥 Deploy Kompromat Blackmail ({dossier['id']}: {dossier['description']})",
                    value=False,
                )
            else:
                st.caption("No unused CIA Kompromat dossier currently unlocked for this leader.")

            with st.form("dm_form", clear_on_submit=True):
                dm_msg = st.text_input("Encrypted Message", placeholder=f"Message to {sel_agent}...")
                send_dm = st.form_submit_button("Send /dm", type="primary", disabled=cable_cut)

            if send_dm and dm_msg.strip():
                with st.spinner(f"Transmitting encrypted cable to {sel_agent}..."):
                    dm_res = run_sync(dm_agent(sel_agent, dm_msg.strip(), blackmail=use_blackmail, sim=sim))
                st.session_state.dm_history.setdefault(sel_agent, []).append(
                    {"us": dm_msg.strip(), "res": dm_res}
                )
                st.rerun()

            for entry in reversed(st.session_state.dm_history.get(sel_agent, [])):
                st.markdown(f"**POTUS:** {entry['us']}")
                r = entry["res"]
                if not r.get("ok"):
                    st.error(r.get("error", "Transmission failed."))
                else:
                    rep = r["reply"]
                    st.info(
                        f"**{r['agent']}** [{rep['stance']} | Compliance {rep['compliance']:.0%} | "
                        f"Trust {r['new_trust']:.0f}]: {rep['reply']}"
                    )

        # ---- TAB 2: /SUMMIT ----
        with tab_summit:
            bloc_options = ["us_coalition", "rival_axis", "non_aligned", "five_eyes", "all"]
            summit_mode = st.radio("Target Mode", ["Bloc Preset", "Custom Selection"], horizontal=True)
            if summit_mode == "Bloc Preset":
                summit_targets: Any = st.selectbox("Select Bloc", bloc_options)
            else:
                summit_targets = st.multiselect("Select Nations", sorted(sim.agents.keys()), default=["United Kingdom", "Japan"])

            with st.form("summit_form", clear_on_submit=True):
                summit_msg = st.text_area("Summit Broadcast", placeholder="Propose joint sanctions or security pact...", height=80)
                send_summit = st.form_submit_button("Broadcast /summit", type="primary")

            if send_summit and summit_msg.strip():
                with st.spinner("Convening multilateral summit..."):
                    st.session_state.last_summit = run_sync(summit_agents(summit_targets, summit_msg.strip(), sim=sim))

            if st.session_state.last_summit:
                sr = st.session_state.last_summit
                if not sr.get("ok"):
                    st.error(sr.get("error", "Summit failed."))
                else:
                    st.success(f"Result: **{sr['consensus']}** (Support Share: {sr['support_share']:.0%})")
                    if sr.get("excluded_cable_cut"):
                        st.warning(f"Excluded (cable severed): {', '.join(sr['excluded_cable_cut'])}")
                    st.dataframe(
                        pd.DataFrame(sr["replies"])[["agent", "stance", "compliance", "new_trust", "reply"]],
                        use_container_width=True,
                        hide_index=True,
                    )

        # ---- TAB 3: /BLACKOPS & /MARKETWAR ----
        with tab_ops:
            st.markdown("#### CIA Covert Operations (/blackops)")
            bo_c1, bo_c2 = st.columns(2)
            bo_op = bo_c1.selectbox("Operation Type", list(BLACKOPS_TYPES))
            bo_targets = list(CABLES.keys()) + sorted(sim.agents.keys()) if "cable" in bo_op else sorted(sim.agents.keys())
            bo_target = bo_c2.selectbox("Target", bo_targets)
            if st.button("Execute /blackops", type="primary", use_container_width=True):
                st.session_state.last_blackops = run_sync(run_blackops(bo_op, bo_target, sim=sim))
                st.rerun()

            if st.session_state.last_blackops:
                bo = st.session_state.last_blackops
                if bo.get("ok"):
                    st.info(f"**{bo['op']} → {bo['target']}** | Outcome: **{bo['outcome']}** (P_int: {bo['p_intercept']:.0%})")
                    for eff in bo.get("effects", []):
                        st.markdown(f"- {eff}")
                else:
                    st.error(bo.get("error"))

            st.divider()
            st.markdown("#### Financial & Market Warfare (/marketwar)")
            mw_c1, mw_c2 = st.columns(2)
            mw_action = mw_c1.selectbox("Market War Action", list(MARKETWAR_ACTIONS))
            mw_amount = mw_c2.number_input("Amount (% SPR or $B ESF)", min_value=1.0, max_value=500.0, value=10.0, step=5.0)
            if st.button("Execute /marketwar", use_container_width=True):
                amt = mw_amount if mw_action != "secondary_sanctions_sweep" else None
                st.session_state.last_marketwar = run_sync(run_marketwar(mw_action, amt, sim=sim))
                st.rerun()

            if st.session_state.last_marketwar:
                mw = st.session_state.last_marketwar
                if mw.get("ok"):
                    for eff in mw.get("effects", []):
                        st.success(eff)
                else:
                    st.error(mw.get("error"))

    # ---- LIVE TOP SECRET // CIA SIGINT WIRETAP FEED ----
    with col_intel:
        st.subheader("🔴 TOP SECRET // CIA SIGINT Wiretap")
        feed = list(reversed(sim.ws.get("wiretap_feed", [])[-25:]))
        if not feed:
            wire_html = "<span style='color:#64748b;'>-- NO INTERCEPTS RECORDED --</span>"
        else:
            lines = []
            for e in feed:
                lvl = e.get("level", "SIGINT")
                lvl_col = "#ef4444" if lvl in ("ALERT", "COVERT", "COUNTERINTEL") else "#f59e0b" if lvl == "HUMINT" else "#22d3ee"
                lines.append(
                    f"<div style='margin-bottom:6px; border-bottom:1px solid #1e293b; padding-bottom:4px;'>"
                    f"<span style='color:#94a3b8;'>[T{e.get('turn', 0)}]</span> "
                    f"<span style='color:{lvl_col}; font-weight:700;'>{lvl}</span> "
                    f"<span style='color:#fde047;'>{e.get('source', '')}:</span> "
                    f"<span>{e.get('text', '')}</span></div>"
                )
            wire_html = "".join(lines)
        st.markdown(f"<div class='sigint-box'>{wire_html}</div>", unsafe_allow_html=True)

        unused_k = [d for d in sim.ws.get("kompromat_dossiers", []) if not d.get("used")]
        st.markdown(f"**Unlocked Kompromat Dossiers ({len(unused_k)}):**")
        for d in unused_k:
            st.caption(f"📁 `{d['id']}` ({d['target']}): {d['description']} [Leverage: {d.get('leverage_value', 0)}]")


# =========================================================================== #
# 9. VIEW 4: QUANT FINANCE & STOCHASTIC MARKET DESK
# =========================================================================== #
elif selected_view == "Quant Finance & Stochastic Market Desk":
    ws = sim.ws
    m = ws["market"]

    # Top Macro KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("S&P 500 VaR (95%)", f"{m.get('sp500_var_95', 0.0):.2f}%")
    k2.metric("S&P 500 CVaR (95%)", f"{m.get('sp500_cvar_95', 0.0):.2f}%")
    k3.metric("Inflation", f"{ws['inflation_pct']:.2f}%")
    k4.metric("Oil Curve", ws["oil_futures_curve"].replace("_", " "))
    k5.metric("Hostile Shorts / ESF", f"${ws['hostile_short_interest_billions']:.0f}B / ${ws['esf_capacity_billions']:.0f}B")

    # Merton Jump-Diffusion 1,000-Path Chart + Yield Curve Widget
    c_mc, c_yc = st.columns([3, 2])
    with c_mc:
        st.plotly_chart(build_merton_mc_chart(sim, n_paths=1000, steps=21), use_container_width=True)
    with c_yc:
        st.plotly_chart(build_yield_curve_chart(ws), use_container_width=True)
        if ws.get("yield_curve_inverted"):
            st.error("⚠️ YIELD CURVE INVERTED (2Y > 10Y) — Recession & Equity Jump Risk Elevated")
        else:
            st.success("✅ Yield Curve Normal (10Y ≥ 2Y)")
        st.caption(
            f"SPR Oil Reserves: **{ws['spr_oil_reserves_pct']:.0f}%** | "
            f"Supply-Chain Stress: **{ws['supply_chain_stress_index']:.0f}/100**"
        )

    # Eisenberg-Noe Sovereign Default & CDS Spreads Table
    st.subheader("Eisenberg-Noe Sovereign Default & CDS Spreads")
    credit_rows = []
    for name, a in sim.agents.items():
        credit_rows.append(
            {
                "Sovereign": name,
                "Bloc": a["coalition"],
                "GDP ($B)": round(a["gdp_bn"], 1),
                "PD (%)": round(100.0 * a["pd"], 2),
                "CDS Spread (bps)": round(a["cds_spread_bps"], 1),
                "Distance to Default": round(a.get("distance_to_default", 0.0), 2),
                "Shapley Value": round(a["shapley_coalition_value"], 2),
            }
        )
    credit_df = pd.DataFrame(credit_rows).sort_values(by="CDS Spread (bps)", ascending=False)
    st.dataframe(credit_df, use_container_width=True, hide_index=True)

    # Export Quantitative CSV into exports/
    st.divider()
    exp_c1, exp_c2 = st.columns([2, 3])
    with exp_c1:
        if st.button("📥 Export Quantitative Metrics CSV to exports/", type="primary", use_container_width=True):
            out_path = export_portfolio_csv(sim=sim)
            st.success(f"Exported to `{out_path}`")
    with exp_c2:
        hist_df = pd.DataFrame(ws.get("metrics_history", []))
        if not hist_df.empty:
            st.download_button(
                label="⬇️ Download CSV Directly in Browser",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="simulation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )