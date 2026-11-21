"""
core/quant_engine.py
====================
QuantMacroEngine: nine quantitative / game-theoretic models that convert the
LLM Referee's shock parameters into updates of ``world_state`` and
``agents_config`` for the 20-agent geopolitical simulator.

Models
------
 1. Merton jump-diffusion Monte Carlo (S&P 500 + crude oil, 1,000 paths)
 2. Merton structural sovereign default + Eisenberg-Noe clearing contagion
 3. 20+1 node NetworkX trade graph, shadow-fleet leaky bucket, supply stress
 4. Strategic stockpile ODEs (SPR, rare earths, munitions, tech edge), curve
 5. Hawkes self-exciting escalation + Bayesian false-alarm (fog of war)
 6. Shapley coalition values + 3x3 mixed Nash (support enumeration + linprog)
 7. Prospect-theory leader psychology + US domestic politics
 8. Augmented Taylor rule, yield curve, ESF short-squeeze
 9. Espionage wheel (logistic), Five-Eyes pooling, kompromat, burned spies

Data layout (relative to project root)::

    data/world_state.json, agents_config.json, diplomacy_log.json,
         historical_benchmarks.json
    exports/   (CSV exports)

``world_state.json`` is a dict (missing keys are filled with defaults)::

    market: {sp500, oil_price, oil_baseline, ten_year_yield, ...}
    undersea_cables_status: "Intact" | "Degraded" | "Cut" (or 0..1 intact fraction)
    hostile_short_interest_billions, trade_corridors_status {"A->B": 0..1},
    spr_oil_reserves_pct, rare_earth_defense_stockpile_pct,
    precision_munitions_magazine_pct, us_military_tech_edge, oil_futures_curve,
    hawkes_escalation_intensity, fed_funds_rate, treasury_yield_2y,
    treasury_yield_10y, yield_curve_inverted, inflation_pct,
    supply_chain_stress_index, us_domestic_politics {...}, ...

``agents_config.json`` may be ``{"agents": {name: {...}}}``, ``{name: {...}}`` or
a list of dicts with a ``name`` key; it is normalised to ``{"agents": {...}}``.
Missing agent fields are filled with defaults. The "US" node lives in
``world_state["us_profile"]``.

Referee shock keys (all optional) accepted by ``run_turn``::

    regime (historical shock key), undersea_cables_status / cables_cut,
    hostile_short_interest_billions / ..._delta_billions,
    corridor_flow_changes {"A->B": delta}, us_secondary_sanctions (bool|list),
    oil_supply_shock_pct, equity_jump_mean, spr_drawdown_pct,
    munitions_expended_pct, rare_earth_consumption_pct, escalation_events [0..1],
    gps_satellite_jamming 0..1, esf_squeeze_billions, war_damage_pct {agent: %},
    sanctions_severity {agent: delta 0..1}, humiliated_leaders {agent: 0..10},
    us_attacked_severity 0..10, us_casualties, war_intensity 0..1,
    controversial_order_severity 0..1, war_spending_pct_gdp, price_update_mode
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"

FIVE_EYES_POOL = ("United Kingdom", "Israel", "Australia", "Japan")
CABLE_MULT = {"intact": 1.0, "degraded": 1.35, "partially_cut": 1.35,
              "cut": 1.9, "severed": 1.9}
US_COALITION, RIVAL_AXIS, NON_ALIGNED = "US_Coalition", "Rival_Axis", "Non_Aligned"
US_STRATEGIES = ["Sanctions_Squeeze", "Military_Signaling", "Diplomatic_Deescalation"]
RIVAL_STRATEGIES = ["Cyber_Cable_Sabotage", "Proxy_Escalation", "Negotiate_Concede"]

# name, coalition, gdp_bn, fx_res_bn, asset_vol, leverage, mil, econ, attack, defense, trust(US)
_AGENT_ROWS: List[Tuple[Any, ...]] = [
    ("United Kingdom", US_COALITION, 3300, 190, 0.30, 0.35, 60, 55, 85, 82, 90),
    ("Israel", US_COALITION, 520, 200, 0.35, 0.35, 55, 25, 90, 85, 85),
    ("Australia", US_COALITION, 1750, 60, 0.28, 0.30, 35, 35, 75, 72, 92),
    ("Japan", US_COALITION, 4200, 1250, 0.32, 0.40, 50, 70, 70, 70, 80),
    ("Germany", US_COALITION, 4500, 350, 0.28, 0.30, 45, 75, 65, 70, 85),
    ("France", US_COALITION, 3100, 250, 0.32, 0.40, 55, 55, 72, 72, 80),
    ("South Korea", US_COALITION, 1700, 420, 0.33, 0.30, 50, 45, 75, 75, 82),
    ("Taiwan", US_COALITION, 790, 570, 0.38, 0.30, 25, 30, 70, 65, 78),
    ("China", RIVAL_AXIS, 18000, 3200, 0.35, 0.35, 85, 90, 88, 80, 10),
    ("Russia", RIVAL_AXIS, 2000, 600, 0.42, 0.40, 70, 25, 85, 75, 8),
    ("Iran", RIVAL_AXIS, 400, 30, 0.50, 0.50, 35, 10, 70, 55, 4),
    ("North Korea", RIVAL_AXIS, 30, 10, 0.55, 0.55, 40, 2, 72, 50, 3),
    ("Belarus", RIVAL_AXIS, 70, 8, 0.48, 0.50, 10, 3, 55, 50, 15),
    ("India", NON_ALIGNED, 3900, 650, 0.35, 0.40, 55, 50, 68, 62, 60),
    ("Saudi Arabia", NON_ALIGNED, 1100, 450, 0.36, 0.30, 30, 35, 60, 58, 65),
    ("Turkey", NON_ALIGNED, 1100, 90, 0.45, 0.40, 40, 30, 64, 60, 45),
    ("Brazil", NON_ALIGNED, 2200, 350, 0.40, 0.40, 25, 35, 55, 55, 55),
    ("UAE", NON_ALIGNED, 510, 200, 0.35, 0.30, 15, 25, 62, 60, 68),
    ("Indonesia", NON_ALIGNED, 1400, 140, 0.38, 0.40, 20, 25, 55, 52, 55),
    ("Mexico", NON_ALIGNED, 1800, 210, 0.38, 0.40, 15, 30, 50, 50, 70),
]
_US_ROW = ("US", US_COALITION, 28000, 250, 0.28, 0.35, 100, 100, 92, 85, 100)

DEFAULT_CORRIDORS: Dict[str, float] = {
    "China->US": 1.0, "Japan->US": 1.0, "South Korea->US": 1.0, "Taiwan->US": 1.0,
    "Germany->US": 1.0, "Saudi Arabia->China": 1.0, "Iran->China": 0.7,
    "Russia->China": 0.9, "Russia->India": 0.8, "UAE->India": 1.0,
    "Russia->Turkey": 0.9, "Saudi Arabia->Japan": 1.0,
}
# (debtor, creditor): sovereign bond / loan obligations in USD billions
CROSS_HOLDINGS: Dict[Tuple[str, str], float] = {
    ("US", "Japan"): 1100.0, ("US", "China"): 760.0, ("US", "United Kingdom"): 700.0,
    ("US", "Taiwan"): 270.0, ("US", "India"): 230.0, ("US", "Saudi Arabia"): 130.0,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-_clip(x, -60.0, 60.0)))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    tmp.replace(path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _agent_from_row(row: Tuple[Any, ...]) -> Dict[str, Any]:
    (name, coalition, gdp, fx, vol, lev, mil, econ, atk, dfn, trust) = row
    base_rt = 0.55 if coalition == RIVAL_AXIS else 0.35
    return {
        "name": name, "leader": f"{name} leadership", "coalition": coalition,
        "gdp_bn": float(gdp), "fx_res_bn": float(fx), "asset_vol": vol, "leverage": lev,
        "mil": float(mil), "econ": float(econ), "attack": float(atk), "defense": float(dfn),
        "trust": float(trust), "base_risk_tolerance": base_rt, "risk_tolerance": base_rt,
        "in_loss_domain": False, "humiliation_stock": 0.0, "war_damage_pct": 0.0,
        "sanctions_severity": 0.0, "pd": 0.0, "cds_spread_bps": 50.0,
        "shapley_coalition_value": 0.0,
    }


def _default_agents() -> Dict[str, Dict[str, Any]]:
    return {row[0]: _agent_from_row(row) for row in _AGENT_ROWS}


def _default_world_state() -> Dict[str, Any]:
    return {
        "turn": 0,
        "market": {"sp500": 5000.0, "oil_price": 80.0, "oil_baseline": 80.0,
                   "ten_year_yield": 4.3, "sp500_change_pct": 0.0},
        "undersea_cables_status": "Intact",
        "hostile_short_interest_billions": 0.0,
        "trade_corridors_status": dict(DEFAULT_CORRIDORS),
        "us_secondary_sanctions_executed": False,
        "secondary_sanction_corridors": [],
        "supply_chain_stress_index": 10.0,
        "inflation_pct": 3.0,
        "spr_oil_reserves_pct": 60.0,
        "rare_earth_defense_stockpile_pct": 55.0,
        "precision_munitions_magazine_pct": 80.0,
        "us_military_tech_edge": 75.0,
        "oil_futures_curve": "Flat",
        "hawkes_escalation_intensity": 0.2,
        "hawkes_params": {"mu0": 0.05, "beta": 0.35, "alpha": 0.8, "cap": 25.0},
        "pending_escalation_events": [],
        "fed_funds_rate": 4.5, "treasury_yield_2y": 4.1, "treasury_yield_10y": 4.3,
        "yield_curve_inverted": False,
        "esf_capacity_billions": 200.0,
        "us_domestic_politics": {
            "rally_effect": 0.0, "war_exhaustion_index": 15.0,
            "wall_street_oligarch_loyalty": 70.0,
            "pentagon_joint_chiefs_obedience": 75.0,
            "deep_state_25th_amendment_risk": 3.0,
        },
        "us_profile": _agent_from_row(_US_ROW),
        "kompromat_dossiers": [],
        "espionage_log": [],
    }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class QuantMacroEngine:
    """Applies the Referee's shock parameters through the nine models."""

    def __init__(self, data_dir: Optional[Path] = None, export_dir: Optional[Path] = None,
                 seed: Optional[int] = None, n_paths: int = 1000, horizon_days: int = 30,
                 autoload: bool = True) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.export_dir = Path(export_dir) if export_dir else EXPORT_DIR
        self.rng = np.random.default_rng(seed)
        self.n_paths = int(n_paths)
        self.horizon_days = int(horizon_days)
        self.world_state: Dict[str, Any] = _default_world_state()
        self.agents_config: Dict[str, Any] = {"agents": _default_agents()}
        self.last_paths: Dict[str, np.ndarray] = {}
        if autoload:
            self.load()
        self._ensure_consistency()

    # ------------------------------------------------------------------ I/O
    def load(self) -> None:
        ws = _read_json(self.data_dir / "world_state.json", {})
        if isinstance(ws, dict):
            _deep_merge(self.world_state, ws)
        raw = _read_json(self.data_dir / "agents_config.json", None)
        if raw:
            agents = self._normalize_agents(raw)
            if agents:
                self.agents_config = {k: v for k, v in (raw.items() if isinstance(raw, dict) else [])
                                      if k != "agents"}
                self.agents_config["agents"] = agents
        self._ensure_consistency()

    def save(self) -> None:
        _write_json(self.data_dir / "world_state.json", self.world_state)
        _write_json(self.data_dir / "agents_config.json", self.agents_config)

    @staticmethod
    def _normalize_agents(raw: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(raw, dict) and "agents" in raw:
            raw = raw["agents"]
        if isinstance(raw, list):
            return {str(a["name"]): a for a in raw if isinstance(a, dict) and "name" in a}
        if isinstance(raw, dict):
            return {k: v for k, v in raw.items() if isinstance(v, dict)}
        return {}

    def _ensure_consistency(self) -> None:
        """Fill missing fields with defaults so every model can run."""
        defaults = _default_agents()
        generic = _agent_from_row(("generic", NON_ALIGNED, 1000, 100, 0.35, 0.40, 20, 20, 60, 60, 50))
        agents = self.agents_config.setdefault("agents", {})
        for name, a in agents.items():
            tpl = defaults.get(name, generic)
            for k, v in tpl.items():
                a.setdefault(k, v)
            a["name"] = name
        us = self.world_state.setdefault("us_profile", _agent_from_row(_US_ROW))
        for k, v in _agent_from_row(_US_ROW).items():
            us.setdefault(k, v)
        mkt = self.world_state.setdefault("market", {})
        mkt.setdefault("sp500", 5000.0)
        mkt.setdefault("oil_price", 80.0)
        mkt.setdefault("oil_baseline", mkt["oil_price"])
        mkt.setdefault("ten_year_yield", self.world_state.get("treasury_yield_10y", 4.3))
        mkt.setdefault("sp500_change_pct", 0.0)
        for k, v in _default_world_state().items():
            self.world_state.setdefault(k, v)
        self.world_state["us_domestic_politics"] = _deep_merge(
            _default_world_state()["us_domestic_politics"], self.world_state["us_domestic_politics"])

    @property
    def agents(self) -> Dict[str, Dict[str, Any]]:
        return self.agents_config["agents"]

    @property
    def entities(self) -> Dict[str, Dict[str, Any]]:
        """20 agents plus the US node (live references, mutations persist)."""
        ents = dict(self.agents)
        ents["US"] = self.world_state["us_profile"]
        return ents

    # ------------------------------------------------------- shared helpers
    def _cable_multiplier(self) -> float:
        status = self.world_state.get("undersea_cables_status", "Intact")
        if isinstance(status, (int, float)):
            return 1.0 + (1.0 - _clip(float(status), 0.0, 1.0)) * 0.9
        return CABLE_MULT.get(str(status).strip().lower().replace(" ", "_"), 1.0)

    def _hawkes_norm(self) -> float:
        return 1.0 - math.exp(-max(0.0, float(self.world_state["hawkes_escalation_intensity"])))

    def _benchmark_params(self, regime: Optional[str]) -> Dict[str, Dict[str, float]]:
        """Calibrated vol/drift for ^GSPC and CL=F from data/historical_benchmarks.json."""
        if not regime:
            return {}
        bench = _read_json(self.data_dir / "historical_benchmarks.json", {})
        try:
            assets = bench["shocks"][regime]["assets"]
            return {"sp500": assets["^GSPC"], "oil": assets["CL=F"]}
        except (KeyError, TypeError):
            return {}

    def apply_shock_inputs(self, shocks: Dict[str, Any]) -> None:
        ws = self.world_state
        if "undersea_cables_status" in shocks:
            ws["undersea_cables_status"] = shocks["undersea_cables_status"]
        if shocks.get("cables_cut"):
            ws["undersea_cables_status"] = "Cut"
        if "hostile_short_interest_billions" in shocks:
            ws["hostile_short_interest_billions"] = max(0.0, float(shocks["hostile_short_interest_billions"]))
        if "hostile_short_interest_delta_billions" in shocks:
            ws["hostile_short_interest_billions"] = max(
                0.0, ws["hostile_short_interest_billions"] + float(shocks["hostile_short_interest_delta_billions"]))
        sanc = shocks.get("us_secondary_sanctions")
        if isinstance(sanc, (list, tuple, set)):
            ws["us_secondary_sanctions_executed"] = len(sanc) > 0
            ws["secondary_sanction_corridors"] = list(sanc)
        elif sanc is not None:
            ws["us_secondary_sanctions_executed"] = bool(sanc)
            ws["secondary_sanction_corridors"] = []

    # =====================================================================
    # MODEL 1 - Merton jump-diffusion Monte Carlo
    # =====================================================================
    @staticmethod
    def merton_paths(s0: float, mu: float, sigma: float, lam: float, k: float, delta: float,
                     days: int, n_paths: int, rng: np.random.Generator,
                     dt: float = 1.0 / 252.0) -> np.ndarray:
        """
        dS/S = (mu - lam*k) dt + sigma dW + J dN, with ln(1+J) ~ N(ln(1+k) - d^2/2, d^2)
        so that E[J] = k. Exact log-Euler scheme, vectorised. Shape (n_paths, days+1).
        """
        k = max(k, -0.95)
        m = math.log1p(k) - 0.5 * delta ** 2
        drift = (mu - 0.5 * sigma ** 2 - lam * k) * dt
        z = rng.standard_normal((n_paths, days))
        n_jumps = rng.poisson(lam * dt, (n_paths, days))
        z2 = rng.standard_normal((n_paths, days))
        jump_sum = n_jumps * m + np.sqrt(n_jumps) * delta * z2
        increments = drift + sigma * math.sqrt(dt) * z + jump_sum
        log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
        return s0 * np.exp(log_path)

    @staticmethod
    def _risk_stats(paths: np.ndarray) -> Dict[str, float]:
        s0 = paths[0, 0]
        terminal = paths[:, -1]
        rets = terminal / s0 - 1.0
        q05 = np.percentile(rets, 5)
        tail = rets[rets <= q05]
        return {
            "median_terminal": float(np.median(terminal)),
            "mean_terminal": float(np.mean(terminal)),
            "var_95_pct": float(max(0.0, -q05) * 100.0),
            "cvar_95_pct": float(max(0.0, -np.mean(tail)) * 100.0) if tail.size else 0.0,
            "p05_terminal": float(np.percentile(terminal, 5)),
            "p95_terminal": float(np.percentile(terminal, 95)),
        }

    def monte_carlo_markets(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws, mkt = self.world_state, self.world_state["market"]
        bench = self._benchmark_params(shocks.get("regime"))
        cable = self._cable_multiplier()
        short_pressure = min(float(ws["hostile_short_interest_billions"]) / 500.0, 1.0)
        h = self._hawkes_norm()
        stress = float(ws["supply_chain_stress_index"])
        spr_low = float(ws["spr_oil_reserves_pct"]) < 25.0

        # --- S&P 500
        sp_vol = float(bench.get("sp500", {}).get("annualized_volatility", 0.17))
        sp_mu = (_clip(bench["sp500"]["mean_daily_drift"] * 252.0, -0.8, 0.3)
                 if "sp500" in bench else 0.07)
        sp_sigma = _clip(sp_vol * cable * (1.0 + 0.5 * short_pressure), 0.05, 2.5)
        sp_lam = 2.0 * (1.0 + 2.0 * h + short_pressure)
        sp_k = (-0.04 + float(shocks.get("equity_jump_mean", 0.0)) - 0.03 * short_pressure - 0.03 * h)
        sp_delta = 0.05

        # --- Crude oil
        oil_vol = float(bench.get("oil", {}).get("annualized_volatility", 0.35))
        oil_mu = (_clip(bench["oil"]["mean_daily_drift"] * 252.0, -0.8, 0.8) if "oil" in bench else 0.0)
        oil_sigma = _clip(oil_vol * (1.0 + 0.5 * spr_low) * (1.0 + 0.3 * h) * (1.0 + 0.1 * (cable - 1.0)),
                          0.05, 3.0)
        oil_lam = 3.0 * (1.0 + 2.0 * h)
        oil_k = (0.06 + float(shocks.get("oil_supply_shock_pct", 0.0)) / 100.0 + 0.0005 * stress
                 + (0.05 if spr_low else 0.0))
        oil_delta = 0.10

        s0 = float(mkt["sp500"])
        o0 = float(mkt["oil_price"])
        sp_paths = self.merton_paths(s0, sp_mu, sp_sigma, sp_lam, sp_k, sp_delta,
                                     self.horizon_days, self.n_paths, self.rng)
        oil_paths = self.merton_paths(o0, oil_mu, oil_sigma, oil_lam, oil_k, oil_delta,
                                      self.horizon_days, self.n_paths, self.rng)
        self.last_paths = {"sp500": sp_paths, "oil": oil_paths}
        sp_stats, oil_stats = self._risk_stats(sp_paths), self._risk_stats(oil_paths)

        if shocks.get("price_update_mode", "median") == "sample":
            idx = int(self.rng.integers(0, self.n_paths))
            new_sp, new_oil = float(sp_paths[idx, -1]), float(oil_paths[idx, -1])
        else:
            new_sp, new_oil = sp_stats["median_terminal"], oil_stats["median_terminal"]

        mkt["sp500_prev"], mkt["oil_price_prev"] = s0, o0
        mkt["sp500"], mkt["oil_price"] = round(new_sp, 2), round(new_oil, 2)
        mkt["sp500_change_pct"] = round((new_sp / s0 - 1.0) * 100.0, 3)
        mkt["oil_change_pct"] = round((new_oil / o0 - 1.0) * 100.0, 3)
        mkt["sp500_var_95"] = round(sp_stats["var_95_pct"], 3)
        mkt["sp500_cvar_95"] = round(sp_stats["cvar_95_pct"], 3)
        mkt["oil_var_95"] = round(oil_stats["var_95_pct"], 3)
        mkt["oil_cvar_95"] = round(oil_stats["cvar_95_pct"], 3)
        ws["sp500_var_95"], ws["sp500_cvar_95"] = mkt["sp500_var_95"], mkt["sp500_cvar_95"]

        return {
            "sp500": {**sp_stats, "sp500_var_95": sp_stats["var_95_pct"],
                      "sp500_cvar_95": sp_stats["cvar_95_pct"], "paths": sp_paths,
                      "params": {"mu": sp_mu, "sigma": sp_sigma, "lambda": sp_lam, "k": sp_k,
                                 "delta": sp_delta}},
            "oil": {**oil_stats, "paths": oil_paths,
                    "params": {"mu": oil_mu, "sigma": oil_sigma, "lambda": oil_lam, "k": oil_k,
                               "delta": oil_delta}},
            "cable_multiplier": cable,
        }

    def export_paths_csv(self, name: str = "mc_paths") -> List[Path]:
        """Write percentile bands (5/25/50/75/95) of the last MC run to exports/."""
        self.export_dir.mkdir(parents=True, exist_ok=True)
        out: List[Path] = []
        for asset, paths in self.last_paths.items():
            bands = np.percentile(paths, [5, 25, 50, 75, 95], axis=0).T
            data = np.column_stack([np.arange(paths.shape[1]), bands])
            target = self.export_dir / f"{name}_{asset}.csv"
            np.savetxt(target, data, delimiter=",", header="day,p05,p25,p50,p75,p95",
                       comments="", fmt="%.4f")
            out.append(target)
        return out

    # =====================================================================
    # MODEL 3 - Trade graph, shadow fleet, supply-chain stress, inflation
    # =====================================================================
    def build_trade_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for name, a in self.entities.items():
            G.add_node(name, coalition=a.get("coalition"))
        for key, flow in self.world_state["trade_corridors_status"].items():
            if "->" not in key:
                continue
            src, dst = (s.strip() for s in key.split("->", 1))
            f = _clip(float(flow), 0.0, 1.0)
            G.add_edge(src, dst, flow=f, weight=f, distance=1.0 / max(f, 0.05), key=key)
        return G

    def _incident_corridor_loss(self) -> Dict[str, float]:
        loss: Dict[str, List[float]] = {}
        for key, flow in self.world_state["trade_corridors_status"].items():
            if "->" not in key:
                continue
            for node in (s.strip() for s in key.split("->", 1)):
                loss.setdefault(node, []).append(1.0 - _clip(float(flow), 0.0, 1.0))
        return {n: float(np.mean(v)) for n, v in loss.items()}

    def _refresh_inflation(self) -> float:
        ws, mkt = self.world_state, self.world_state["market"]
        oil_pct = (mkt["oil_price"] / max(mkt["oil_baseline"], 1e-9) - 1.0) * 100.0
        target = 2.0 + 0.045 * ws["supply_chain_stress_index"] + 0.05 * max(0.0, oil_pct)
        ws["inflation_pct"] = round(_clip(0.6 * ws["inflation_pct"] + 0.4 * target, 0.0, 40.0), 3)
        return ws["inflation_pct"]

    def update_trade_network(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws = self.world_state
        corridors = ws["trade_corridors_status"]
        for key, delta in (shocks.get("corridor_flow_changes") or {}).items():
            corridors[key] = round(_clip(float(corridors.get(key, 1.0)) + float(delta), 0.0, 1.0), 4)

        # Leaky bucket / shadow-fleet evasion: weak corridors self-heal 0.08/turn
        # unless the US executed secondary sanctions on that corridor.
        blocked_all = bool(ws["us_secondary_sanctions_executed"]) and not ws["secondary_sanction_corridors"]
        blocked_list = set(ws["secondary_sanction_corridors"])
        recovered: List[str] = []
        for key, flow in list(corridors.items()):
            if flow < 0.6 and not (blocked_all or key in blocked_list):
                corridors[key] = round(min(1.0, flow + 0.08), 4)
                recovered.append(key)

        G = self.build_trade_graph()
        edge_bc = nx.edge_betweenness_centrality(G, weight="distance", normalized=True)
        importance = {e: 1.0 + 6.0 * edge_bc.get(e, 0.0) for e in G.edges}
        den = sum(importance.values())
        num = sum(importance[e] * (1.0 - G.edges[e]["flow"]) for e in G.edges)
        base_stress = 100.0 * num / den if den else 0.0
        cable_pen = 12.0 * (self._cable_multiplier() - 1.0) / 0.9
        ws["supply_chain_stress_index"] = round(_clip(base_stress + cable_pen, 0.0, 100.0), 3)
        inflation = self._refresh_inflation()
        try:
            btw = nx.betweenness_centrality(G, weight="distance", normalized=True)
        except Exception:
            btw = {}
        return {
            "supply_chain_stress_index": ws["supply_chain_stress_index"],
            "inflation_pct": inflation,
            "shadow_fleet_recovered": recovered,
            "corridors": dict(corridors),
            "node_betweenness": btw,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
        }

    # =====================================================================
    # MODEL 4 - Strategic stockpiles, futures curve, tech leakage
    # =====================================================================
    def update_strategic_stockpiles(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws = self.world_state
        spr = float(ws["spr_oil_reserves_pct"])
        re = float(ws["rare_earth_defense_stockpile_pct"])
        mun = float(ws["precision_munitions_magazine_pct"])
        edge = float(ws["us_military_tech_edge"])
        oil = float(ws["market"]["oil_price"])
        oil_base = float(ws["market"]["oil_baseline"])

        draw = max(0.0, float(shocks.get("spr_drawdown_pct", 0.0)))
        spr_target = 70.0
        refill_rate = 0.04 if oil < 1.4 * oil_base else 0.0          # no refilling in a price spike
        spr = _clip(spr - draw + refill_rate * (spr_target - spr), 0.0, 100.0)  # dSPR/dt Euler, dt=1 turn

        expended = max(0.0, float(shocks.get("munitions_expended_pct", 0.0)))
        china_flow = float(ws["trade_corridors_status"].get("China->US", 1.0))
        re_supply = 0.4 + 0.6 * _clip(china_flow, 0.0, 1.0)           # 40% domestic processing floor
        re_consume = float(shocks.get("rare_earth_consumption_pct", 0.0)) + 0.35 * expended
        re = _clip(re + 0.05 * (65.0 - re) * re_supply - re_consume, 0.0, 100.0)

        production = 6.0 * (re / 100.0) * (1.0 - mun / 100.0)          # industrial base limited by REE
        mun = _clip(mun - expended + production, 0.0, 100.0)

        rival_def = float(np.mean([a["defense"] for a in self.agents.values()
                                   if a.get("coalition") == RIVAL_AXIS] or [60.0]))
        leakage = 0.15 * expended * (rival_def / 100.0) * (self._cable_multiplier())  # reverse-engineering
        innovation = 0.25 * (re / 100.0)
        edge = _clip(edge - leakage + innovation - 0.05 * max(0.0, 40.0 - mun), 0.0, 100.0)

        if spr < 25.0:
            curve = "Extreme_Backwardation"
        elif spr < 40.0:
            curve = "Backwardation"
        elif spr > 85.0:
            curve = "Contango"
        else:
            curve = "Flat"
        ws.update({
            "spr_oil_reserves_pct": round(spr, 3),
            "rare_earth_defense_stockpile_pct": round(re, 3),
            "precision_munitions_magazine_pct": round(mun, 3),
            "us_military_tech_edge": round(edge, 3),
            "oil_futures_curve": curve,
            "oil_convenience_yield_pct": round(max(0.0, 25.0 - spr) * 0.6, 3),
        })
        return {"spr_oil_reserves_pct": ws["spr_oil_reserves_pct"],
                "rare_earth_defense_stockpile_pct": ws["rare_earth_defense_stockpile_pct"],
                "precision_munitions_magazine_pct": ws["precision_munitions_magazine_pct"],
                "us_military_tech_edge": ws["us_military_tech_edge"],
                "oil_futures_curve": curve, "tech_leakage": leakage}

    # =====================================================================
    # MODEL 5 - Hawkes escalation + Bayesian fog-of-war false alarm
    # =====================================================================
    def update_hawkes_escalation(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws = self.world_state
        hp = ws["hawkes_params"]
        events = [float(x) for x in (shocks.get("escalation_events") or [])]
        events += [float(x) for x in ws.get("pending_escalation_events", [])]
        ws["pending_escalation_events"] = []
        if self._cable_multiplier() > 1.0:
            events.append(0.3)
        events.append(0.5 * max(0.0, float(shocks.get("munitions_expended_pct", 0.0))) / 10.0)
        events = [_clip(e, 0.0, 3.0) for e in events if e > 0]

        # lambda_t = mu0 + lambda_{t-1} e^{-beta} + sum(alpha * shocks)
        lam = (hp["mu0"] + ws["hawkes_escalation_intensity"] * math.exp(-hp["beta"])
               + hp["alpha"] * sum(events))
        lam = _clip(lam, 0.0, hp["cap"])

        # Bayesian false-alarm model
        cable_glitch = _clip((self._cable_multiplier() - 1.0) / 0.9, 0.0, 1.0)
        jam = _clip(float(shocks.get("gps_satellite_jamming", 0.0)), 0.0, 1.0)
        prior_attack = 1.0 - math.exp(-0.02 * lam)
        tpr = _clip(0.97 * (1.0 - 0.25 * cable_glitch - 0.15 * jam), 0.3, 1.0)
        fpr = _clip(0.0008 * (1.0 + 4.0 * cable_glitch + 3.0 * jam + 0.5 * lam), 0.0, 0.2)
        p_alarm = tpr * prior_attack + fpr * (1.0 - prior_attack)
        posterior_real = tpr * prior_attack / p_alarm if p_alarm > 0 else 0.0
        readiness = min(1.0, 0.25 + 0.12 * lam)
        p_false_defcon2 = fpr * (1.0 - prior_attack) * readiness
        triggered = bool(self.rng.random() < p_false_defcon2)
        if triggered:
            lam = min(hp["cap"], lam + 0.5 * hp["alpha"])
            ws["pending_escalation_events"].append(0.25)
        ws["hawkes_escalation_intensity"] = round(lam, 5)
        ws["defcon2_false_alarm_probability"] = round(p_false_defcon2, 6)
        ws["defcon2_false_alarm_triggered"] = triggered
        return {"hawkes_escalation_intensity": ws["hawkes_escalation_intensity"],
                "events_applied": events, "prior_real_attack": prior_attack,
                "sensor_tpr": tpr, "sensor_fpr": fpr,
                "posterior_real_given_alarm": posterior_real,
                "p_defcon2_false_alarm": p_false_defcon2,
                "false_alarm_triggered": triggered}

    # =====================================================================
    # MODEL 8 (part) - ESF short squeeze
    # =====================================================================
    def esf_short_squeeze(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws, mkt = self.world_state, self.world_state["market"]
        request = max(0.0, float(shocks.get("esf_squeeze_billions", 0.0)))
        short = float(ws["hostile_short_interest_billions"])
        if request <= 0 or short <= 0:
            return {"executed": False, "deployed_billions": 0.0}
        deployed = min(request, float(ws["esf_capacity_billions"]))
        forced_cover = _clip(0.6 * deployed / short, 0.0, 1.0)
        covered = short * forced_cover
        ws["hostile_short_interest_billions"] = round(short - covered, 3)
        ws["esf_capacity_billions"] = round(float(ws["esf_capacity_billions"]) - 0.25 * deployed, 3)
        mkt["sp500"] = round(mkt["sp500"] * (1.0 + 0.03 * forced_cover), 2)
        return {"executed": True, "deployed_billions": deployed, "forced_cover_fraction": forced_cover,
                "hostile_losses_billions": 0.35 * covered,
                "remaining_short_interest": ws["hostile_short_interest_billions"],
                "esf_capacity_billions": ws["esf_capacity_billions"]}

    # =====================================================================
    # MODEL 2 - Merton structural default + Eisenberg-Noe
    # =====================================================================
    def _build_liabilities(self, names: List[str], gdp: np.ndarray) -> np.ndarray:
        idx = {n: i for i, n in enumerate(names)}
        L = np.zeros((len(names), len(names)))
        for key, flow in self.world_state["trade_corridors_status"].items():
            if "->" not in key:
                continue
            exporter, importer = (s.strip() for s in key.split("->", 1))
            if exporter in idx and importer in idx:      # importer owes exporter trade credit
                L[idx[importer], idx[exporter]] += 0.004 * gdp[idx[importer]] * _clip(float(flow), 0.0, 1.0)
        for (debtor, creditor), amt in CROSS_HOLDINGS.items():
            if debtor in idx and creditor in idx:
                L[idx[debtor], idx[creditor]] += amt
        for name, a in self.entities.items():            # explicit overrides
            for creditor, amt in (a.get("liabilities") or {}).items():
                if name in idx and creditor in idx and name != creditor:
                    L[idx[name], idx[creditor]] += float(amt)
        np.fill_diagonal(L, 0.0)
        return L

    @staticmethod
    def eisenberg_noe(L: np.ndarray, e: np.ndarray, tol: float = 1e-9,
                      max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Greatest clearing payment vector via fixed-point iteration. Returns (p, ratio p/pbar)."""
        pbar = L.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Pi = np.where(pbar[:, None] > 0, L / pbar[:, None], 0.0)
        p = pbar.copy()
        for _ in range(max_iter):
            p_new = np.minimum(pbar, np.maximum(e + Pi.T @ p, 0.0))
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pbar > 0, p / pbar, 1.0)
        return p, ratio

    def update_sovereign_credit(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws, mkt = self.world_state, self.world_state["market"]
        ents = self.entities
        names = list(ents)
        r = max(float(ws["fed_funds_rate"]), 0.0) / 100.0
        stress = float(ws["supply_chain_stress_index"]) / 100.0
        h = self._hawkes_norm()
        sp_dd = max(0.0, -float(mkt.get("sp500_change_pct", 0.0))) / 100.0
        incident = self._incident_corridor_loss()
        war_in = shocks.get("war_damage_pct") or {}
        sanc_in = shocks.get("sanctions_severity") or {}

        V = np.zeros(len(names)); V0 = np.zeros(len(names))
        stand: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(names):
            a = ents[name]
            a["war_damage_pct"] = _clip(a.get("war_damage_pct", 0.0) + float(war_in.get(name, 0.0)), 0.0, 90.0)
            a["sanctions_severity"] = _clip(a.get("sanctions_severity", 0.0) + float(sanc_in.get(name, 0.0)), 0.0, 1.0)
            v0 = a["gdp_bn"] * 3.0
            beta = 0.5 if name == "US" else 0.3
            factor = (1.0 - a["war_damage_pct"] / 100.0 - 0.25 * a["sanctions_severity"]
                      - 0.30 * incident.get(name, 0.0) - 0.15 * stress - beta * sp_dd)
            v = max(v0 * factor, 0.05 * v0)
            d = a["leverage"] * v0
            sigma = a["asset_vol"] * (1.0 + 0.8 * h + 0.5 * stress)
            dd = (math.log(v / d) + (r - 0.5 * sigma ** 2)) / sigma     # T = 1y
            pd = float(norm.cdf(-dd))
            V[i], V0[i] = v, v0
            stand[name] = {"dd": dd, "pd": pd}

        gdp = np.array([ents[n]["gdp_bn"] for n in names], dtype=float)
        fx = np.array([ents[n]["fx_res_bn"] for n in names], dtype=float)
        L = self._build_liabilities(names, gdp)
        e = 0.08 * V + fx * (V / V0) ** 2
        p, ratio = self.eisenberg_noe(L, e)

        out: Dict[str, Any] = {}
        recovery = 0.40
        for i, name in enumerate(names):
            a = ents[name]
            shortfall = float(1.0 - ratio[i])
            pd_final = _clip(stand[name]["pd"] + (1.0 - stand[name]["pd"]) * 0.5 * shortfall, 0.0, 0.999)
            hazard = -math.log(1.0 - pd_final)
            cds = _clip(1e4 * hazard * (1.0 - recovery), 8.0, 6000.0)
            a["distance_to_default"] = round(stand[name]["dd"], 4)
            a["pd_standalone"] = round(stand[name]["pd"], 6)
            a["pd"] = round(pd_final, 6)
            a["cds_spread_bps"] = round(cds, 2)
            a["en_clearing_ratio"] = round(float(ratio[i]), 6)
            out[name] = {"distance_to_default": a["distance_to_default"], "pd": a["pd"],
                         "cds_spread_bps": a["cds_spread_bps"], "en_clearing_ratio": a["en_clearing_ratio"]}
        return {"agents": out,
                "systemic_shortfall_billions": float(np.sum(L.sum(axis=1) - p)),
                "n_in_default_cascade": int(np.sum(ratio < 0.999)),
                "total_obligations_billions": float(L.sum())}

    # =====================================================================
    # MODEL 8 - Augmented Taylor rule, yield curve
    # =====================================================================
    def update_rates(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws, mkt = self.world_state, self.world_state["market"]
        pi, pi_star, r_star = float(ws["inflation_pct"]), 2.0, 2.0
        sp_drop = max(0.0, -float(mkt.get("sp500_change_pct", 0.0)))
        stress = float(ws["supply_chain_stress_index"])
        gap = 0.4 - 0.10 * sp_drop - 0.03 * stress                       # output-gap proxy, %
        taylor = r_star + pi + 0.5 * (pi - pi_star) + 0.5 * gap
        us_cds = float(ws["us_profile"].get("cds_spread_bps", 8.0))
        dom = ws["us_domestic_politics"]
        fiscal = (0.6 * max(0.0, us_cds - 8.0) / 100.0
                  + 0.15 * float(shocks.get("war_spending_pct_gdp", 0.0))
                  + 0.004 * max(0.0, dom["war_exhaustion_index"] - 30.0))
        fed = _clip(0.65 * ws["fed_funds_rate"] + 0.35 * taylor, 0.0, 15.0)
        flight = _clip(sp_drop / 20.0 + 0.5 * self._hawkes_norm(), 0.0, 1.0)
        y2 = _clip(fed + 0.5 * (taylor - fed) - 0.30 * flight + 0.5 * fiscal, 0.0, 20.0)
        y10 = _clip(0.45 * y2 + 0.55 * (r_star + pi_star + 0.3 * (pi - pi_star)) + 1.2 * fiscal
                    - 0.5 * flight, 0.0, 20.0)
        ws["fed_funds_rate"] = round(fed, 4)
        ws["treasury_yield_2y"] = round(y2, 4)
        ws["treasury_yield_10y"] = round(y10, 4)
        mkt["ten_year_yield"] = ws["treasury_yield_10y"]
        ws["yield_curve_inverted"] = bool(y2 > y10)
        return {"taylor_rate": taylor, "output_gap": gap, "fiscal_risk_premium": fiscal,
                "fed_funds_rate": ws["fed_funds_rate"], "treasury_yield_2y": ws["treasury_yield_2y"],
                "treasury_yield_10y": ws["treasury_yield_10y"],
                "yield_curve_inverted": ws["yield_curve_inverted"]}

    # =====================================================================
    # MODEL 7 - Prospect theory + US domestic politics
    # =====================================================================
    @staticmethod
    def prospect_value(x: float) -> float:
        """Kahneman-Tversky value function: gains x^0.88, losses -2.25 * (-x)^0.88."""
        return float(x ** 0.88) if x >= 0 else float(-2.25 * ((-x) ** 0.88))

    def update_leader_psychology(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        hum = shocks.get("humiliated_leaders") or {}
        if isinstance(hum, (list, tuple, set)):
            hum = {n: 3.0 for n in hum}
        out: Dict[str, Any] = {}
        for name, a in self.entities.items():
            sev = max(0.0, float(hum.get(name, 0.0)))
            stock = 0.85 * a.get("humiliation_stock", 0.0) + sev
            v = self.prospect_value(-stock) if stock > 0 else 0.0
            base = a.get("base_risk_tolerance", 0.4)
            target = base + (1.0 - base) * (1.0 - math.exp(-abs(v) / 8.0))
            a["risk_tolerance"] = round(_clip(0.5 * a.get("risk_tolerance", base) + 0.5 * target, 0.0, 1.0), 4)
            a["humiliation_stock"] = round(stock, 4)
            a["in_loss_domain"] = bool(sev > 0 or stock >= 0.5)
            a["prospect_value"] = round(v, 4)
            out[name] = {"in_loss_domain": a["in_loss_domain"], "risk_tolerance": a["risk_tolerance"],
                         "prospect_value": a["prospect_value"]}
        return out

    def update_us_domestic_politics(self, shocks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        shocks = shocks or {}
        ws, mkt = self.world_state, self.world_state["market"]
        d = ws["us_domestic_politics"]
        attacked = max(0.0, float(shocks.get("us_attacked_severity", 0.0)))
        casualties = max(0.0, float(shocks.get("us_casualties", 0.0)))
        war_int = _clip(float(shocks.get("war_intensity", 0.0)), 0.0, 1.0)
        order = _clip(float(shocks.get("controversial_order_severity", 0.0)), 0.0, 1.0)
        chg = float(mkt.get("sp500_change_pct", 0.0))
        sp_drop, sp_gain = max(0.0, -chg), max(0.0, chg)

        d["rally_effect"] = _clip(d["rally_effect"] * math.exp(-0.12) + 6.0 * attacked
                                  - 0.02 * casualties, 0.0, 60.0)
        d["war_exhaustion_index"] = _clip(
            d["war_exhaustion_index"] + 6.0 * war_int + 0.15 * casualties
            + 0.8 * max(0.0, ws["inflation_pct"] - 3.0) - 0.04 * d["rally_effect"]
            - 1.0 * (1.0 - war_int), 0.0, 100.0)
        d["wall_street_oligarch_loyalty"] = _clip(
            d["wall_street_oligarch_loyalty"] - 0.9 * sp_drop + 0.5 * sp_gain
            - 0.01 * ws["hostile_short_interest_billions"] + 0.05 * (70.0 - d["wall_street_oligarch_loyalty"]),
            0.0, 100.0)
        d["pentagon_joint_chiefs_obedience"] = _clip(
            d["pentagon_joint_chiefs_obedience"] - 8.0 * order
            - 0.04 * max(0.0, d["war_exhaustion_index"] - 50.0)
            + 0.05 * (75.0 - d["pentagon_joint_chiefs_obedience"]), 0.0, 100.0)
        z = (-3.5 + 0.03 * d["war_exhaustion_index"] + 0.02 * (100.0 - d["wall_street_oligarch_loyalty"])
             + 0.02 * (100.0 - d["pentagon_joint_chiefs_obedience"]) - 0.02 * d["rally_effect"]
             + 1.5 * order)
        d["deep_state_25th_amendment_risk"] = round(100.0 * _sigmoid(z), 3)
        for k in ("rally_effect", "war_exhaustion_index", "wall_street_oligarch_loyalty",
                  "pentagon_joint_chiefs_obedience"):
            d[k] = round(d[k], 3)
        return dict(d)

    # =====================================================================
    # MODEL 6 - Shapley values and 3x3 Nash
    # =====================================================================
    def _power(self, a: Dict[str, Any]) -> float:
        return (0.55 * a["mil"] + 0.45 * a["econ"]) * (1.0 - a.get("war_damage_pct", 0.0) / 100.0)

    @staticmethod
    def _coalition_value(power: Dict[str, float], trust: Dict[str, float], members: Sequence[str]) -> float:
        if not members:
            return 0.0
        total = sum(power[m] for m in members)
        mean_trust = sum(trust[m] for m in members) / (100.0 * len(members))
        return (total ** 0.85) * (1.0 + 0.3 * mean_trust)

    def compute_shapley_values(self, n_perms: int = 1500) -> Dict[str, float]:
        """Permutation-sampling Shapley value of each agent inside its own coalition."""
        ents = self.entities
        power = {n: self._power(a) for n, a in ents.items()}
        out: Dict[str, float] = {}
        for coalition in (US_COALITION, RIVAL_AXIS, NON_ALIGNED):
            group = [n for n, a in ents.items() if a.get("coalition") == coalition]
            if not group:
                continue
            if coalition == RIVAL_AXIS:
                trust = {n: max(ents[n].get("axis_trust", 100.0 - ents[n]["trust"]), 0.0) for n in group}
            elif coalition == NON_ALIGNED:
                trust = {n: 50.0 for n in group}
            else:
                trust = {n: ents[n]["trust"] for n in group}
            phi = {n: 0.0 for n in group}
            for _ in range(n_perms):
                order = self.rng.permutation(len(group))
                current: List[str] = []
                prev = 0.0
                for j in order:
                    current.append(group[j])
                    val = self._coalition_value(power, trust, current)
                    phi[group[j]] += val - prev
                    prev = val
            for n in group:
                out[n] = phi[n] / n_perms
        for n, a in ents.items():
            a["shapley_coalition_value"] = round(out.get(n, 0.0), 4)
        return out

    @staticmethod
    def _nash_support_enumeration(A: np.ndarray, B: np.ndarray) -> List[Dict[str, Any]]:
        m, n = A.shape
        found: List[Dict[str, Any]] = []
        seen = set()
        for k in range(1, min(m, n) + 1):
            for I in itertools.combinations(range(m), k):
                for J in itertools.combinations(range(n), k):
                    try:
                        M1 = np.zeros((k + 1, k + 1))
                        M1[:k, :k] = A[np.ix_(I, J)]
                        M1[:k, k] = -1.0
                        M1[k, :k] = 1.0
                        rhs = np.zeros(k + 1); rhs[k] = 1.0
                        s1 = np.linalg.solve(M1, rhs)                    # col player's mix y and u1
                        M2 = np.zeros((k + 1, k + 1))
                        M2[:k, :k] = B[np.ix_(I, J)].T
                        M2[:k, k] = -1.0
                        M2[k, :k] = 1.0
                        s2 = np.linalg.solve(M2, rhs)                    # row player's mix x and u2
                    except np.linalg.LinAlgError:
                        continue
                    y, u1, x, u2 = s1[:k], s1[k], s2[:k], s2[k]
                    if (y < -1e-9).any() or (x < -1e-9).any():
                        continue
                    yf, xf = np.zeros(n), np.zeros(m)
                    yf[list(J)], xf[list(I)] = np.clip(y, 0, None), np.clip(x, 0, None)
                    if (A @ yf).max() > u1 + 1e-9 or (xf @ B).max() > u2 + 1e-9:
                        continue
                    sig = (tuple(np.round(xf, 5)), tuple(np.round(yf, 5)))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    found.append({"x": xf, "y": yf, "u1": float(u1), "u2": float(u2)})
        return found

    @staticmethod
    def _zero_sum_maximin(A: np.ndarray) -> Tuple[np.ndarray, float]:
        """Row player's maximin strategy via linprog (fallback for degenerate games)."""
        m, n = A.shape
        c = np.zeros(m + 1); c[-1] = -1.0
        A_ub = np.hstack([-A.T, np.ones((n, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=[[1.0] * m + [0.0]], b_eq=[1.0],
                      bounds=[(0, None)] * m + [(None, None)], method="highs")
        x = np.clip(res.x[:m], 0, None)
        return x / x.sum(), float(res.x[m])

    def solve_superpower_nash(self) -> Dict[str, Any]:
        """3x3 mixed-strategy Nash between the US Coalition and the Rival Axis."""
        ents = self.entities
        power = {n: self._power(a) for n, a in ents.items()}
        trust = {n: a["trust"] for n, a in ents.items()}
        us_group = [n for n, a in ents.items() if a.get("coalition") == US_COALITION]
        rv_group = [n for n, a in ents.items() if a.get("coalition") == RIVAL_AXIS]
        v_us = self._coalition_value(power, trust, us_group)
        v_rv = self._coalition_value(power, {n: 100.0 - ents[n]["trust"] for n in rv_group}, rv_group)
        r = v_us / (v_us + v_rv) if (v_us + v_rv) > 0 else 0.5
        h = self._hawkes_norm()
        w = self.world_state["us_domestic_politics"]["war_exhaustion_index"] / 100.0
        rt = float(np.mean([ents[n]["risk_tolerance"] for n in rv_group] or [0.5]))
        edge = self.world_state["us_military_tech_edge"] / 100.0
        r_eff = _clip(r * (0.8 + 0.4 * edge), 0.05, 0.95)

        A = np.array([
            [1.5 * r_eff - 0.5 * h, 1.0 * r_eff - 1.5 * h, 3.0 * r_eff + 0.5 * (1 - h)],
            [2.5 * r_eff - h * (1 + rt) - 0.5 * w, 2.0 * r_eff - 2.5 * h * (1 + rt) - w, 3.5 * r_eff],
            [-1.5 * (1 - r_eff) - 0.5 * h, -1.0 * (1 - r_eff) - 0.2 * h, 1.0 + 0.5 * (1 - r_eff)],
        ])
        B = np.array([
            [2.0 * (1 - r_eff) * (1 + rt), 1.5 * (1 - r_eff), -3.0 * r_eff],
            [1.0 * (1 - r_eff) - 1.5 * h, 2.0 * (1 - r_eff) * (1 + rt) - 2.0 * h, -2.0 * r_eff],
            [2.5 * (1 - r_eff) * (1 + rt), 3.0 * (1 - r_eff) * (1 + rt), 0.5],
        ])
        sols = self._nash_support_enumeration(A, B)
        method = "support_enumeration"
        if sols:
            best = max(sols, key=lambda s: s["u1"] + s["u2"])
            x, y, u1, u2 = best["x"], best["y"], best["u1"], best["u2"]
        else:  # degenerate game: zero-sum maximin via scipy.optimize.linprog
            method = "linprog_zero_sum_fallback"
            x, u1 = self._zero_sum_maximin(A)
            y, neg_u = self._zero_sum_maximin(-A.T)
            u2 = float(x @ B @ y)
            u1 = float(x @ A @ y)
        result = {
            "method": method, "n_equilibria_found": len(sols), "coalition_power_share": r,
            "us_coalition_strategy": dict(zip(US_STRATEGIES, np.round(x, 5).tolist())),
            "rival_axis_strategy": dict(zip(RIVAL_STRATEGIES, np.round(y, 5).tolist())),
            "expected_payoff_us": float(u1), "expected_payoff_rival": float(u2),
            "payoff_matrix_us": A.tolist(), "payoff_matrix_rival": B.tolist(),
        }
        self.world_state["superpower_nash"] = result
        return result

    # =====================================================================
    # MODEL 9 - Espionage wheel, Five-Eyes pooling, kompromat
    # =====================================================================
    _KOMPROMAT_TEMPLATES = [
        ("offshore_finance", "Offshore shell-company holdings tied to {leader}'s inner circle"),
        ("health_concealment", "Concealed medical condition affecting {leader}'s capacity to govern"),
        ("covert_deal", "Back-channel arms/energy deal that contradicts {target}'s public position"),
        ("family_business", "Family members' business interests profiting from state contracts"),
        ("election_interference", "Evidence of domestic election manipulation by {target}'s security services"),
    ]

    def _append_diplomacy_log(self, entry: Dict[str, Any]) -> None:
        path = self.data_dir / "diplomacy_log.json"
        log = _read_json(path, [])
        if not isinstance(log, list):
            log = []
        log.append(entry)
        try:
            _write_json(path, log)
        except OSError:
            pass

    def spin_espionage_wheel(self, attacker: str, target: str, op_type: str = "HUMINT",
                             attacker_attack: Optional[float] = None,
                             target_defense: Optional[float] = None,
                             persist_log: bool = True) -> Dict[str, Any]:
        """
        P(intercept) = 1 / (1 + exp(-0.08 * (attacker_attack - target_defense - 15))).
        Wheel: spin ~ U(0,100). Top 3% => critical success (US/CIA also unlocks a
        Kompromat Dossier); bottom 4% => burned spy + diplomatic blowback; the
        middle is resolved by the logistic probability.
        """
        ents = self.entities
        if attacker not in ents or target not in ents:
            raise KeyError(f"Unknown agent in ({attacker}, {target})")
        atk = float(attacker_attack if attacker_attack is not None else ents[attacker]["attack"])
        dfn = float(target_defense if target_defense is not None else ents[target]["defense"])

        pool: List[str] = []
        bonus = 0.0
        if attacker == "US" or attacker in FIVE_EYES_POOL:
            pool = [n for n in ("US",) + FIVE_EYES_POOL
                    if n not in (attacker, target) and n in ents and ents[n].get("trust", 0.0) >= 75.0]
            bonus = min(20.0, 0.06 * sum(ents[n]["attack"] for n in pool))
        atk_eff = atk + bonus
        p = _sigmoid(0.08 * (atk_eff - dfn - 15.0))

        spin = float(self.rng.uniform(0.0, 100.0))
        turn = int(self.world_state.get("turn", 0))
        dossier = None
        blowback: Dict[str, Any] = {}
        if spin >= 97.0:
            outcome = "CRITICAL_SUCCESS"
        elif spin < 4.0:
            outcome = "BURNED"
        else:
            outcome = "SUCCESS" if self.rng.random() < p else "FAILURE"

        if outcome in ("SUCCESS", "CRITICAL_SUCCESS"):
            intel = ents[attacker].setdefault("intel_on", {})
            intel[target] = _clip(intel.get(target, 0.0) + (25.0 if outcome == "CRITICAL_SUCCESS" else 12.0),
                                  0.0, 100.0)
            if outcome == "CRITICAL_SUCCESS" and attacker == "US":
                cat, text = self.rng.choice(self._KOMPROMAT_TEMPLATES)
                dossier = {
                    "id": f"KOMP-{turn}-{len(self.world_state['kompromat_dossiers']) + 1}",
                    "target": target, "category": str(cat),
                    "description": str(text).format(leader=ents[target].get("leader", target), target=target),
                    "leverage_value": round(float(self.rng.uniform(15.0, 40.0)), 1),
                    "created_turn": turn, "used": False,
                }
                self.world_state["kompromat_dossiers"].append(dossier)
        elif outcome == "BURNED":
            sev = float(self.rng.uniform(10.0, 25.0))
            if attacker == "US":
                ents[target]["trust"] = _clip(ents[target]["trust"] - sev, 0.0, 100.0)
                for n, a in ents.items():
                    if n not in (attacker, target) and a.get("coalition") == NON_ALIGNED:
                        a["trust"] = _clip(a["trust"] - 3.0, 0.0, 100.0)
            else:
                g = ents[target].setdefault("grievances", {})
                g[attacker] = g.get(attacker, 0.0) + sev
            ents[attacker]["attack"] = _clip(ents[attacker]["attack"] - 3.0, 0.0, 100.0)
            self.world_state["pending_escalation_events"].append(0.15)
            blowback = {"trust_penalty": sev, "attacker_network_burned": True}
            self._append_diplomacy_log({"turn": turn, "type": "espionage_blowback",
                                        "attacker": attacker, "target": target, "severity": sev})
        result = {"attacker": attacker, "target": target, "op_type": op_type, "outcome": outcome,
                  "p_intercept": p, "spin": spin, "attacker_attack": atk, "target_defense": dfn,
                  "pooling_members": pool, "pooling_bonus": bonus, "kompromat": dossier,
                  "blowback": blowback}
        self.world_state["espionage_log"].append(
            {k: result[k] for k in ("attacker", "target", "op_type", "outcome", "p_intercept")} | {"turn": turn})
        if persist_log:
            self._append_diplomacy_log({"turn": turn, "type": "espionage", "attacker": attacker,
                                        "target": target, "outcome": outcome})
        return result

    def deploy_kompromat(self, dossier_id: str) -> Dict[str, Any]:
        """Spend a dossier: lowers the target's attack score and risk tolerance."""
        for d in self.world_state["kompromat_dossiers"]:
            if d["id"] == dossier_id and not d["used"]:
                t = self.entities[d["target"]]
                t["attack"] = _clip(t["attack"] - 0.2 * d["leverage_value"], 0.0, 100.0)
                t["risk_tolerance"] = _clip(t["risk_tolerance"] - 0.1, 0.0, 1.0)
                d["used"] = True
                return {"deployed": True, "dossier": d}
        return {"deployed": False, "reason": "unknown or already used dossier"}

    # =====================================================================
    # Orchestration
    # =====================================================================
    def run_turn(self, shocks: Optional[Dict[str, Any]] = None, persist: bool = True) -> Dict[str, Any]:
        """Run all models in dependency order and (optionally) persist state to data/."""
        shocks = dict(shocks or {})
        ws = self.world_state
        ws["turn"] = int(ws.get("turn", 0)) + 1
        self.apply_shock_inputs(shocks)
        out: Dict[str, Any] = {"turn": ws["turn"]}
        out["trade_network"] = self.update_trade_network(shocks)
        out["stockpiles"] = self.update_strategic_stockpiles(shocks)
        out["hawkes"] = self.update_hawkes_escalation(shocks)
        out["monte_carlo"] = self.monte_carlo_markets(shocks)
        out["esf"] = self.esf_short_squeeze(shocks)
        out["inflation_pct"] = self._refresh_inflation()
        out["sovereign"] = self.update_sovereign_credit(shocks)
        out["rates"] = self.update_rates(shocks)
        out["leader_psychology"] = self.update_leader_psychology(shocks)
        out["us_domestic_politics"] = self.update_us_domestic_politics(shocks)
        out["shapley"] = self.compute_shapley_values()
        out["nash"] = self.solve_superpower_nash()
        if persist:
            self.save()
        return out


if __name__ == "__main__":
    engine = QuantMacroEngine(seed=42, autoload=False)
    demo = engine.run_turn({
        "undersea_cables_status": "Cut",
        "hostile_short_interest_billions": 150,
        "corridor_flow_changes": {"China->US": -0.5, "Iran->China": -0.4},
        "oil_supply_shock_pct": 12, "spr_drawdown_pct": 10, "munitions_expended_pct": 6,
        "escalation_events": [0.6], "humiliated_leaders": {"Russia": 4.0},
        "us_attacked_severity": 2.0, "war_intensity": 0.5, "esf_squeeze_billions": 60,
    }, persist=False)
    sp = demo["monte_carlo"]["sp500"]
    print(f"S&P median {sp['median_terminal']:.0f}  VaR95 {sp['sp500_var_95']:.2f}%  "
          f"CVaR95 {sp['sp500_cvar_95']:.2f}%  paths {sp['paths'].shape}")
    print("Stress", demo["trade_network"]["supply_chain_stress_index"],
          "| Hawkes", demo["hawkes"]["hawkes_escalation_intensity"],
          "| Curve", engine.world_state["oil_futures_curve"])
    print("Nash US:", demo["nash"]["us_coalition_strategy"])
    print("Espionage:", engine.spin_espionage_wheel("US", "China")["outcome"])
