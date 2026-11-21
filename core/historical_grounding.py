"""
core/historical_grounding.py
============================
Historical calibration + scenario loading for the 20-Agent Geopolitical,
Espionage & Macroeconomic Simulator.

Responsibilities
----------------
1. Calibrate market-shock parameters (annualised volatility, mean daily drift,
   worst/best 30-day deltas) from real Yahoo Finance data (``yfinance``) for
   four historical shocks and cache them in ``data/historical_benchmarks.json``.
   If the network / yfinance is unavailable, realistic hard-coded fallback
   values are used (per asset, per shock) so the simulator always runs.
2. ``get_historical_market_state(date_str)`` -> real S&P 500, WTI crude and
   10Y yield for any date, optionally written into ``data/world_state.json``.
3. Free Wikipedia REST summary fetcher (``urllib``) for grounding the LLM
   Referee prompt with brief historical crisis context.

Only free libraries are used: numpy, yfinance (optional at runtime), urllib.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_FILE = DATA_DIR / "historical_benchmarks.json"
WORLD_STATE_FILE = DATA_DIR / "world_state.json"

TRADING_DAYS = 252
SHOCK_WINDOW_DAYS = 30
TICKERS: Dict[str, str] = {"sp500": "^GSPC", "yield_10y": "^TNX", "oil": "CL=F"}
WIKI_USER_AGENT = "GeoMacroSimulator/1.0 (educational research; python-urllib)"

HISTORICAL_SHOCKS: Dict[str, Dict[str, str]] = {
    "2018_trade_war": {
        "label": "2018 US-China Trade War",
        "start": "2018-03-22",
        "end": "2018-12-24",
        "wiki_title": "China–United_States_trade_war",
    },
    "2020_covid_supply_freeze": {
        "label": "2020 COVID Supply Freeze",
        "start": "2020-02-19",
        "end": "2020-04-30",
        "wiki_title": "Economic_impact_of_the_COVID-19_pandemic",
    },
    "2022_energy_sanctions": {
        "label": "2022 Energy / Sanctions Shock",
        "start": "2022-02-23",
        "end": "2022-07-31",
        "wiki_title": "Economic_impact_of_the_Russian_invasion_of_Ukraine",
    },
    "1990_gulf_oil_spike": {
        "label": "1990 Gulf Oil Price Spike",
        "start": "1990-07-02",
        "end": "1990-10-31",
        "wiki_title": "1990_oil_price_shock",
    },
}

# (annualised_vol, mean_daily_log_drift, worst-or-largest 30d move in %)
# Approximate values reflecting the real episodes; used only when live data
# is unavailable (offline, rate-limited, or a ticker does not exist yet, e.g.
# CL=F has no Yahoo history before 2000).
_FALLBACKS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "2018_trade_war": {
        "^GSPC": (0.165, -0.00035, -9.2),
        "^TNX": (0.220, 0.00040, 9.0),
        "CL=F": (0.380, -0.00150, -27.0),
    },
    "2020_covid_supply_freeze": {
        "^GSPC": (0.780, -0.00300, -26.0),
        "^TNX": (0.950, -0.01200, -58.0),
        "CL=F": (1.350, -0.01200, -55.0),
    },
    "2022_energy_sanctions": {
        "^GSPC": (0.250, -0.00040, -13.5),
        "^TNX": (0.420, 0.00500, 55.0),
        "CL=F": (0.550, 0.00080, 28.0),
    },
    "1990_gulf_oil_spike": {
        "^GSPC": (0.210, -0.00130, -14.5),
        "^TNX": (0.150, 0.00030, 6.0),
        "CL=F": (1.100, 0.01000, 85.0),
    },
}

# Approximate anchor values (S&P 500, WTI $/bbl, 10Y yield %) used only when
# live data for a date cannot be downloaded.
_FALLBACK_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    "1990-07-02": (358.0, 17.5, 8.40),
    "1990-08-02": (355.0, 23.0, 8.95),
    "1990-10-11": (295.0, 40.0, 8.65),
    "2018-03-22": (2643.0, 65.0, 2.82),
    "2018-10-03": (2925.0, 76.4, 3.19),
    "2018-12-24": (2351.0, 42.5, 2.74),
    "2020-02-19": (3386.0, 53.4, 1.56),
    "2020-03-23": (2237.0, 23.4, 0.76),
    "2020-04-30": (2912.0, 19.9, 0.64),
    "2022-02-23": (4225.0, 92.4, 1.97),
    "2022-03-08": (4170.0, 123.7, 1.88),
    "2022-07-29": (4130.0, 98.6, 2.67),
}

_FALLBACK_CONTEXT: Dict[str, str] = {
    "2018_trade_war": (
        "In 2018 the United States imposed escalating tariffs on Chinese goods and China "
        "retaliated, hitting global supply chains, equities and commodity prices; the S&P 500 "
        "fell sharply in Q4 2018 amid tightening Fed policy and trade-war uncertainty."
    ),
    "2020_covid_supply_freeze": (
        "The COVID-19 pandemic froze global production and travel in early 2020, triggering the "
        "fastest bear market on record, a collapse in oil demand and emergency central-bank action."
    ),
    "2022_energy_sanctions": (
        "Russia's 2022 invasion of Ukraine brought sweeping Western sanctions, an energy-price "
        "spike, European gas supply disruption, surging inflation and aggressive rate hikes."
    ),
    "1990_gulf_oil_spike": (
        "Iraq's invasion of Kuwait in August 1990 caused crude oil prices to roughly double within "
        "weeks, tipping the US toward recession and driving equities down about 20 percent."
    ),
}

_CONTEXT_CACHE: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
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


# --------------------------------------------------------------------------- #
# yfinance access
# --------------------------------------------------------------------------- #
def _download_series(ticker: str, start: str, end: str) -> Optional[Tuple[List[Any], np.ndarray]]:
    """Return (dates, closes) for ``ticker`` in [start, end) or None on any failure."""
    try:
        import yfinance as yf  # imported lazily so the module loads offline
    except ImportError:
        logger.warning("yfinance not installed; using fallback data for %s", ticker)
        return None
    try:
        tkr = yf.Ticker(ticker)
        try:
            hist = tkr.history(start=start, end=end, interval="1d",
                               auto_adjust=False, actions=False, timeout=15)
        except TypeError:  # older yfinance without timeout kwarg
            hist = tkr.history(start=start, end=end, interval="1d",
                               auto_adjust=False, actions=False)
    except Exception as exc:  # network, rate-limit, parsing...
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None
    if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
        return None
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return None
    dates = [d.date() for d in hist.index.to_pydatetime()]
    closes = np.asarray(hist["Close"], dtype=float)
    return dates, closes


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _compute_stats(closes: np.ndarray) -> Optional[Dict[str, float]]:
    """Annualised vol, mean daily log drift and 30-trading-day shock deltas."""
    closes = np.asarray(closes, dtype=float)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if closes.size < 10:
        return None
    log_ret = np.diff(np.log(closes))
    vol = float(np.std(log_ret, ddof=1) * math.sqrt(TRADING_DAYS))
    drift = float(np.mean(log_ret))
    if closes.size > SHOCK_WINDOW_DAYS:
        pct = (closes[SHOCK_WINDOW_DAYS:] / closes[:-SHOCK_WINDOW_DAYS] - 1.0) * 100.0
    else:  # window shorter than 30 obs: use whole-window change
        pct = np.array([(closes[-1] / closes[0] - 1.0) * 100.0])
    worst, best = float(np.min(pct)), float(np.max(pct))
    delta = worst if abs(worst) >= abs(best) else best
    return {
        "annualized_volatility": round(vol, 6),
        "mean_daily_drift": round(drift, 8),
        "shock_delta_30d_pct": round(delta, 3),
        "worst_30d_pct": round(worst, 3),
        "best_30d_pct": round(best, 3),
        "n_obs": int(closes.size),
    }


def _fallback_stats(shock_key: str, ticker: str) -> Dict[str, Any]:
    vol, drift, delta = _FALLBACKS[shock_key][ticker]
    return {
        "annualized_volatility": vol,
        "mean_daily_drift": drift,
        "shock_delta_30d_pct": delta,
        "worst_30d_pct": min(delta, 0.0),
        "best_30d_pct": max(delta, 0.0),
        "n_obs": 0,
        "source": "fallback",
    }


# --------------------------------------------------------------------------- #
# 1. Historical market shock calibration
# --------------------------------------------------------------------------- #
def load_benchmarks() -> Dict[str, Any]:
    """Load cached benchmarks (empty dict when none exist)."""
    return _read_json(BENCHMARK_FILE, {})


def _cache_is_fresh(cache: Dict[str, Any], max_age_days: float) -> bool:
    try:
        stamp = datetime.fromisoformat(cache["meta"]["generated_utc"])
    except (KeyError, ValueError, TypeError):
        return False
    if set(cache.get("shocks", {})) != set(HISTORICAL_SHOCKS):
        return False
    age_days = (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0
    # Fully-fallback caches are retried after one day so live data is picked up later.
    limit = max_age_days if cache["meta"].get("live_assets", 0) > 0 else min(1.0, max_age_days)
    return age_days < limit


def calibrate_historical_shocks(force_refresh: bool = False, max_age_days: float = 30.0) -> Dict[str, Any]:
    """
    Fetch ^GSPC, ^TNX and CL=F across the four historical shock windows, compute
    real statistics, and cache to ``data/historical_benchmarks.json``.

    Returns a dict::

        {"meta": {...}, "shocks": {shock_key: {"label", "window",
            "assets": {ticker: {annualized_volatility, mean_daily_drift,
                                shock_delta_30d_pct, worst_30d_pct, best_30d_pct,
                                n_obs, source}}}}}
    """
    cache = load_benchmarks()
    if cache and not force_refresh and _cache_is_fresh(cache, max_age_days):
        return cache

    shocks_out: Dict[str, Any] = {}
    live_assets = 0
    total_assets = 0
    for key, spec in HISTORICAL_SHOCKS.items():
        end_exclusive = (datetime.strptime(spec["end"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        assets: Dict[str, Any] = {}
        for ticker in TICKERS.values():
            total_assets += 1
            stats = None
            series = _download_series(ticker, spec["start"], end_exclusive)
            if series is not None:
                stats = _compute_stats(series[1])
            if stats is not None:
                stats["source"] = "yfinance"
                live_assets += 1
                assets[ticker] = stats
            else:
                assets[ticker] = _fallback_stats(key, ticker)
        shocks_out[key] = {
            "label": spec["label"],
            "window": {"start": spec["start"], "end": spec["end"]},
            "assets": assets,
        }

    result = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": ("yfinance" if live_assets == total_assets
                       else "mixed" if live_assets else "fallback"),
            "live_assets": live_assets,
            "total_assets": total_assets,
            "trading_days_per_year": TRADING_DAYS,
            "shock_window_days": SHOCK_WINDOW_DAYS,
        },
        "shocks": shocks_out,
    }
    try:
        _write_json(BENCHMARK_FILE, result)
    except OSError as exc:
        logger.warning("Could not write %s: %s", BENCHMARK_FILE, exc)
    return result


def get_shock_calibration(shock_key: str) -> Dict[str, Any]:
    """Convenience accessor used by the engine/referee: stats for one shock."""
    data = calibrate_historical_shocks()
    if shock_key not in data["shocks"]:
        raise KeyError(f"Unknown shock '{shock_key}'. Options: {list(HISTORICAL_SHOCKS)}")
    return data["shocks"][shock_key]


# --------------------------------------------------------------------------- #
# 2a. Historical scenario loader
# --------------------------------------------------------------------------- #
def _nearest_anchor(target: datetime) -> Tuple[str, Tuple[float, float, float]]:
    best_key = min(
        _FALLBACK_ANCHORS,
        key=lambda k: abs((datetime.strptime(k, "%Y-%m-%d") - target).days),
    )
    return best_key, _FALLBACK_ANCHORS[best_key]


def get_historical_market_state(date_str: str, write_world_state: bool = False,
                                lookback_days: int = 10) -> Dict[str, Any]:
    """
    Real S&P 500, WTI crude oil and 10Y Treasury yield as of ``date_str``
    (YYYY-MM-DD; last trading day on or before that date).

    If ``write_world_state`` is True, the values are merged into
    ``data/world_state.json`` (market.sp500, market.oil_price,
    market.ten_year_yield, treasury_yield_10y, scenario_date) without
    touching any other key.
    """
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date_str must be formatted YYYY-MM-DD") from exc
    target = target_dt.date()
    start = (target_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    values: Dict[str, Optional[float]] = {}
    as_of: Dict[str, Optional[str]] = {}
    sources: Dict[str, str] = {}
    for name, ticker in TICKERS.items():
        series = _download_series(ticker, start, end)
        values[name], as_of[name], sources[name] = None, None, "unavailable"
        if series is not None:
            dates, closes = series
            for d, c in zip(reversed(dates), reversed(closes)):
                if d <= target:
                    values[name], as_of[name], sources[name] = float(c), d.isoformat(), "yfinance"
                    break

    if any(v is None for v in values.values()):
        anchor_key, (sp, oil, tnx) = _nearest_anchor(target_dt)
        fallback = {"sp500": sp, "oil": oil, "yield_10y": tnx}
        for name in values:
            if values[name] is None:
                values[name] = fallback[name]
                as_of[name] = anchor_key
                sources[name] = "approximate_fallback_anchor"

    state = {
        "date_requested": date_str,
        "sp500": round(float(values["sp500"]), 2),
        "oil_price": round(float(values["oil"]), 2),
        "ten_year_yield": round(float(values["yield_10y"]), 3),
        "as_of": as_of,
        "sources": sources,
    }

    if write_world_state:
        world = _read_json(WORLD_STATE_FILE, {})
        if not isinstance(world, dict):
            world = {}
        market = world.setdefault("market", {})
        market["sp500"] = state["sp500"]
        market["oil_price"] = state["oil_price"]
        market["oil_baseline"] = state["oil_price"]
        market["ten_year_yield"] = state["ten_year_yield"]
        world["treasury_yield_10y"] = state["ten_year_yield"]
        world["scenario_date"] = date_str
        world["scenario_source"] = sources
        _write_json(WORLD_STATE_FILE, world)
    return state


# --------------------------------------------------------------------------- #
# 2b. Wikipedia context fetcher
# --------------------------------------------------------------------------- #
def fetch_wikipedia_summary(title: str, timeout: float = 8.0, max_chars: int = 900) -> Optional[str]:
    """Brief plain-text summary from the free Wikipedia REST API (None on failure)."""
    if title in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[title]
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
        title.replace(" ", "_"), safe="_()%,–-")
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT,
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Wikipedia fetch failed for %s: %s", title, exc)
        return None
    extract = (payload.get("extract") or "").strip()
    if not extract:
        return None
    if len(extract) > max_chars:
        cut = extract[:max_chars].rsplit(". ", 1)[0]
        extract = (cut if len(cut) > 200 else extract[:max_chars]).rstrip() + "."
    _CONTEXT_CACHE[title] = extract
    return extract


def get_crisis_context(shock_key: str) -> str:
    """Wikipedia summary for a known shock, or a built-in fallback blurb."""
    spec = HISTORICAL_SHOCKS.get(shock_key)
    if spec is None:
        raise KeyError(f"Unknown shock '{shock_key}'. Options: {list(HISTORICAL_SHOCKS)}")
    return fetch_wikipedia_summary(spec["wiki_title"]) or _FALLBACK_CONTEXT[shock_key]


def build_referee_context(shock_key: Optional[str] = None, date_str: Optional[str] = None) -> str:
    """
    Compact text block to prepend to the LLM Referee prompt: historical crisis
    summary + calibrated market statistics + optional market snapshot.
    """
    lines: List[str] = []
    keys = [shock_key] if shock_key else list(HISTORICAL_SHOCKS)
    bench = calibrate_historical_shocks()
    for key in keys:
        lines.append(f"[{HISTORICAL_SHOCKS[key]['label']}] {get_crisis_context(key)}")
        assets = bench["shocks"][key]["assets"]
        for name, ticker in TICKERS.items():
            a = assets[ticker]
            lines.append(
                f"  {name} ({ticker}): ann. vol {a['annualized_volatility']:.0%}, "
                f"mean daily drift {a['mean_daily_drift']:+.4f}, "
                f"30d shock {a['shock_delta_30d_pct']:+.1f}% [{a['source']}]"
            )
    if date_str:
        snap = get_historical_market_state(date_str)
        lines.append(
            f"Market on {date_str}: S&P 500 {snap['sp500']:.0f}, WTI ${snap['oil_price']:.1f}, "
            f"10Y {snap['ten_year_yield']:.2f}%."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = calibrate_historical_shocks(force_refresh=True)
    print("Benchmarks source:", data["meta"]["source"])
    print(get_historical_market_state("2020-03-16"))
    print(build_referee_context("1990_gulf_oil_spike"))
