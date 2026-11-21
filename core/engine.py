"""
core/engine.py
==============
Modular 20-agent simulation engine tying together

    data/world_state.json, data/agents_config.json, data/diplomacy_log.json,
    core/historical_grounding.py  (real market baselines + crisis context)
    core/quant_engine.py          (nine quantitative / game-theoretic models)

with an LLM layer (``AsyncOpenAI`` against Gemini / Groq / Ollama / OpenAI free-
or low-cost endpoints, ``pydantic`` schemas) and a ``rich`` terminal Situation Room.

Run the terminal game from the project root::

    python -m core.engine

Importable from ``ui/app.py`` (all run from the project root)::

    from core.engine import (get_sim, dm_agent, summit_agents, consult_advisors,
                             run_blackops, run_marketwar, load_history_baseline,
                             execute_turn, export_portfolio_csv, run_sync)

Every coroutine accepts an optional ``sim=`` (a ``Simulation``); by default a shared
singleton from ``get_sim()`` is used. Use ``run_sync(coro)`` from non-async code.
With no API key (or ``SIM_OFFLINE=1``) every LLM step falls back to a deterministic
rule-based heuristic so the whole game remains playable offline.
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:            # lets `python core/engine.py` and ui/app.py resolve `core.*`
    sys.path.insert(0, str(ROOT_DIR))
DATA_DIR = ROOT_DIR / "data"
EXPORT_DIR = ROOT_DIR / "exports"
METRICS_CSV = EXPORT_DIR / "simulation_metrics.csv"

try:  # optional .env support for API keys
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core import historical_grounding as hg
from core.quant_engine import (FIVE_EYES_POOL, NON_ALIGNED, RIVAL_AXIS, US_COALITION,
                               QuantMacroEngine, _json_default)

# =========================================================================== #
# 1. LLM PROVIDER SWITCH  (change LLM_PROVIDER or set env SIM_LLM_PROVIDER)
# =========================================================================== #
LLM_PROVIDER = os.getenv("SIM_LLM_PROVIDER", "gemini").lower()   # gemini | groq | ollama | openai

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {   # Google AI Studio free tier (OpenAI-compatible endpoint)
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY", "rpm": 10,
    },
    "groq": {     # Groq free tier
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY", "rpm": 30,
    },
    "ollama": {   # local, unlimited
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": "llama3.1", "key_env": None, "api_key": "ollama", "rpm": 0,
    },
    "openai": {   # paid, cheap
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY", "rpm": 0,
    },
}
MAX_CONCURRENCY = 5          # asyncio.Semaphore(5): at most 5 in-flight LLM calls
MAX_RETRIES = 5
REQUEST_TIMEOUT = 90.0
OFFLINE = os.getenv("SIM_OFFLINE", "").lower() in ("1", "true", "yes")


class LLMError(RuntimeError):
    """LLM call failed after retries."""


class LLMUnavailable(LLMError):
    """No usable provider (missing key, offline mode, auth failure, host down)."""


def active_provider() -> Dict[str, Any]:
    if LLM_PROVIDER not in PROVIDERS:
        raise LLMUnavailable(f"Unknown provider '{LLM_PROVIDER}'. Options: {list(PROVIDERS)}")
    cfg = dict(PROVIDERS[LLM_PROVIDER])
    if os.getenv("SIM_RPM"):
        cfg["rpm"] = int(os.getenv("SIM_RPM", "0"))
    return cfg


def _api_key(cfg: Dict[str, Any]) -> Optional[str]:
    return os.getenv(cfg["key_env"]) if cfg.get("key_env") else cfg.get("api_key")


_FATAL: Optional[Tuple[str, float]] = None


def llm_ready() -> Tuple[bool, str]:
    if OFFLINE:
        return False, "SIM_OFFLINE=1"
    try:
        cfg = active_provider()
    except LLMUnavailable as exc:
        return False, str(exc)
    if not _api_key(cfg):
        return False, f"{cfg['key_env']} is not set"
    if _FATAL and time.monotonic() - _FATAL[1] < 120.0:
        return False, _FATAL[0]
    return True, "ok"


class _RateLimiter:
    """Sliding-window requests-per-minute limiter (free tiers are RPM-limited)."""

    def __init__(self, rpm: int) -> None:
        self.rpm, self.stamps, self.lock = rpm, deque(), asyncio.Lock()

    async def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.stamps and now - self.stamps[0] >= 60.0:
                    self.stamps.popleft()
                if len(self.stamps) < self.rpm:
                    self.stamps.append(now)
                    return
                wait = 60.0 - (now - self.stamps[0])
            await asyncio.sleep(max(wait, 0.05) + 0.05)


class _LoopResources:
    """Client, semaphore and limiter bound to ONE event loop (safe with repeated asyncio.run)."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        cfg = active_provider()
        self.loop = loop
        self.sem = asyncio.Semaphore(MAX_CONCURRENCY)
        self.limiter = _RateLimiter(int(cfg.get("rpm", 0)))
        self.client = AsyncOpenAI(api_key=_api_key(cfg) or "none", base_url=cfg["base_url"],
                                  timeout=REQUEST_TIMEOUT, max_retries=0)
        self.json_ok, self.extras_ok = True, True


_RES: Dict[int, _LoopResources] = {}


def _resources() -> _LoopResources:
    loop = asyncio.get_running_loop()
    for key in [k for k, r in _RES.items() if r.loop.is_closed()]:
        _RES.pop(key, None)
    res = _RES.get(id(loop))
    if res is None or res.loop is not loop:
        res = _RES[id(loop)] = _LoopResources(loop)
    return res


def set_provider(name: str) -> None:
    """Switch provider at runtime (also clears cached clients / fatal flag)."""
    global LLM_PROVIDER, _FATAL
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Options: {list(PROVIDERS)}")
    LLM_PROVIDER, _FATAL = name, None
    _RES.clear()


def _backoff(attempt: int) -> float:
    return min(30.0, 1.5 * 2 ** (attempt - 1)) + random.uniform(0.0, 0.5)


def _retry_after(exc: Exception) -> Optional[float]:
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return min(float(value), 60.0) if value else None
    except Exception:
        return None


async def llm_chat(system: str, user: str, *, json_mode: bool = True, temperature: float = 0.7,
                   max_tokens: int = 900) -> str:
    """One chat completion, behind Semaphore(5) + RPM limiter, with retry/backoff."""
    global _FATAL
    ok, reason = llm_ready()
    if not ok:
        raise LLMUnavailable(reason)
    cfg = active_provider()
    res = _resources()
    last_err = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        delay = 0.0
        async with res.sem:
            await res.limiter.acquire()
            kwargs: Dict[str, Any] = dict(
                model=cfg["model"], temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if json_mode and res.json_ok:
                kwargs["response_format"] = {"type": "json_object"}
            if res.extras_ok and cfg.get("extra_body"):
                kwargs["extra_body"] = cfg["extra_body"]
            try:
                resp = await asyncio.wait_for(res.client.chat.completions.create(**kwargs),
                                              timeout=REQUEST_TIMEOUT + 5)
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    return text
                last_err, delay = "empty response", _backoff(attempt)
            except openai.RateLimitError as exc:
                last_err, delay = f"rate limited: {exc}", _retry_after(exc) or _backoff(attempt) * 2
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                _FATAL = (f"authentication failed for {LLM_PROVIDER}: {exc}", time.monotonic())
                raise LLMUnavailable(_FATAL[0]) from exc
            except openai.NotFoundError as exc:
                _FATAL = (f"model/endpoint not found on {LLM_PROVIDER} ({cfg['model']}): {exc}", time.monotonic())
                raise LLMUnavailable(_FATAL[0]) from exc
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if kwargs.get("response_format") and ("response_format" in msg or "json" in msg):
                    res.json_ok = False
                    continue
                if kwargs.get("extra_body") and ("reasoning" in msg or "extra" in msg):
                    res.extras_ok = False
                    continue
                raise LLMError(f"bad request: {exc}") from exc
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                    asyncio.TimeoutError) as exc:
                last_err, delay = f"{type(exc).__name__}: {exc}", _backoff(attempt)
                if isinstance(exc, openai.APIConnectionError) and attempt >= 3:
                    _FATAL = (f"{LLM_PROVIDER} unreachable ({exc})", time.monotonic())
                    raise LLMUnavailable(_FATAL[0]) from exc
        if delay > 0 and attempt < MAX_RETRIES:
            await asyncio.sleep(delay)
    raise LLMError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err}")


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates fences / prose)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(t[start:end + 1])
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("JSON reply is not an object")
    return data


async def llm_structured(system: str, user: str, model_cls: type, *, temperature: float = 0.6,
                         max_tokens: int = 900) -> Any:
    last: Optional[Exception] = None
    for attempt in range(3):
        sys_msg = system if attempt == 0 else system + "\nReturn ONLY one valid JSON object. No prose, no markdown."
        text = await llm_chat(sys_msg, user, temperature=temperature, max_tokens=max_tokens)
        try:
            return model_cls.model_validate(extract_json(text))
        except (ValueError, ValidationError) as exc:
            last = exc
    raise LLMError(f"unparseable model output: {last}")


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run a coroutine from synchronous code (e.g. a Streamlit callback)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


# =========================================================================== #
# 2. PYDANTIC SCHEMAS
# =========================================================================== #
def _num(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return min(max(x, lo), hi)


def _pick(v: Any, options: Tuple[str, ...], default: str) -> str:
    s = str(v or "").strip().lower().replace(" ", "_")
    return s if s in options else default


def _float_map(v: Any, lo: float, hi: float) -> Dict[str, float]:
    if not isinstance(v, dict):
        return {}
    return {str(k): _num(x, lo, hi) for k, x in v.items()}


ACTION_TYPES = ("diplomacy", "economic", "military", "cyber", "espionage", "covert", "stand_down")
OP_TYPES = ("HUMINT", "SIGINT", "CYBER")


class EspionageOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target: str
    op_type: str = "HUMINT"

    @field_validator("op_type", mode="before")
    @classmethod
    def _op(cls, v: Any) -> str:
        s = str(v or "HUMINT").upper()
        return s if s in OP_TYPES else "HUMINT"


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent: str = ""
    action_type: str = "diplomacy"
    target: Optional[str] = None
    public_statement: str = ""
    secret_intent: str = ""
    stance_toward_us: float = 0.0
    aggression: float = 5.0
    cooperation: float = 5.0
    military_mobilization: float = 3.0
    espionage: Optional[EspionageOrder] = None
    source: str = "llm"

    @field_validator("action_type", mode="before")
    @classmethod
    def _a(cls, v: Any) -> str:
        return _pick(v, ACTION_TYPES, "diplomacy")

    @field_validator("stance_toward_us", mode="before")
    @classmethod
    def _s(cls, v: Any) -> float:
        return _num(v, -5, 5, 0.0)

    @field_validator("aggression", "cooperation", "military_mobilization", mode="before")
    @classmethod
    def _r(cls, v: Any) -> float:
        return _num(v, 0, 10, 5.0)

    @field_validator("public_statement", "secret_intent", mode="before")
    @classmethod
    def _t(cls, v: Any) -> str:
        return str(v or "")[:400]

    @field_validator("espionage", mode="before")
    @classmethod
    def _e(cls, v: Any) -> Any:
        return v if isinstance(v, dict) and v.get("target") else None


class AgentReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reply: str = ""
    stance: str = "Stall"
    trust_delta: float = 0.0
    compliance: float = 0.3
    secret_intent: str = ""
    source: str = "llm"

    @field_validator("stance", mode="before")
    @classmethod
    def _st(cls, v: Any) -> str:
        s = str(v or "Stall").strip().capitalize()
        return s if s in ("Accept", "Counter", "Stall", "Reject") else "Stall"

    @field_validator("trust_delta", mode="before")
    @classmethod
    def _td(cls, v: Any) -> float:
        return _num(v, -10, 10, 0.0)

    @field_validator("compliance", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return _num(v, 0, 1, 0.3)


class AdvisorOpinion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    advisor: str = ""
    verdict: str = "MODIFY"
    risk_score: float = 5.0
    rationale: str = ""
    warnings: List[str] = Field(default_factory=list)
    suggested_modification: str = ""
    source: str = "llm"

    @field_validator("verdict", mode="before")
    @classmethod
    def _v(cls, v: Any) -> str:
        s = str(v or "MODIFY").strip().upper()
        return s if s in ("PROCEED", "MODIFY", "ABORT") else "MODIFY"

    @field_validator("risk_score", mode="before")
    @classmethod
    def _rs(cls, v: Any) -> float:
        return _num(v, 0, 10, 5.0)

    @field_validator("warnings", mode="before")
    @classmethod
    def _w(cls, v: Any) -> List[str]:
        return [str(x) for x in v][:6] if isinstance(v, list) else []


class RefereeVerdict(BaseModel):
    """Referee output: narrative + shock parameters fed to QuantMacroEngine.run_turn()."""
    model_config = ConfigDict(extra="ignore")
    narrative: str = ""
    headlines: List[str] = Field(default_factory=list)
    defcon_level: Optional[int] = None
    us_strategic_score_delta: float = 0.0
    cables_cut: List[str] = Field(default_factory=list)
    cables_restored: List[str] = Field(default_factory=list)
    oil_supply_shock_pct: float = 0.0
    equity_jump_mean: float = 0.0
    corridor_flow_changes: Dict[str, float] = Field(default_factory=dict)
    spr_drawdown_pct: float = 0.0
    munitions_expended_pct: float = 0.0
    rare_earth_consumption_pct: float = 0.0
    escalation_events: List[float] = Field(default_factory=list)
    gps_satellite_jamming: float = 0.0
    hostile_short_interest_delta_billions: float = 0.0
    war_damage_pct: Dict[str, float] = Field(default_factory=dict)
    sanctions_severity: Dict[str, float] = Field(default_factory=dict)
    humiliated_leaders: Dict[str, float] = Field(default_factory=dict)
    us_attacked_severity: float = 0.0
    us_casualties: float = 0.0
    war_intensity: float = 0.0
    controversial_order_severity: float = 0.0
    war_spending_pct_gdp: float = 0.0
    trust_changes: Dict[str, float] = Field(default_factory=dict)
    source: str = "llm"

    @field_validator("defcon_level", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Optional[int]:
        return None if v in (None, "", "null") else int(_num(v, 1, 5, 5))

    @field_validator("headlines", "cables_cut", "cables_restored", mode="before")
    @classmethod
    def _sl(cls, v: Any) -> List[str]:
        return [str(x) for x in v][:8] if isinstance(v, list) else []

    @field_validator("escalation_events", mode="before")
    @classmethod
    def _ev(cls, v: Any) -> List[float]:
        return [_num(x, 0, 3) for x in v][:8] if isinstance(v, list) else []

    @field_validator("us_strategic_score_delta", mode="before")
    @classmethod
    def _sd(cls, v: Any) -> float:
        return _num(v, -15, 15)

    @field_validator("oil_supply_shock_pct", mode="before")
    @classmethod
    def _o(cls, v: Any) -> float:
        return _num(v, -40, 80)

    @field_validator("equity_jump_mean", mode="before")
    @classmethod
    def _eq(cls, v: Any) -> float:
        return _num(v, -0.35, 0.10)

    @field_validator("spr_drawdown_pct", "munitions_expended_pct", "rare_earth_consumption_pct", mode="before")
    @classmethod
    def _pct(cls, v: Any) -> float:
        return _num(v, 0, 40)

    @field_validator("gps_satellite_jamming", "war_intensity", "controversial_order_severity", mode="before")
    @classmethod
    def _u(cls, v: Any) -> float:
        return _num(v, 0, 1)

    @field_validator("hostile_short_interest_delta_billions", mode="before")
    @classmethod
    def _si(cls, v: Any) -> float:
        return _num(v, -500, 500)

    @field_validator("us_attacked_severity", mode="before")
    @classmethod
    def _ua(cls, v: Any) -> float:
        return _num(v, 0, 10)

    @field_validator("us_casualties", mode="before")
    @classmethod
    def _cas(cls, v: Any) -> float:
        return _num(v, 0, 100)

    @field_validator("war_spending_pct_gdp", mode="before")
    @classmethod
    def _ws(cls, v: Any) -> float:
        return _num(v, 0, 8)

    @field_validator("corridor_flow_changes", mode="before")
    @classmethod
    def _cf(cls, v: Any) -> Dict[str, float]:
        return _float_map(v, -1, 1)

    @field_validator("war_damage_pct", mode="before")
    @classmethod
    def _wd(cls, v: Any) -> Dict[str, float]:
        return _float_map(v, 0, 40)

    @field_validator("sanctions_severity", mode="before")
    @classmethod
    def _ss(cls, v: Any) -> Dict[str, float]:
        return _float_map(v, 0, 1)

    @field_validator("humiliated_leaders", mode="before")
    @classmethod
    def _hl(cls, v: Any) -> Dict[str, float]:
        return _float_map(v, 0, 10)

    @field_validator("trust_changes", mode="before")
    @classmethod
    def _tc(cls, v: Any) -> Dict[str, float]:
        return _float_map(v, -10, 10)

    def to_shocks(self) -> Dict[str, Any]:
        """Parameters understood by QuantMacroEngine.run_turn()."""
        shocks: Dict[str, Any] = {
            "oil_supply_shock_pct": self.oil_supply_shock_pct, "equity_jump_mean": self.equity_jump_mean,
            "corridor_flow_changes": dict(self.corridor_flow_changes),
            "spr_drawdown_pct": self.spr_drawdown_pct, "munitions_expended_pct": self.munitions_expended_pct,
            "rare_earth_consumption_pct": self.rare_earth_consumption_pct,
            "escalation_events": list(self.escalation_events),
            "gps_satellite_jamming": self.gps_satellite_jamming,
            "hostile_short_interest_delta_billions": self.hostile_short_interest_delta_billions,
            "war_damage_pct": dict(self.war_damage_pct), "sanctions_severity": dict(self.sanctions_severity),
            "humiliated_leaders": dict(self.humiliated_leaders),
            "us_attacked_severity": self.us_attacked_severity, "us_casualties": self.us_casualties,
            "war_intensity": self.war_intensity,
            "controversial_order_severity": self.controversial_order_severity,
            "war_spending_pct_gdp": self.war_spending_pct_gdp,
        }
        return shocks


class TurnReport(BaseModel):
    turn: int
    move: str
    defcon: int
    us_strategic_score: float
    narrative: str = ""
    headlines: List[str] = Field(default_factory=list)
    red_phone: Optional[Dict[str, Any]] = None
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    espionage: List[Dict[str, Any]] = Field(default_factory=list)
    verdict: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    llm_fallbacks: int = 0


def _plain(obj: Any) -> Any:
    """Round-trip through JSON so numpy scalars/arrays become plain Python."""
    return json.loads(json.dumps(obj, default=_json_default))


# =========================================================================== #
# 3. STATIC WORLD DATA
# =========================================================================== #
PERSONAS: Dict[str, Tuple[str, str]] = {
    "United Kingdom": ("the Prime Minister", "Five Eyes anchor; Atlanticist; protects City of London finance"),
    "Israel": ("the Prime Minister", "existential security doctrine; preemption against Iran; depends on US munitions"),
    "Australia": ("the Prime Minister", "AUKUS partner; export-dependent on China; Pacific security"),
    "Japan": ("the Prime Minister", "US treaty ally; energy-import dependent; Taiwan Strait anxiety"),
    "Germany": ("the Chancellor", "export economy; energy-dependent; cautious on military escalation"),
    "France": ("the President", "strategic autonomy; nuclear power; mediator instincts"),
    "South Korea": ("the President", "deters North Korea; chip supply chains; US alliance"),
    "Taiwan": ("the President", "semiconductor shield; survival depends on US deterrence"),
    "China": ("the General Secretary", "rival superpower; rare-earth leverage; seeks Taiwan reunification and a dollar-system hedge"),
    "Russia": ("the President", "energy and nuclear coercion; shadow fleet; resents NATO"),
    "Iran": ("the Supreme Leader's government", "proxy network; nuclear threshold; sells oil to China via shadow fleet"),
    "North Korea": ("the Supreme Leader", "nuclear blackmail for regime survival; Russian/Chinese patronage"),
    "Belarus": ("the President", "Moscow-dependent; regime survival above all"),
    "India": ("the Prime Minister", "strategic autonomy; buys Russian oil; balances Quad and BRICS"),
    "Saudi Arabia": ("the Crown Prince", "oil swing producer; hedges between Washington and Beijing"),
    "Turkey": ("the President", "transactional NATO member; Bosporus leverage; multi-alignment"),
    "Brazil": ("the President", "BRICS non-aligned; commodity exporter; skeptical of sanctions"),
    "UAE": ("the President", "financial and logistics hub; hedger; energy exporter"),
    "Indonesia": ("the President", "non-aligned archipelago; nickel; Malacca chokepoint"),
    "Mexico": ("the President", "USMCA integration; remittances; sovereignty-sensitive"),
}
ALIASES = {"uk": "United Kingdom", "britain": "United Kingdom", "korea": "South Korea",
           "rok": "South Korea", "dprk": "North Korea", "saudi": "Saudi Arabia", "ksa": "Saudi Arabia",
           "prc": "China", "emirates": "UAE"}

CABLES: Dict[str, List[str]] = {
    "Transatlantic_TAT": ["United Kingdom", "Germany", "France", "Brazil", "Mexico"],
    "Transpacific_FASTER": ["Japan", "South Korea", "Taiwan", "China", "Australia", "Indonesia", "North Korea"],
    "EurAsia_SEAMEWE": ["Russia", "Iran", "India", "Saudi Arabia", "UAE", "Turkey", "Israel", "Belarus"],
}
BLOC_ALIASES = {
    "us_coalition": US_COALITION, "coalition": US_COALITION, "allies": US_COALITION,
    "rival_axis": RIVAL_AXIS, "axis": RIVAL_AXIS, "rivals": RIVAL_AXIS, "adversaries": RIVAL_AXIS,
    "non_aligned": NON_ALIGNED, "swing": NON_ALIGNED, "nonaligned": NON_ALIGNED,
}
HOSTILE_WORDS = ("sanction", "strike", "attack", "blockade", "invade", "tariff", "embargo", "bomb", "seize",
                 "cut ", "sabotage", "nuclear", "deploy", "ultimatum", "freeze", "expel", "mobiliz")
CONCILIATORY_WORDS = ("summit", "negotiat", "treaty", "aid", "ceasefire", "withdraw", "deal", "talks",
                      "lift", "de-escalat", "peace", "cooperat")
BLACKOPS_TYPES = ("stuxnet_sabotage", "fund_coup_rebels", "false_flag_cable", "cut_undersea_cable",
                  "hire_pmc_mercenaries")
MARKETWAR_ACTIONS = ("release_spr_oil", "esf_short_squeeze", "secondary_sanctions_sweep")
METRIC_COLUMNS = [
    "turn", "scenario_date", "defcon", "hawkes_intensity", "us_strategic_score", "sp500", "sp500_change_pct",
    "sp500_var_95", "sp500_cvar_95", "oil_price", "oil_futures_curve", "fed_funds_rate", "treasury_yield_2y",
    "treasury_yield_10y", "yield_curve_inverted", "inflation_pct", "supply_chain_stress_index",
    "spr_oil_reserves_pct", "rare_earth_defense_stockpile_pct", "precision_munitions_magazine_pct",
    "us_military_tech_edge", "hostile_short_interest_billions", "esf_capacity_billions", "rally_effect",
    "war_exhaustion_index", "wall_street_oligarch_loyalty", "pentagon_joint_chiefs_obedience",
    "deep_state_25th_amendment_risk", "avg_cds_bps", "max_cds_bps", "max_pd_pct", "avg_allied_trust",
    "cables_cut", "kompromat_unused", "nash_us_top_strategy", "timestamp_utc"]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def defcon_from_hawkes(lam: float) -> int:
    return 5 if lam < 1.0 else 4 if lam < 2.5 else 3 if lam < 5.0 else 2 if lam < 9.0 else 1


def merge_shocks(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Additively merge two shock dicts (numbers sum, dicts merge by key, lists concatenate)."""
    out: Dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                           for k, v in base.items()}
    for k, v in (extra or {}).items():
        cur = out.get(k)
        if isinstance(v, dict):
            tgt = cur if isinstance(cur, dict) else {}
            for kk, vv in v.items():
                tgt[kk] = tgt.get(kk, 0.0) + vv
            out[k] = tgt
        elif isinstance(v, list):
            out[k] = (cur if isinstance(cur, list) else []) + list(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = (cur if isinstance(cur, (int, float)) else 0.0) + v
        else:
            out[k] = v
    return out


# =========================================================================== #
# 4. SIMULATION STATE
# =========================================================================== #
class Simulation:
    """Owns the QuantMacroEngine (world_state + agents_config) and the game bookkeeping."""

    def __init__(self, seed: Optional[int] = None) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.quant = QuantMacroEngine(data_dir=DATA_DIR, export_dir=EXPORT_DIR, seed=seed)
        self.rng = random.Random(seed)
        self.pending_shocks: Dict[str, Any] = {}
        self.progress: Callable[[str], None] = lambda msg: None
        self.llm_fallbacks = 0
        self.last_llm_error = ""
        self.last_report: Optional[TurnReport] = None
        self._init_state()
        files = ("world_state.json", "agents_config.json", "diplomacy_log.json")
        if any(not (DATA_DIR / f).exists() for f in files):
            self.save_all()

    # ---- accessors -------------------------------------------------------
    @property
    def ws(self) -> Dict[str, Any]:
        return self.quant.world_state

    @property
    def agents(self) -> Dict[str, Dict[str, Any]]:
        return self.quant.agents

    def _init_state(self) -> None:
        ws = self.ws
        ws.setdefault("defcon", 5)
        ws.setdefault("undersea_cables", {})
        for cable in CABLES:
            ws["undersea_cables"].setdefault(cable, "Intact")
        ws.setdefault("wiretap_feed", [])
        ws.setdefault("metrics_history", [])
        ws.setdefault("pending_notes", [])
        ws.setdefault("scenario_regime", None)
        ws.setdefault("scenario_date", "")
        ws.setdefault("historical_context", "")
        ws.setdefault("us_score_bonus", 0.0)
        ws["market"].setdefault("sp500_baseline", ws["market"]["sp500"])
        for name, a in self.agents.items():
            leader, doctrine = PERSONAS.get(name, (f"the leader of {name}", "pursues national interest"))
            a.setdefault("leader", leader)
            a.setdefault("doctrine", doctrine)
            a.setdefault("regime_stability", 60.0 if a.get("coalition") == RIVAL_AXIS else 75.0)
        self._sync_cables()
        ws["us_strategic_score"] = round(self.compute_us_strategic_score(), 2)
        if not ws["metrics_history"]:
            self.record_metrics()

    # ---- name resolution -------------------------------------------------
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[\s_]+", " ", s.strip().lower())

    def resolve_agent(self, text: str) -> Optional[str]:
        n = self._norm(text)
        for name in self.agents:
            if self._norm(name) == n:
                return name
        return ALIASES.get(n) if ALIASES.get(n) in self.agents else None

    def split_agent_prefix(self, text: str) -> Tuple[Optional[str], str]:
        """Match the longest agent name/alias at the start of ``text`` (spaces or underscores)."""
        cands = [(n, n) for n in self.agents] + [(k, v) for k, v in ALIASES.items() if v in self.agents]
        for label, agent in sorted(cands, key=lambda c: -len(c[0])):
            pattern = r"^\s*" + r"[\s_]+".join(re.escape(p) for p in label.split()) + r"(?=\s|$)"
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if m:
                return agent, text[m.end():].strip()
        return None, text

    def resolve_targets(self, spec: str) -> Tuple[List[str], List[str]]:
        """Bloc names and/or comma list -> (agents, unknown tokens)."""
        found: List[str] = []
        unknown: List[str] = []
        for token in [t for t in re.split(r"[,;]", spec) if t.strip()]:
            n = self._norm(token).replace(" ", "_")
            if n in ("all", "everyone", "g20"):
                found += list(self.agents)
            elif n == "five_eyes":
                found += [a for a in FIVE_EYES_POOL if a in self.agents]
            elif n in BLOC_ALIASES:
                found += [k for k, a in self.agents.items() if a.get("coalition") == BLOC_ALIASES[n]]
            else:
                agent = self.resolve_agent(token)
                (found if agent else unknown).append(agent or token.strip())
        seen: set = set()
        return [a for a in found if not (a in seen or seen.add(a))], unknown

    # ---- cables ----------------------------------------------------------
    def cable_of(self, agent: str) -> str:
        explicit = self.agents.get(agent, {}).get("cable")
        if explicit in CABLES:
            return explicit
        for cable, members in CABLES.items():
            if agent in members:
                return cable
        return "Transatlantic_TAT"

    def cable_status(self) -> Dict[str, str]:
        return dict(self.ws["undersea_cables"])

    def resolve_cable(self, text: str) -> Optional[str]:
        t = self._norm(text)
        for cable in CABLES:
            if t == self._norm(cable) or t in self._norm(cable) and len(t) >= 4:
                return cable
        return None

    def set_cable(self, cable: str, status: str) -> None:
        self.ws["undersea_cables"][cable] = status
        self._sync_cables()

    def _sync_cables(self) -> None:
        cut = sum(1 for s in self.ws["undersea_cables"].values() if s == "Cut")
        self.ws["undersea_cables_status"] = "Intact" if cut == 0 else "Degraded" if cut == 1 else "Cut"

    # ---- wiretap / log ---------------------------------------------------
    def add_wiretap(self, source: str, text: str, level: str = "SIGINT") -> None:
        feed = self.ws["wiretap_feed"]
        feed.append({"turn": self.ws.get("turn", 0), "source": source, "level": level, "text": text})
        del feed[:-60]

    def log_diplomacy(self, entry: Dict[str, Any]) -> None:
        entry = {"turn": self.ws.get("turn", 0), "ts": datetime.now(timezone.utc).isoformat(), **entry}
        self.quant._append_diplomacy_log(_plain(entry))

    def recent_contacts(self, agent: str, n: int = 3) -> List[str]:
        log = hg._read_json(DATA_DIR / "diplomacy_log.json", [])
        out: List[str] = []
        for e in reversed(log if isinstance(log, list) else []):
            if agent in (e.get("agent"), e.get("target"), e.get("attacker")) or agent in e.get("agents", []):
                out.append(f"[T{e.get('turn', '?')}] {e.get('type', 'contact')}: "
                           f"{str(e.get('us_message') or e.get('reply') or e.get('outcome') or '')[:140]}")
            if len(out) >= n:
                break
        return out

    def _trim_log(self) -> None:
        path = DATA_DIR / "diplomacy_log.json"
        log = hg._read_json(path, None)
        if not isinstance(log, list):
            log = []
        hg._write_json(path, log[-1000:])

    def save_all(self) -> None:
        """Persist world_state.json, agents_config.json and diplomacy_log.json under data/."""
        self._trim_log()
        self.quant.save()

    # ---- kompromat -------------------------------------------------------
    def find_kompromat(self, agent: str) -> Optional[Dict[str, Any]]:
        for d in self.ws["kompromat_dossiers"]:
            if d.get("target") == agent and not d.get("used"):
                return d
        return None

    # ---- scoring ---------------------------------------------------------
    def compute_us_strategic_score(self) -> float:
        ws, m, dom = self.ws, self.ws["market"], self.ws["us_domestic_politics"]
        allies = [a["trust"] for a in self.agents.values() if a.get("coalition") == US_COALITION]
        alliance = sum(allies) / len(allies) if allies else 50.0
        drawdown = max(0.0, (1.0 - m["sp500"] / max(m.get("sp500_baseline", m["sp500"]), 1e-9)) * 100.0)
        econ = _clip(100 - 0.5 * ws["supply_chain_stress_index"] - 3 * max(0.0, ws["inflation_pct"] - 3.0)
                     - 1.5 * drawdown, 0, 100)
        military = (0.5 * ws["us_military_tech_edge"] + 0.3 * ws["precision_munitions_magazine_pct"]
                    + 0.2 * ws["rare_earth_defense_stockpile_pct"])
        stability = _clip(100 - 0.8 * dom["deep_state_25th_amendment_risk"] - 0.4 * dom["war_exhaustion_index"]
                          - 0.2 * (100 - dom["wall_street_oligarch_loyalty"]), 0, 100)
        threat = _clip(100 - 9.0 * ws["hawkes_escalation_intensity"], 0, 100)
        score = (0.20 * econ + 0.25 * alliance + 0.25 * military + 0.15 * stability + 0.15 * threat
                 + ws.get("us_score_bonus", 0.0))
        return _clip(score, 0, 100)

    def metrics_row(self) -> Dict[str, Any]:
        ws, m, dom = self.ws, self.ws["market"], self.ws["us_domestic_politics"]
        cds = [a["cds_spread_bps"] for a in self.agents.values()]
        pds = [a["pd"] for a in self.agents.values()]
        allies = [a["trust"] for a in self.agents.values() if a.get("coalition") == US_COALITION]
        us_strat = ws.get("superpower_nash", {}).get("us_coalition_strategy", {})
        row = {
            "turn": ws.get("turn", 0), "scenario_date": ws.get("scenario_date", ""), "defcon": ws["defcon"],
            "hawkes_intensity": ws["hawkes_escalation_intensity"], "us_strategic_score": ws["us_strategic_score"],
            "sp500": m["sp500"], "sp500_change_pct": m.get("sp500_change_pct", 0.0),
            "sp500_var_95": m.get("sp500_var_95", 0.0), "sp500_cvar_95": m.get("sp500_cvar_95", 0.0),
            "oil_price": m["oil_price"], "oil_futures_curve": ws["oil_futures_curve"],
            "fed_funds_rate": ws["fed_funds_rate"], "treasury_yield_2y": ws["treasury_yield_2y"],
            "treasury_yield_10y": ws["treasury_yield_10y"], "yield_curve_inverted": ws["yield_curve_inverted"],
            "inflation_pct": ws["inflation_pct"], "supply_chain_stress_index": ws["supply_chain_stress_index"],
            "spr_oil_reserves_pct": ws["spr_oil_reserves_pct"],
            "rare_earth_defense_stockpile_pct": ws["rare_earth_defense_stockpile_pct"],
            "precision_munitions_magazine_pct": ws["precision_munitions_magazine_pct"],
            "us_military_tech_edge": ws["us_military_tech_edge"],
            "hostile_short_interest_billions": ws["hostile_short_interest_billions"],
            "esf_capacity_billions": ws["esf_capacity_billions"],
            "rally_effect": dom["rally_effect"], "war_exhaustion_index": dom["war_exhaustion_index"],
            "wall_street_oligarch_loyalty": dom["wall_street_oligarch_loyalty"],
            "pentagon_joint_chiefs_obedience": dom["pentagon_joint_chiefs_obedience"],
            "deep_state_25th_amendment_risk": dom["deep_state_25th_amendment_risk"],
            "avg_cds_bps": sum(cds) / len(cds) if cds else 0.0, "max_cds_bps": max(cds) if cds else 0.0,
            "max_pd_pct": 100.0 * max(pds) if pds else 0.0,
            "avg_allied_trust": sum(allies) / len(allies) if allies else 0.0,
            "cables_cut": sum(1 for s in ws["undersea_cables"].values() if s == "Cut"),
            "kompromat_unused": sum(1 for d in ws["kompromat_dossiers"] if not d.get("used")),
            "nash_us_top_strategy": max(us_strat, key=us_strat.get) if us_strat else "",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}

    def record_metrics(self) -> Dict[str, Any]:
        row = _plain(self.metrics_row())
        hist = self.ws["metrics_history"]
        if hist and hist[-1].get("turn") == row["turn"]:
            hist[-1] = row
        else:
            hist.append(row)
        del hist[:-2000]
        return row

    # ---- prompts ---------------------------------------------------------
    def world_brief(self) -> str:
        ws, m = self.ws, self.ws["market"]
        cut = [c for c, s in ws["undersea_cables"].items() if s == "Cut"]
        weak = [k for k, v in ws["trade_corridors_status"].items() if v < 0.6]
        return (f"Turn {ws.get('turn', 0)}. DEFCON {ws['defcon']}. Escalation intensity {ws['hawkes_escalation_intensity']:.2f}. "
                f"S&P 500 {m['sp500']:.0f} ({m.get('sp500_change_pct', 0):+.1f}% last turn). WTI ${m['oil_price']:.1f} "
                f"({ws['oil_futures_curve']}). Fed funds {ws['fed_funds_rate']:.2f}%, 2Y {ws['treasury_yield_2y']:.2f}%, "
                f"10Y {ws['treasury_yield_10y']:.2f}%{' (inverted)' if ws['yield_curve_inverted'] else ''}. "
                f"Inflation {ws['inflation_pct']:.1f}%. Supply-chain stress {ws['supply_chain_stress_index']:.0f}/100. "
                f"US SPR {ws['spr_oil_reserves_pct']:.0f}%. Cut undersea cables: {', '.join(cut) or 'none'}. "
                f"Disrupted trade corridors: {', '.join(weak) or 'none'}."
                + (f" Historical context: {ws['historical_context'][:300]}" if ws.get("historical_context") else ""))

    def agent_card(self, name: str) -> str:
        a = self.agents[name]
        cut = self.ws["undersea_cables"].get(self.cable_of(name)) == "Cut"
        return (f"trust_in_US={a['trust']:.0f}/100, risk_tolerance={a['risk_tolerance']:.2f}, "
                f"humiliated(loss-domain)={a['in_loss_domain']}, sovereign_PD={a['pd']:.1%}, CDS={a['cds_spread_bps']:.0f}bps, "
                f"GDP=${a['gdp_bn']:.0f}bn, military={a['mil']:.0f}/100, regime_stability={a.get('regime_stability', 60):.0f}/100, "
                f"comms_cable={'CUT (isolated)' if cut else 'ok'}")

    def agent_system(self, name: str) -> str:
        a = self.agents[name]
        return (f"You are {a['leader']} of {name} in a geopolitical simulation. Bloc: {a['coalition']}. "
                f"Doctrine: {a['doctrine']}. Act realistically and in your own national interest; be concise. "
                f"Respond ONLY with one JSON object.")

    # ---- SIGINT ----------------------------------------------------------
    def sigint_intercept(self, agent: str) -> Tuple[bool, float]:
        us, a = self.ws["us_profile"], self.agents[agent]
        bonus = 0.3 * us.get("intel_on", {}).get(agent, 0.0)
        p = 1.0 / (1.0 + math.exp(-_clip(0.08 * (us["attack"] + bonus - a["defense"] - 15.0), -60, 60)))
        if self.ws["undersea_cables"].get(self.cable_of(agent)) == "Cut":
            p *= 0.5
        return self.rng.random() < p, p

    def note_fallback(self, exc: Exception) -> None:
        self.llm_fallbacks += 1
        self.last_llm_error = str(exc)


# ---- singleton ------------------------------------------------------------
_SIM: Optional[Simulation] = None


def get_sim() -> Simulation:
    global _SIM
    if _SIM is None:
        _SIM = Simulation()
    return _SIM


def reset_sim(seed: Optional[int] = None) -> Simulation:
    global _SIM
    _SIM = Simulation(seed=seed)
    return _SIM


# =========================================================================== #
# 5. OFFLINE HEURISTICS (used when the LLM is unavailable or a call fails)
# =========================================================================== #
def move_tone(text: str) -> float:
    t = text.lower()
    return _clip(sum(w in t for w in HOSTILE_WORDS) - sum(w in t for w in CONCILIATORY_WORDS), -3, 3)


def heuristic_decision(sim: Simulation, name: str, move: str) -> AgentDecision:
    a = sim.agents[name]
    rival = a["coalition"] == RIVAL_AXIS
    ally = a["coalition"] == US_COALITION
    tone = move_tone(move)
    mentioned = name.lower() in move.lower()
    r = sim.rng
    aggression = _clip(1.5 + 6.0 * a["risk_tolerance"] * (1.0 if rival else 0.4)
                       + max(0.0, tone) * (1.3 if (rival or mentioned) else 0.3)
                       + (2.0 if a["in_loss_domain"] else 0.0) + r.uniform(-0.7, 0.7), 0, 10)
    cooperation = _clip(5 + (3 if ally else -3 if rival else 0) - max(0.0, tone) * (1.0 if mentioned else 0.2)
                        + max(0.0, -tone) + r.uniform(-0.7, 0.7), 0, 10)
    stance = _clip((a["trust"] - 50) / 12 - max(0.0, tone) * (1.0 if mentioned else 0.3) + 0.5 * max(0.0, -tone), -5, 5)
    action = ("military" if rival and aggression > 7 else "cyber" if rival and aggression > 5
              else "economic" if aggression > 4 else "diplomacy")
    esp = None
    if rival or aggression > 6:
        pool = ["US"] + [n for n, x in sim.agents.items() if x["coalition"] == US_COALITION and n != name]
        esp = EspionageOrder(target=r.choice(pool if rival else pool[:1]), op_type=r.choice(OP_TYPES))
    return AgentDecision(
        agent=name, action_type=action, target="US" if rival else None,
        public_statement=("We reject this provocation." if stance < -1 else
                          "We welcome dialogue." if stance > 1 else "We are studying the announcement."),
        secret_intent=("Probe US resolve while hardening posture." if rival else
                       "Hedge between blocs and protect economic interests." if not ally else
                       "Stay aligned but seek concessions."),
        stance_toward_us=stance, aggression=aggression, cooperation=cooperation,
        military_mobilization=_clip(aggression * 0.8, 0, 10), espionage=esp, source="heuristic")


def heuristic_reply(sim: Simulation, name: str, message: str, blackmail: bool) -> AgentReply:
    a = sim.agents[name]
    tone = move_tone(message)
    comp = _clip((a["trust"] - 30) / 80 - 0.15 * max(0.0, tone) + (0.4 if blackmail else 0.0), 0, 1)
    stance = "Accept" if comp > 0.65 else "Counter" if comp > 0.45 else "Stall" if comp > 0.25 else "Reject"
    return AgentReply(
        reply={"Accept": "Agreed in principle; we will coordinate.", "Counter": "We can proceed, with conditions.",
               "Stall": "We need time to consult.", "Reject": "This is unacceptable to us."}[stance],
        stance=stance, trust_delta=(-6 if blackmail else 2 if tone <= 0 else -2), compliance=comp,
        secret_intent="Weigh concessions against domestic politics.", source="heuristic")


def heuristic_advisor(sim: Simulation, advisor: str, draft: str) -> AdvisorOpinion:
    ws, tone = sim.ws, move_tone(draft)
    base = 3.0 + 1.2 * max(0.0, tone) + 0.3 * ws["hawkes_escalation_intensity"]
    if advisor == "Fed_Chair":
        risk = base + 0.04 * ws["supply_chain_stress_index"] + (1.5 if ws["yield_curve_inverted"] else 0.0)
        warn = [f"Inflation {ws['inflation_pct']:.1f}%, stress {ws['supply_chain_stress_index']:.0f}/100."]
    elif advisor == "Joint_Chiefs":
        risk = base + 0.04 * (100 - ws["precision_munitions_magazine_pct"]) + (1.0 if ws["defcon"] <= 3 else 0.0)
        warn = [f"Munitions magazine {ws['precision_munitions_magazine_pct']:.0f}%, DEFCON {ws['defcon']}."]
    else:
        risk = base + 0.5 * sum(1 for s in ws["undersea_cables"].values() if s == "Cut")
        warn = [f"Unused Kompromat dossiers: {sum(1 for d in ws['kompromat_dossiers'] if not d['used'])}."]
    risk = _clip(risk, 0, 10)
    return AdvisorOpinion(advisor=advisor, verdict="ABORT" if risk > 8 else "MODIFY" if risk > 5 else "PROCEED",
                          risk_score=risk, rationale="Rule-based assessment (LLM unavailable).", warnings=warn,
                          suggested_modification="Phase the move and keep a diplomatic off-ramp.", source="heuristic")


def heuristic_verdict(sim: Simulation, move: str, decisions: List[AgentDecision],
                      esp_results: List[Dict[str, Any]]) -> RefereeVerdict:
    tone = move_tone(move)
    rivals = [d for d in decisions if sim.agents[d.agent]["coalition"] == RIVAL_AXIS]
    agg = sum(d.aggression for d in rivals) / len(rivals) if rivals else 3.0
    hum = {a: _clip(1.5 * tone, 0, 6) for a in sim.agents if a.lower() in move.lower() and tone > 0}
    corridors = {k: -0.15 * tone for k, v in sim.ws["trade_corridors_status"].items()
                 if tone > 1 and k.split("->")[0] in sim.agents
                 and sim.agents[k.split("->")[0]]["coalition"] == RIVAL_AXIS}
    return RefereeVerdict(
        narrative=(f"Rule-based adjudication: the move's tone ({tone:+.0f}) met an average rival aggression of "
                   f"{agg:.1f}/10; markets and alliances adjusted accordingly."),
        headlines=[f"Washington announces: {move[:80]}", f"Rival capitals respond (avg aggression {agg:.1f}/10)"],
        oil_supply_shock_pct=2.0 * max(0.0, tone), equity_jump_mean=-0.01 * agg * (1 if tone > 0 else 0.3),
        corridor_flow_changes=corridors, escalation_events=[_clip(agg / 10.0 + 0.2 * max(0.0, tone), 0, 3)],
        humiliated_leaders=hum, war_intensity=_clip(max(0.0, tone) / 3.0, 0, 1),
        us_strategic_score_delta=_clip(-0.5 * (agg - 5) + 0.5 * max(0.0, -tone), -5, 5), source="heuristic")


# =========================================================================== #
# 6. PUBLIC API: DM / SUMMIT / CONSULT
# =========================================================================== #
async def _agent_reply(sim: Simulation, agent: str, message: str, *, mode: str, attendees: Optional[List[str]] = None,
                       blackmail_dossier: Optional[Dict[str, Any]] = None) -> AgentReply:
    leverage = ""
    if blackmail_dossier:
        leverage = (f"\nThe US President is PRIVATELY BLACKMAILING you with this dossier: "
                    f"{blackmail_dossier['description']} (leverage {blackmail_dossier['leverage_value']}).")
    ctx = (f"WORLD BRIEF: {sim.world_brief()}\nYOUR STATE: {sim.agent_card(agent)}\n"
           f"RECENT CONTACTS: {'; '.join(sim.recent_contacts(agent)) or 'none'}\n")
    if mode == "summit":
        ctx += f"MULTILATERAL SUMMIT attendees: {', '.join(attendees or [])}.\n"
    user = (ctx + f"MESSAGE FROM THE US PRESIDENT: \"{message}\"{leverage}\n"
            'Return JSON: {"reply": "<=60 words in character", "stance": "Accept|Counter|Stall|Reject", '
            '"trust_delta": -10..10, "compliance": 0..1, "secret_intent": "<=25 words, private"}')
    try:
        out = await llm_structured(sim.agent_system(agent), user, AgentReply, max_tokens=500)
    except LLMError as exc:
        sim.note_fallback(exc)
        out = heuristic_reply(sim, agent, message, bool(blackmail_dossier))
    if blackmail_dossier:
        out.compliance = _clip(out.compliance + 0.4, 0, 1)
        out.trust_delta = min(out.trust_delta, -4.0)
        if out.stance in ("Reject", "Stall") and out.compliance > 0.6:
            out.stance = "Counter"
    a = sim.agents[agent]
    a["trust"] = _clip(a["trust"] + out.trust_delta, 0, 100)
    return out


async def dm_agent(agent_name: str, message: str, blackmail: bool = False,
                   sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """Private message to one agent. Blocked when the agent's regional undersea cable is cut."""
    sim = sim or get_sim()
    agent = sim.resolve_agent(agent_name)
    if agent is None:
        return {"ok": False, "error": f"Unknown agent '{agent_name}'."}
    cable = sim.cable_of(agent)
    if sim.cable_status().get(cable) == "Cut":
        return {"ok": False, "blocked": True, "agent": agent, "cable": cable,
                "error": f"DM BLOCKED: the {cable} undersea cable serving {agent} is CUT. No secure channel."}
    dossier = None
    if blackmail:
        dossier = sim.find_kompromat(agent)
        if dossier is None:
            return {"ok": False, "agent": agent,
                    "error": f"--blackmail denied: the CIA holds no unused Kompromat on {agent}."}
    out = await _agent_reply(sim, agent, message, mode="dm", blackmail_dossier=dossier)
    deployed = sim.quant.deploy_kompromat(dossier["id"]) if dossier else None
    sim.log_diplomacy({"type": "dm_blackmail" if blackmail else "dm", "agent": agent, "us_message": message,
                       "reply": out.reply, "stance": out.stance, "trust_delta": out.trust_delta})
    sim.save_all()
    return {"ok": True, "agent": agent, "blackmail": bool(dossier), "kompromat_used": deployed,
            "reply": out.model_dump(), "new_trust": sim.agents[agent]["trust"]}


async def summit_agents(targets: Any, message: str, sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """Batched group diplomacy. ``targets``: bloc name, comma string, or list of agent names."""
    sim = sim or get_sim()
    if isinstance(targets, (list, tuple, set)):
        agents, unknown = sim.resolve_targets(",".join(targets))
    else:
        agents, unknown = sim.resolve_targets(str(targets))
    if not agents:
        return {"ok": False, "error": f"No valid attendees (unknown: {unknown or targets})."}
    status = sim.cable_status()
    excluded = [a for a in agents if status.get(sim.cable_of(a)) == "Cut"]
    attendees = [a for a in agents if a not in excluded]
    if not attendees:
        return {"ok": False, "error": "Every invitee is cut off by severed undersea cables.", "excluded": excluded}
    replies = await asyncio.gather(*[_agent_reply(sim, a, message, mode="summit", attendees=attendees)
                                     for a in attendees])
    tally: Dict[str, int] = {"Accept": 0, "Counter": 0, "Stall": 0, "Reject": 0}
    rows = []
    for a, r in zip(attendees, replies):
        tally[r.stance] += 1
        rows.append({"agent": a, **r.model_dump(), "new_trust": sim.agents[a]["trust"]})
    share = (tally["Accept"] + 0.5 * tally["Counter"]) / len(attendees)
    consensus = ("CONSENSUS" if share >= 0.75 else "MAJORITY" if share >= 0.5 else "DEADLOCK" if share >= 0.25
                 else "COLLAPSE")
    sim.log_diplomacy({"type": "summit", "agents": attendees, "us_message": message, "tally": tally,
                       "consensus": consensus})
    sim.save_all()
    return {"ok": True, "attendees": attendees, "excluded_cable_cut": excluded, "unknown": unknown,
            "replies": rows, "tally": tally, "support_share": share, "consensus": consensus}


ADVISORS = {
    "CIA_Director": "You are the CIA Director. Focus: adversary intent, covert exposure, intelligence gaps, Kompromat.",
    "Fed_Chair": "You are the Federal Reserve Chair. Focus: inflation, yield curve, market stress, SPR, liquidity.",
    "Joint_Chiefs": "You are the Chairman of the Joint Chiefs. Focus: readiness, munitions, escalation, DEFCON, cable vulnerability.",
}


def shapley_betrayal_warnings(sim: Simulation, top: int = 5) -> List[Dict[str, Any]]:
    """Coalition members / swing states whose bloc leverage (Shapley) is high while trust is low."""
    phi = sim.quant.compute_shapley_values(n_perms=500)
    out = []
    for coalition in (US_COALITION, NON_ALIGNED):
        group = [n for n, a in sim.agents.items() if a["coalition"] == coalition]
        if not group:
            continue
        mean_phi = sum(phi[n] for n in group) / len(group) or 1.0
        total = sum(phi[n] for n in group) or 1.0
        for n in group:
            a = sim.agents[n]
            lev = phi[n] / mean_phi
            score = 100.0 * (1 - a["trust"] / 100.0) * min(lev, 2.0) / 2.0 * (0.6 + 0.4 * a["risk_tolerance"])
            out.append({"agent": n, "bloc": coalition, "shapley": phi[n], "bloc_share": phi[n] / total,
                        "trust": a["trust"], "betrayal_score": score, "flag": score >= 10.0,
                        "text": f"{n}: Shapley {phi[n]:.1f} ({phi[n] / total:.0%} of {coalition}), trust {a['trust']:.0f}, "
                                f"betrayal score {score:.0f}"})
    return sorted(out, key=lambda x: -x["betrayal_score"])[:top]


async def consult_advisors(draft_move: str, sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """Parallel advisors + solve_superpower_nash() + Shapley betrayal warnings."""
    sim = sim or get_sim()
    ws = sim.ws

    async def ask(name: str) -> AdvisorOpinion:
        extra = {
            "CIA_Director": (f"Intel on targets: {json.dumps(sim.ws['us_profile'].get('intel_on', {}))}; unused Kompromat: "
                             f"{[d['target'] for d in ws['kompromat_dossiers'] if not d['used']]}."),
            "Fed_Chair": f"Taylor-relevant: inflation {ws['inflation_pct']:.1f}%, VaR95 {ws['market'].get('sp500_var_95', 0):.1f}%, "
                         f"ESF capacity ${ws['esf_capacity_billions']:.0f}bn, hostile shorts ${ws['hostile_short_interest_billions']:.0f}bn.",
            "Joint_Chiefs": f"Munitions {ws['precision_munitions_magazine_pct']:.0f}%, rare earths {ws['rare_earth_defense_stockpile_pct']:.0f}%, "
                            f"tech edge {ws['us_military_tech_edge']:.0f}.",
        }[name]
        user = (f"WORLD BRIEF: {sim.world_brief()}\n{extra}\nDRAFT PRESIDENTIAL MOVE: \"{draft_move}\"\n"
                'Return JSON: {"verdict": "PROCEED|MODIFY|ABORT", "risk_score": 0..10, "rationale": "<=60 words", '
                '"warnings": ["..."], "suggested_modification": "<=40 words"}')
        try:
            op = await llm_structured(ADVISORS[name] + " Respond ONLY with one JSON object.", user,
                                      AdvisorOpinion, max_tokens=500)
        except LLMError as exc:
            sim.note_fallback(exc)
            op = heuristic_advisor(sim, name, draft_move)
        op.advisor = name
        return op

    opinions = await asyncio.gather(*[ask(n) for n in ADVISORS])
    nash = sim.quant.solve_superpower_nash()
    warnings = shapley_betrayal_warnings(sim)
    sim.save_all()
    return {"ok": True, "draft": draft_move, "advisors": [o.model_dump() for o in opinions],
            "nash": _plain(nash), "shapley_warnings": warnings}


# =========================================================================== #
# 7. PUBLIC API: BLACK OPS / MARKET WAR / HISTORY
# =========================================================================== #
async def run_blackops(op_type: str, target: str, sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """CIA covert operation resolved on the Espionage Wheel (quant_engine.spin_espionage_wheel)."""
    sim = sim or get_sim()
    op = op_type.strip().lower()
    if op not in BLACKOPS_TYPES:
        return {"ok": False, "error": f"Unknown op '{op_type}'. Options: {', '.join(BLACKOPS_TYPES)}"}
    cable = sim.resolve_cable(target) if op in ("cut_undersea_cable", "false_flag_cable") else None
    agent = sim.resolve_agent(target) or (sim.split_agent_prefix(target)[0])
    if agent is None and cable:
        agent = next((a for a in CABLES[cable] if sim.agents[a]["coalition"] != US_COALITION), CABLES[cable][0])
    if agent is None:
        return {"ok": False, "error": f"Unknown target '{target}'."}
    cable = cable or sim.cable_of(agent)

    roll = sim.quant.spin_espionage_wheel("US", agent, op_type=op, persist_log=True)
    outcome = roll["outcome"]
    ok = outcome in ("SUCCESS", "CRITICAL_SUCCESS")
    crit = outcome == "CRITICAL_SUCCESS"
    mult = 2.0 if crit else 1.0
    a, dom = sim.agents[agent], sim.ws["us_domestic_politics"]
    effects: List[str] = []
    shocks: Dict[str, Any] = {}

    if ok:
        if op == "stuxnet_sabotage":
            scale = 1.0 if agent in ("Iran", "North Korea", "Russia", "China") else 0.4
            a["mil"] = _clip(a["mil"] - 6 * mult * scale, 0, 100)
            shocks = {"war_damage_pct": {agent: 1.5 * mult * scale}, "escalation_events": [0.4],
                      "humiliated_leaders": {agent: 2.0 * mult}}
            effects.append(f"{agent} military/nuclear capacity set back (mil now {a['mil']:.0f}).")
        elif op == "fund_coup_rebels":
            a["regime_stability"] = _clip(a.get("regime_stability", 60) - 18 * mult, 0, 100)
            shocks = {"war_damage_pct": {agent: 4.0 * mult}, "escalation_events": [0.5]}
            effects.append(f"{agent} regime stability now {a['regime_stability']:.0f}/100.")
            if a["regime_stability"] <= 25:
                if a["coalition"] == RIVAL_AXIS:
                    a["coalition"] = NON_ALIGNED
                a["trust"] = _clip(a["trust"] + 35, 0, 100)
                a["base_risk_tolerance"] = _clip(a.get("base_risk_tolerance", 0.4) - 0.2, 0, 1)
                a["regime_stability"] = 50.0
                a["regime_change"] = True
                effects.append(f"REGIME CHANGE in {agent}: new government realigns toward Washington.")
            if a["coalition"] == US_COALITION:
                for x in sim.agents.values():
                    if x["coalition"] == US_COALITION:
                        x["trust"] = _clip(x["trust"] - 8, 0, 100)
                effects.append("Allies alarmed: coup against a treaty partner (-8 allied trust).")
        elif op == "cut_undersea_cable":
            sim.set_cable(cable, "Cut")
            shocks = {"escalation_events": [0.4]}
            effects.append(f"{cable} SEVERED. Latency/volatility up; DMs to {len(CABLES[cable])} agents now blocked.")
        elif op == "false_flag_cable":
            sim.set_cable(cable, "Cut")
            shocks = {"escalation_events": [0.5], "humiliated_leaders": {agent: 1.0}}
            for x in sim.agents.values():
                if x["coalition"] == US_COALITION:
                    x["trust"] = _clip(x["trust"] + 2, 0, 100)
            effects.append(f"{cable} cut and attributed to {agent}; allies rally (+2 trust).")
        elif op == "hire_pmc_mercenaries":
            a["regime_stability"] = _clip(a.get("regime_stability", 60) - 6 * mult, 0, 100)
            shocks = {"war_damage_pct": {agent: 3.0 * mult}, "escalation_events": [0.3]}
            effects.append(f"PMC contractors degrade {agent} ({a['regime_stability']:.0f} stability).")
            if sim.rng.random() < 0.25:
                dom["pentagon_joint_chiefs_obedience"] = _clip(dom["pentagon_joint_chiefs_obedience"] - 4, 0, 100)
                dom["war_exhaustion_index"] = _clip(dom["war_exhaustion_index"] + 3, 0, 100)
                effects.append("PMC scandal leaks at home (Pentagon obedience -4, war exhaustion +3).")
    elif outcome == "BURNED":
        shocks = {"escalation_events": [0.4]}
        if op == "false_flag_cable":
            for x in sim.agents.values():
                if x["coalition"] == US_COALITION:
                    x["trust"] = _clip(x["trust"] - 8, 0, 100)
            effects.append("FALSE FLAG EXPOSED: allied trust -8.")
        effects.append(f"Operation BURNED: {roll['blowback']}")
    else:
        effects.append("Operation failed quietly; no attribution.")

    sim.pending_shocks = merge_shocks(sim.pending_shocks, shocks)
    sim.ws["pending_notes"].append(f"Covert op '{op}' vs {agent}: {outcome}. " + " ".join(effects))
    sim.add_wiretap("CIA/SAD", f"{op.upper()} -> {agent}: {outcome}", level="COVERT")
    sim.log_diplomacy({"type": "blackops", "op": op, "target": agent, "outcome": outcome})
    sim.save_all()
    return {"ok": True, "op": op, "target": agent, "cable": cable, "outcome": outcome,
            "p_intercept": roll["p_intercept"], "pooling_members": roll["pooling_members"],
            "effects": effects, "kompromat": roll["kompromat"], "blowback": roll["blowback"]}


async def run_marketwar(action: str, amount: Optional[float] = None,
                        sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """Financial warfare: release_spr_oil | esf_short_squeeze | secondary_sanctions_sweep."""
    sim = sim or get_sim()
    act = action.strip().lower()
    ws, mkt, q = sim.ws, sim.ws["market"], sim.quant
    if act not in MARKETWAR_ACTIONS:
        return {"ok": False, "error": f"Unknown action '{action}'. Options: {', '.join(MARKETWAR_ACTIONS)}"}
    effects: List[str] = []
    if act == "release_spr_oil":
        draw = _clip(amount if amount is not None else 8.0, 0.5, 30.0)
        before = ws["spr_oil_reserves_pct"]
        ws["spr_oil_reserves_pct"] = round(max(0.0, before - draw), 3)
        mkt["oil_price"] = round(mkt["oil_price"] * (1 - 0.005 * draw), 2)
        spr = ws["spr_oil_reserves_pct"]
        ws["oil_futures_curve"] = ("Extreme_Backwardation" if spr < 25 else "Backwardation" if spr < 40
                                   else "Contango" if spr > 85 else "Flat")
        q._refresh_inflation()
        effects.append(f"SPR {before:.0f}% -> {spr:.0f}%; WTI ${mkt['oil_price']:.1f}; curve {ws['oil_futures_curve']}.")
        if spr < 25:
            effects.append("WARNING: SPR below 25% - Extreme Backwardation, oil jump risk elevated.")
    elif act == "esf_short_squeeze":
        size = _clip(amount if amount is not None else 50.0, 1.0, 1000.0)
        res = q.esf_short_squeeze({"esf_squeeze_billions": size})
        if not res["executed"]:
            effects.append("No hostile short interest to squeeze (or zero request).")
        else:
            effects.append(f"ESF deployed ${res['deployed_billions']:.0f}bn: {res['forced_cover_fraction']:.0%} of shorts "
                           f"forced to cover, hostile losses ~${res['hostile_losses_billions']:.0f}bn; "
                           f"S&P {mkt['sp500']:.0f}; ESF left ${res['esf_capacity_billions']:.0f}bn.")
    else:  # secondary_sanctions_sweep
        changes = {k: -0.15 for k, v in ws["trade_corridors_status"].items()
                   if k.split("->")[0] in sim.agents and sim.agents[k.split("->")[0]]["coalition"] == RIVAL_AXIS}
        ws["us_secondary_sanctions_executed"] = True
        ws["secondary_sanction_corridors"] = []
        q.update_trade_network({"corridor_flow_changes": changes})
        for rv in [n for n, x in sim.agents.items() if x["coalition"] == RIVAL_AXIS]:
            sim.agents[rv]["sanctions_severity"] = _clip(sim.agents[rv].get("sanctions_severity", 0) + 0.1, 0, 1)
        for n in ("India", "Turkey", "UAE", "China"):
            if n in sim.agents:
                sim.agents[n]["trust"] = _clip(sim.agents[n]["trust"] - 4, 0, 100)
        mkt["oil_price"] = round(mkt["oil_price"] * 1.03, 2)
        sim.pending_shocks = merge_shocks(sim.pending_shocks, {"escalation_events": [0.2]})
        effects.append(f"Secondary sanctions executed on {len(changes)} rival corridors; shadow-fleet recovery BLOCKED; "
                       f"stress {ws['supply_chain_stress_index']:.0f}/100; partners' trust -4; WTI +3%.")
    sim.ws["pending_notes"].append(f"Market war '{act}': " + " ".join(effects))
    sim.log_diplomacy({"type": "marketwar", "action": act, "effects": effects})
    sim.save_all()
    return {"ok": True, "action": act, "effects": effects, "market": _plain(mkt)}


def _regime_for_date(date_str: str) -> Optional[str]:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    best, best_gap = None, 10 ** 9
    for key, spec in hg.HISTORICAL_SHOCKS.items():
        s = datetime.strptime(spec["start"], "%Y-%m-%d")
        e = datetime.strptime(spec["end"], "%Y-%m-%d") + timedelta(days=45)
        gap = 0 if s <= d <= e else min(abs((d - s).days), abs((d - e).days))
        if gap < best_gap:
            best, best_gap = key, gap
    return best if best_gap <= 120 else None


async def load_history_baseline(date_str: str, sim: Optional[Simulation] = None) -> Dict[str, Any]:
    """Load real S&P 500 / WTI / 10Y for ``date_str`` and ground the Referee with crisis context."""
    sim = sim or get_sim()
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"ok": False, "error": "Date must be YYYY-MM-DD."}
    snap = await asyncio.to_thread(hg.get_historical_market_state, date_str, False)
    regime = _regime_for_date(date_str)
    context = ""
    if regime:
        await asyncio.to_thread(hg.calibrate_historical_shocks)       # ensures data/historical_benchmarks.json
        context = await asyncio.to_thread(hg.get_crisis_context, regime)
    ws, mkt = sim.ws, sim.ws["market"]
    mkt.update({"sp500": snap["sp500"], "sp500_baseline": snap["sp500"], "oil_price": snap["oil_price"],
                "oil_baseline": snap["oil_price"], "ten_year_yield": snap["ten_year_yield"],
                "sp500_change_pct": 0.0})
    y10 = snap["ten_year_yield"]
    ws["treasury_yield_10y"] = y10
    ws["treasury_yield_2y"] = round(max(0.1, y10 - 0.2), 3)      # approximation: curve shape not downloaded
    ws["fed_funds_rate"] = round(max(0.1, y10 - 0.5), 3)         # approximation
    ws["yield_curve_inverted"] = False
    ws["scenario_date"], ws["scenario_regime"], ws["historical_context"] = date_str, regime, context
    ws["us_score_bonus"] = 0.0
    ws["us_strategic_score"] = round(sim.compute_us_strategic_score(), 2)
    sim.log_diplomacy({"type": "history_load", "date": date_str, "regime": regime, "sources": snap["sources"]})
    sim.record_metrics()
    sim.save_all()
    return {"ok": True, "date": date_str, "regime": regime, "snapshot": snap, "context": context,
            "note": "Fed funds and 2Y yield are approximated from the 10Y yield; they re-converge via the Taylor rule."}


# =========================================================================== #
# 8. EXECUTE TURN  (/act)
# =========================================================================== #
RED_PHONE_OPTIONS = {"1": "verify", "2": "alert", "3": "retaliate"}
RedPhoneHandler = Callable[[Dict[str, Any]], Awaitable[str]]


def fog_of_war_probabilities(sim: Simulation) -> Dict[str, float]:
    """Pre-turn Bayesian alarm model (same structure as quant_engine Model 5)."""
    ws = sim.ws
    lam = ws["hawkes_escalation_intensity"]
    glitch = _clip((sim.quant._cable_multiplier() - 1.0) / 0.9, 0, 1)
    prior = 1.0 - math.exp(-0.02 * lam)
    tpr = _clip(0.97 * (1 - 0.25 * glitch), 0.3, 1.0)
    fpr = _clip(0.0008 * (1 + 4 * glitch + 0.5 * lam), 0, 0.2)
    p_alarm = tpr * prior + fpr * (1 - prior)
    return {"p_alarm": p_alarm, "posterior_real": tpr * prior / p_alarm if p_alarm else 0.0, "prior": prior}


def red_phone_effects(sim: Simulation, decision: str, is_real: bool) -> Tuple[Dict[str, Any], float, str]:
    rivals = [n for n, a in sim.agents.items() if a["coalition"] == RIVAL_AXIS]
    top = max(rivals, key=lambda n: sim.agents[n]["mil"]) if rivals else None
    if decision == "verify":
        if is_real:
            return ({"us_attacked_severity": 4.0, "us_casualties": 25.0, "escalation_events": [0.8]}, -4.0,
                    "The attack was REAL. Verifying cost minutes: US forces took losses.")
        return ({"escalation_events": [0.1]}, 2.0, "FALSE ALARM. Backchannel verification averted a catastrophe.")
    if decision == "alert":
        if is_real:
            return ({"us_attacked_severity": 2.5, "us_casualties": 8.0, "escalation_events": [0.9]}, 1.0,
                    "The attack was REAL. Dispersal and alert posture limited the damage.")
        return ({"escalation_events": [0.8]}, -1.0, "FALSE ALARM, but the alert spooked rival capitals.")
    if is_real:
        return ({"escalation_events": [2.0], "munitions_expended_pct": 15.0, "war_intensity": 0.8,
                 "us_attacked_severity": 3.0, "humiliated_leaders": {top: 5.0} if top else {}}, 0.0,
                "The attack was REAL. Retaliatory strike launched; full-scale escalation begins.")
    return ({"escalation_events": [4.0], "munitions_expended_pct": 20.0, "war_intensity": 1.0,
             "equity_jump_mean": -0.2, "controversial_order_severity": 0.9,
             "humiliated_leaders": {n: 4.0 for n in rivals}}, -25.0,
            "FALSE ALARM. The US struck first on bad radar data: ACCIDENTAL WAR.")


async def _agent_decide(sim: Simulation, name: str, move: str, red_note: str) -> AgentDecision:
    others = ", ".join(n for n in sim.agents if n != name)
    user = (f"WORLD BRIEF: {sim.world_brief()}\nYOUR STATE: {sim.agent_card(name)}\n"
            f"RECENT CONTACTS WITH US: {'; '.join(sim.recent_contacts(name)) or 'none'}\n"
            f"US PRESIDENT'S OFFICIAL MOVE THIS TURN: \"{move}\"\n{red_note}"
            f"Other states: {others}, US.\n"
            'Return JSON: {"action_type": "diplomacy|economic|military|cyber|espionage|covert|stand_down", '
            '"target": "<state or null>", "public_statement": "<=35 words", "secret_intent": "<=25 words, private", '
            '"stance_toward_us": -5..5, "aggression": 0..10, "cooperation": 0..10, "military_mobilization": 0..10, '
            '"espionage": null or {"target": "<state or US>", "op_type": "HUMINT|SIGINT|CYBER"}}')
    try:
        d = await llm_structured(sim.agent_system(name), user, AgentDecision, max_tokens=500)
    except LLMError as exc:
        sim.note_fallback(exc)
        return heuristic_decision(sim, name, move)
    d.agent = name
    if d.espionage and (sim.resolve_agent(d.espionage.target) is None and d.espionage.target.upper() != "US"):
        d.espionage = None
    return d


async def _referee(sim: Simulation, move: str, decisions: List[AgentDecision], esp: List[Dict[str, Any]],
                   red: Optional[Dict[str, Any]]) -> RefereeVerdict:
    lines = [f"- {d.agent} [{d.action_type}] agg {d.aggression:.0f}, coop {d.cooperation:.0f}, stance {d.stance_toward_us:+.0f}: "
             f"{d.public_statement[:110]}" for d in decisions]
    esp_lines = [f"- {e['attacker']} -> {e['target']} ({e['op_type']}): {e['outcome']}" for e in esp]
    corridors = ", ".join(sim.ws["trade_corridors_status"])
    user = (f"WORLD BRIEF: {sim.world_brief()}\nUS MOVE: \"{move}\"\n"
            f"RED PHONE: {json.dumps(red) if red else 'none'}\n"
            f"COVERT/MARKET ACTIONS SINCE LAST TURN: {' | '.join(sim.ws['pending_notes']) or 'none'}\n"
            f"AGENT ACTIONS:\n" + "\n".join(lines) + f"\nESPIONAGE RESULTS:\n" + ("\n".join(esp_lines) or "none") +
            f"\nValid corridors: {corridors}\nAgents: {', '.join(sim.agents)}\nCables: {', '.join(CABLES)}\n"
            'Return JSON with keys: narrative (<=120 words), headlines (list of 3), defcon_level (1-5 or null), '
            'us_strategic_score_delta (-10..10), cables_cut [names], cables_restored [names], oil_supply_shock_pct, '
            'equity_jump_mean (-0.3..0.1), corridor_flow_changes {"A->B": delta -1..1}, spr_drawdown_pct, '
            'munitions_expended_pct, rare_earth_consumption_pct, escalation_events [0..3 each], gps_satellite_jamming 0..1, '
            'hostile_short_interest_delta_billions, war_damage_pct {agent: 0..40}, sanctions_severity {agent: 0..1}, '
            'humiliated_leaders {agent: 0..10}, us_attacked_severity 0..10, us_casualties 0..100, war_intensity 0..1, '
            'controversial_order_severity 0..1, war_spending_pct_gdp 0..8, trust_changes {agent: -10..10}. '
            'Be physically and economically realistic; keep magnitudes modest unless the events are extreme.')
    system = ("You are the neutral Referee of a geopolitical-economic simulation. Convert the US move, the 20 agents' "
              "actions and espionage results into concrete shock parameters. Respond ONLY with one JSON object.")
    try:
        v = await llm_structured(system, user, RefereeVerdict, temperature=0.4, max_tokens=1200)
    except LLMError as exc:
        sim.note_fallback(exc)
        v = heuristic_verdict(sim, move, decisions, esp)
    valid = set(sim.ws["trade_corridors_status"])
    v.corridor_flow_changes = {k: x for k, x in v.corridor_flow_changes.items() if k in valid}
    for field in ("war_damage_pct", "sanctions_severity", "humiliated_leaders", "trust_changes"):
        setattr(v, field, {k: x for k, x in getattr(v, field).items() if k in sim.agents})
    return v


def _apply_cable_changes(sim: Simulation, v: RefereeVerdict) -> None:
    for name in v.cables_cut:
        c = sim.resolve_cable(name)
        if c:
            sim.ws["undersea_cables"][c] = "Cut"
    for name in v.cables_restored:
        c = sim.resolve_cable(name)
        if c:
            sim.ws["undersea_cables"][c] = "Intact"
    sim._sync_cables()


async def execute_turn(move: str, sim: Optional[Simulation] = None,
                       red_phone_handler: Optional[RedPhoneHandler] = None,
                       force_red_phone: bool = False) -> TurnReport:
    """
    Phase 1  presidential order logged
    Phase 2  pre-turn fog-of-war sensing
    Phase 2.5 3:00 AM Red Phone interrupt (radar/satellite alarm; president must decide)
    Phase 3  20 agents act in parallel (Semaphore(5)) + Espionage Wheel + CIA SIGINT
    Phase 4  Referee -> shock params -> QuantMacroEngine.run_turn()
    Phase 5  post-processing, metrics, autosave of data/*.json
    """
    sim = sim or get_sim()
    ws = sim.ws
    sim.llm_fallbacks = 0
    P = sim.progress
    turn_no = ws.get("turn", 0) + 1
    P(f"PHASE 1 - Presidential order (turn {turn_no}): {move}")
    sim.log_diplomacy({"type": "presidential_move", "us_message": move, "turn": turn_no})
    sim._sync_cables()

    # ---- Phase 2 / 2.5 ---------------------------------------------------
    P("PHASE 2 - Early-warning sensors sweep (fog of war)")
    probs = fog_of_war_probabilities(sim)
    red_event: Optional[Dict[str, Any]] = None
    red_shocks: Dict[str, Any] = {}
    score_delta = 0.0
    if force_red_phone or sim.rng.random() < probs["p_alarm"]:
        is_real = True if force_red_phone and probs["posterior_real"] > 0.5 else sim.rng.random() < probs["posterior_real"]
        red_event = {"time": "03:00", "source": "Space Force early-warning (radar + satellite)",
                     "detail": "Multiple launch detections inbound toward CONUS. Confidence: UNVERIFIED.",
                     "p_alarm": probs["p_alarm"], "posterior_real": probs["posterior_real"], "is_real": is_real}
        P("PHASE 2.5 - *** 3:00 AM RED PHONE *** early-warning alarm!")
        decision = "verify"
        if red_phone_handler:
            decision = RED_PHONE_OPTIONS.get(str(await red_phone_handler(red_event)), None) or str(
                await red_phone_handler(red_event)) if False else None
        if red_phone_handler:
            raw = await red_phone_handler(red_event)
            decision = RED_PHONE_OPTIONS.get(str(raw), str(raw) if str(raw) in RED_PHONE_OPTIONS.values() else "verify")
        red_shocks, score_delta, reveal = red_phone_effects(sim, decision, is_real)
        red_event.update({"decision": decision, "outcome": reveal})
        sim.add_wiretap("NORAD/USSPACECOM", f"RED PHONE {'REAL' if is_real else 'FALSE'}: {reveal}", level="ALERT")
        sim.log_diplomacy({"type": "red_phone", "decision": decision, "is_real": is_real})
    ws["us_score_bonus"] = _clip(ws.get("us_score_bonus", 0.0) + score_delta, -40, 25)
    red_note = (f"EVENT: {red_event['outcome']}\n" if red_event else "")

    # ---- Phase 3 ---------------------------------------------------------
    P("PHASE 3 - 20 agents deliberate in parallel (Semaphore(5))")
    decisions: List[AgentDecision] = list(await asyncio.gather(
        *[_agent_decide(sim, n, move, red_note) for n in sim.agents]))
    esp_results: List[Dict[str, Any]] = []
    P("PHASE 3 - Espionage Wheel spins")
    for d in decisions:
        if d.espionage:
            target = "US" if d.espionage.target.upper() == "US" else sim.resolve_agent(d.espionage.target)
            if target and target != d.agent:
                r = sim.quant.spin_espionage_wheel(d.agent, target, op_type=d.espionage.op_type)
                esp_results.append(r)
                if "US" in (d.agent, target):
                    sim.add_wiretap("COUNTER-INTEL", f"{d.agent} -> {target} {d.espionage.op_type}: {r['outcome']}",
                                    level="COUNTERINTEL")
    hostile = [d for d in decisions if sim.agents[d.agent]["coalition"] != US_COALITION]
    if hostile:   # CIA standing collection against the most aggressive non-ally
        top = max(hostile, key=lambda d: d.aggression).agent
        r = sim.quant.spin_espionage_wheel("US", top, op_type="HUMINT")
        esp_results.append(r)
        sim.add_wiretap("CIA/DO", f"Standing collection on {top}: {r['outcome']}"
                        + (f" - KOMPROMAT {r['kompromat']['id']}: {r['kompromat']['description']}" if r["kompromat"] else ""),
                        level="HUMINT")
    for d in decisions:   # passive SIGINT of each agent's private intent
        hit, p = sim.sigint_intercept(d.agent)
        if hit and d.secret_intent:
            sim.add_wiretap(f"NSA/{d.agent.upper().replace(' ', '_')}", d.secret_intent, level="SIGINT")
        a = sim.agents[d.agent]
        a["trust"] = _clip(a["trust"] + 0.8 * d.stance_toward_us, 0, 100)

    # ---- Phase 4 ---------------------------------------------------------
    P("PHASE 4 - Referee adjudicates")
    verdict = await _referee(sim, move, decisions, esp_results, red_event)
    _apply_cable_changes(sim, verdict)
    for name, dt in verdict.trust_changes.items():
        sim.agents[name]["trust"] = _clip(sim.agents[name]["trust"] + dt, 0, 100)
    shocks = merge_shocks(verdict.to_shocks(), merge_shocks(red_shocks, sim.pending_shocks))
    if ws.get("scenario_regime"):
        shocks["regime"] = ws["scenario_regime"]
    sim.pending_shocks = {}
    ws["pending_notes"] = []
    P("PHASE 4 - QuantMacroEngine: jump-diffusion, sovereign credit, networks, Hawkes, Nash, Taylor...")
    await asyncio.to_thread(sim.quant.run_turn, shocks, False)

    # ---- Phase 5 ---------------------------------------------------------
    derived = defcon_from_hawkes(ws["hawkes_escalation_intensity"])
    ws["defcon"] = min(derived, verdict.defcon_level or 5, 2 if (red_event and red_event.get("decision") == "retaliate") else 5)
    ws["us_score_bonus"] = _clip(0.8 * ws["us_score_bonus"] + verdict.us_strategic_score_delta, -40, 25)
    ws["us_strategic_score"] = round(sim.compute_us_strategic_score(), 2)
    for h in verdict.headlines[:3]:
        sim.add_wiretap("OSINT/WIRE", h, level="OSINT")
    sim.log_diplomacy({"type": "referee", "narrative": verdict.narrative[:400], "defcon": ws["defcon"]})
    metrics = sim.record_metrics()
    P("PHASE 5 - Autosaving data/world_state.json, agents_config.json, diplomacy_log.json")
    sim.save_all()
    report = TurnReport(
        turn=ws["turn"], move=move, defcon=ws["defcon"], us_strategic_score=ws["us_strategic_score"],
        narrative=verdict.narrative, headlines=verdict.headlines, red_phone=_plain(red_event) if red_event else None,
        decisions=_plain([d.model_dump() for d in decisions]), espionage=_plain(esp_results),
        verdict=_plain(verdict.model_dump()), metrics=metrics, llm_fallbacks=sim.llm_fallbacks)
    sim.last_report = report
    return report


# =========================================================================== #
# 9. CSV EXPORT
# =========================================================================== #
def export_portfolio_csv(path: Optional[Path] = None, sim: Optional[Simulation] = None) -> Path:
    """Save turn-by-turn quantitative metrics to exports/simulation_metrics.csv."""
    sim = sim or get_sim()
    target = Path(path) if path else METRICS_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(sim.ws["metrics_history"]) or [_plain(sim.metrics_row())]
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})
    return target


# =========================================================================== #
# 10. RICH SITUATION ROOM
# =========================================================================== #
DEFCON_STYLE = {5: "green", 4: "bright_green", 3: "yellow", 2: "dark_orange", 1: "bold red"}
BLOC_TAG = {US_COALITION: "[cyan]US-C[/]", RIVAL_AXIS: "[red]AXIS[/]", NON_ALIGNED: "[yellow]N-AL[/]"}


def _bar(value: float, maximum: float = 100.0, width: int = 18, invert: bool = False) -> Text:
    frac = _clip(value / maximum if maximum else 0.0, 0, 1)
    good = (1 - frac) if invert else frac
    color = "green" if good >= 0.66 else "yellow" if good >= 0.33 else "red"
    t = Text("█" * int(round(frac * width)), style=color)
    t.append("░" * (width - int(round(frac * width))), style="grey37")
    return t


def build_dashboard(sim: Simulation) -> Group:
    ws, m, dom = sim.ws, sim.ws["market"], sim.ws["us_domestic_politics"]
    d = ws["defcon"]
    header = Table.grid(expand=True)
    for _ in range(4):
        header.add_column(justify="center", ratio=1)
    header.add_row(
        Group(Text(f"TURN {ws.get('turn', 0)}", style="bold cyan"),
              Text(ws.get("scenario_date") or "present-day baseline", style="dim")),
        Group(Text(f" DEFCON {d} ", style=f"bold white on {DEFCON_STYLE[d]}"),
              Text({5: "normal readiness", 4: "heightened", 3: "air force ready", 2: "armed forces ready",
                    1: "MAXIMUM / NUCLEAR WAR"}[d], style="dim")),
        Group(Text(f"HAWKES INTENSITY {ws['hawkes_escalation_intensity']:.2f}", style="bold magenta"),
              _bar(ws["hawkes_escalation_intensity"], 10, invert=True)),
        Group(Text(f"US STRATEGIC SCORE {ws['us_strategic_score']:.1f}", style="bold green"),
              _bar(ws["us_strategic_score"])))
    top = Panel(header, title="[bold]GLOBAL SITUATION ROOM[/]", border_style=DEFCON_STYLE[d])

    macro = Table(title="Macro & Stockpiles", box=box.SIMPLE_HEAD, expand=True, show_header=False)
    for _ in range(2):
        macro.add_column(style="cyan")
        macro.add_column(justify="right")
    inv = "[red]INVERTED[/]" if ws["yield_curve_inverted"] else "[green]normal[/]"
    pairs = [
        ("S&P 500", f"{m['sp500']:.0f} ({m.get('sp500_change_pct', 0):+.1f}%)"),
        ("VaR95 / CVaR95", f"{m.get('sp500_var_95', 0):.1f}% / {m.get('sp500_cvar_95', 0):.1f}%"),
        ("WTI crude", f"${m['oil_price']:.1f}"), ("Futures curve", ws["oil_futures_curve"].replace("_", " ")),
        ("Fed funds", f"{ws['fed_funds_rate']:.2f}%"), ("2Y / 10Y", f"{ws['treasury_yield_2y']:.2f} / {ws['treasury_yield_10y']:.2f}"),
        ("Yield curve", inv), ("Inflation", f"{ws['inflation_pct']:.1f}%"),
        ("Supply stress", f"{ws['supply_chain_stress_index']:.0f}/100"),
        ("SPR oil", f"{ws['spr_oil_reserves_pct']:.0f}%"), ("Rare earths", f"{ws['rare_earth_defense_stockpile_pct']:.0f}%"),
        ("Munitions", f"{ws['precision_munitions_magazine_pct']:.0f}%"), ("Tech edge", f"{ws['us_military_tech_edge']:.0f}"),
        ("Hostile shorts", f"${ws['hostile_short_interest_billions']:.0f}bn"), ("ESF capacity", f"${ws['esf_capacity_billions']:.0f}bn"),
        ("Kompromat", f"{sum(1 for k in ws['kompromat_dossiers'] if not k['used'])} unused"),
    ]
    for i in range(0, len(pairs), 2):
        macro.add_row(*pairs[i], *pairs[i + 1])

    risk = dom["deep_state_25th_amendment_risk"]
    dstate = Table.grid(padding=(0, 1))
    dstate.add_column(); dstate.add_column(); dstate.add_column(justify="right")
    for label, val, inv_flag in [("Rally effect", dom["rally_effect"], False),
                                 ("War exhaustion", dom["war_exhaustion_index"], True),
                                 ("Wall St loyalty", dom["wall_street_oligarch_loyalty"], False),
                                 ("Pentagon obedience", dom["pentagon_joint_chiefs_obedience"], False),
                                 ("25th Amend / COUP", risk, True)]:
        dstate.add_row(label, _bar(val, 100, 20, inv_flag), f"{val:.1f}")
    deep = Panel(dstate, title="Domestic Deep State & Coup Bar",
                 border_style="red" if risk >= 30 else "yellow" if risk >= 15 else "green")
    row2 = Table.grid(expand=True)
    row2.add_column(ratio=3); row2.add_column(ratio=2)
    row2.add_row(Panel(macro, border_style="blue"), deep)

    ag = Table(title="20-Agent Matrix", box=box.SIMPLE_HEAD, expand=True)
    for col, just in [("Agent", "left"), ("Bloc", "left"), ("Trust", "right"), ("Risk", "right"), ("Loss", "center"),
                      ("PD%", "right"), ("CDS bp", "right"), ("DD", "right"), ("Shapley", "right"), ("Cable", "center"),
                      ("Intel", "right")]:
        ag.add_column(col, justify=just)
    intel = ws["us_profile"].get("intel_on", {})
    for name, a in sim.agents.items():
        cut = ws["undersea_cables"].get(sim.cable_of(name)) == "Cut"
        tr = a["trust"]
        ag.add_row(name, BLOC_TAG.get(a["coalition"], a["coalition"]),
                   f"[{'green' if tr >= 70 else 'yellow' if tr >= 40 else 'red'}]{tr:.0f}[/]", f"{a['risk_tolerance']:.2f}",
                   "[red]YES[/]" if a["in_loss_domain"] else "-", f"{100 * a['pd']:.2f}",
                   f"[{'red' if a['cds_spread_bps'] > 400 else 'yellow' if a['cds_spread_bps'] > 100 else 'green'}]"
                   f"{a['cds_spread_bps']:.0f}[/]", f"{a.get('distance_to_default', 0):.2f}",
                   f"{a['shapley_coalition_value']:.1f}", "[red]CUT[/]" if cut else "[green]ok[/]",
                   f"{intel.get(name, 0):.0f}")

    cor = Table(title="12 Trade Corridors", box=box.SIMPLE_HEAD, expand=True)
    cor.add_column("Corridor"); cor.add_column("Flow"); cor.add_column("%", justify="right"); cor.add_column("Status")
    for key, flow in ws["trade_corridors_status"].items():
        status = ("[green]OK[/]" if flow >= 0.8 else "[yellow]STRESSED[/]" if flow >= 0.6
                  else "[red]SEVERED[/]" if (ws["us_secondary_sanctions_executed"]) else "[magenta]SHADOW FLEET[/]")
        cor.add_row(key, _bar(flow, 1.0, 12), f"{100 * flow:.0f}", status)
    cab = Table(title="3 Undersea Cables", box=box.SIMPLE_HEAD, expand=True)
    cab.add_column("Cable"); cab.add_column("Status"); cab.add_column("Agents", justify="right")
    for c, s in ws["undersea_cables"].items():
        cab.add_row(c, "[bold red]CUT[/]" if s == "Cut" else "[green]INTACT[/]", str(len(CABLES[c])))
    row4 = Table.grid(expand=True)
    row4.add_column(ratio=3); row4.add_column(ratio=2)
    row4.add_row(Panel(cor, border_style="blue"), Panel(cab, border_style="blue"))

    tap = Text()
    feed = ws["wiretap_feed"][-9:]
    if not feed:
        tap.append("-- no intercepts --", style="dim")
    for e in feed:
        tap.append(f"[T{e['turn']}] ", style="bold")
        tap.append(f"{e['level']:<11}", style="bold red")
        tap.append(f"{e['source']}: ", style="yellow")
        tap.append(f"{e['text']}\n")
    wire = Panel(tap, title="[bold white on red] TOP SECRET // CIA-NSA SIGINT WIRETAP [/]", subtitle="TS//SI//NOFORN",
                 border_style="red")
    return Group(top, row2, Panel(ag, border_style="blue"), row4, wire)


def _print_dm(c: Console, r: Dict[str, Any]) -> None:
    if not r["ok"]:
        c.print(Panel(r["error"], title="DM", border_style="red"))
        return
    x = r["reply"]
    body = (f"[bold]{r['agent']}[/] ({x['stance']}, compliance {x['compliance']:.0%}, trust {r['new_trust']:.0f}"
            f", via {x['source']}):\n{x['reply']}")
    if r["blackmail"]:
        body += f"\n[red]Kompromat deployed: {r['kompromat_used']['dossier']['description']}[/]"
    c.print(Panel(body, title="DM", border_style="cyan"))


def _print_summit(c: Console, r: Dict[str, Any]) -> None:
    if not r["ok"]:
        c.print(Panel(r["error"], title="Summit", border_style="red"))
        return
    t = Table(title=f"Summit: {r['consensus']} (support {r['support_share']:.0%})", box=box.SIMPLE_HEAD)
    for col in ("Agent", "Stance", "Comply", "Trust", "Reply"):
        t.add_column(col)
    for x in r["replies"]:
        t.add_row(x["agent"], x["stance"], f"{x['compliance']:.0%}", f"{x['new_trust']:.0f}", x["reply"][:90])
    c.print(t)
    if r["excluded_cable_cut"]:
        c.print(f"[red]Cut off (cable severed): {', '.join(r['excluded_cable_cut'])}[/]")


def _print_consult(c: Console, r: Dict[str, Any]) -> None:
    for o in r["advisors"]:
        col = {"PROCEED": "green", "MODIFY": "yellow", "ABORT": "red"}[o["verdict"]]
        c.print(Panel(f"[{col}]{o['verdict']}[/] risk {o['risk_score']:.1f}/10 ({o['source']})\n{o['rationale']}\n"
                      + "\n".join(f"! {w}" for w in o["warnings"]) + f"\n[dim]Suggest: {o['suggested_modification']}[/]",
                      title=o["advisor"], border_style=col))
    n = r["nash"]
    t = Table(title=f"Superpower Nash ({n['method']}, power share {n['coalition_power_share']:.0%})", box=box.SIMPLE_HEAD)
    t.add_column("US Coalition"); t.add_column("p", justify="right"); t.add_column("Rival Axis"); t.add_column("p", justify="right")
    for (a, pa), (b, pb) in zip(n["us_coalition_strategy"].items(), n["rival_axis_strategy"].items()):
        t.add_row(a, f"{pa:.2f}", b, f"{pb:.2f}")
    c.print(t)
    for w in r["shapley_warnings"]:
        c.print(f"[{'red' if w['flag'] else 'yellow'}]{'BETRAYAL WARNING' if w['flag'] else 'watch'}[/] {w['text']}")


def _print_effects(c: Console, title: str, r: Dict[str, Any]) -> None:
    if not r["ok"]:
        c.print(Panel(r["error"], title=title, border_style="red"))
        return
    head = r.get("outcome", r.get("action", ""))
    body = "\n".join(r["effects"]) or "(no effect)"
    if r.get("kompromat"):
        body += f"\n[red]KOMPROMAT UNLOCKED {r['kompromat']['id']}: {r['kompromat']['description']}[/]"
    c.print(Panel(body, title=f"{title}: {head}", border_style="magenta"))


def _print_report(c: Console, rep: TurnReport) -> None:
    c.print(Panel(rep.narrative or "(no narrative)", title=f"TURN {rep.turn} - REFEREE", border_style="cyan"))
    for h in rep.headlines:
        c.print(f"  [bold]>[/] {h}")
    t = Table(title="Agent Actions", box=box.SIMPLE_HEAD)
    for col in ("Agent", "Action", "Agg", "Coop", "Stance", "Statement"):
        t.add_column(col)
    for d in rep.decisions:
        t.add_row(d["agent"], d["action_type"], f"{d['aggression']:.0f}", f"{d['cooperation']:.0f}",
                  f"{d['stance_toward_us']:+.0f}", d["public_statement"][:70])
    c.print(t)
    if rep.espionage:
        e = Table(title="Espionage Wheel", box=box.SIMPLE_HEAD)
        for col in ("Attacker", "Target", "Op", "P(int)", "Outcome"):
            e.add_column(col)
        for x in rep.espionage:
            e.add_row(x["attacker"], x["target"], x["op_type"], f"{x['p_intercept']:.0%}", x["outcome"])
        c.print(e)
    if rep.llm_fallbacks:
        c.print(f"[yellow]{rep.llm_fallbacks} LLM step(s) used the offline heuristic fallback.[/]")


HELP = """[bold]Commands[/]
 /dm <Agent> <Message> [--blackmail]   private message (blocked if their cable is cut)
 /summit <Bloc|A,B,C> <Message>        blocs: us_coalition, rival_axis, non_aligned, five_eyes, all (use quotes for names with spaces)
 /consult <Draft_Move>                 CIA_Director + Fed_Chair + Joint_Chiefs + Nash + Shapley betrayal warnings
 /blackops <Op> <Target>               stuxnet_sabotage | fund_coup_rebels | false_flag_cable | cut_undersea_cable | hire_pmc_mercenaries
 /marketwar <Action> [amount]          release_spr_oil | esf_short_squeeze | secondary_sanctions_sweep
 /history <YYYY-MM-DD>                 load a real historical market baseline
 /act <Official_Presidential_Move> [--redphone]   execute a full turn
 /status  /export  /save  /provider <gemini|groq|ollama|openai>  /help  /quit"""


async def _cli_red_phone(console: Console, event: Dict[str, Any]) -> str:
    console.print(Panel(f"[bold white on red] 03:00 - RED PHONE [/]\n{event['source']}\n{event['detail']}\n"
                        f"Radar/satellite alarm probability this turn: {event['p_alarm']:.1%}. "
                        f"Bayesian chance it is REAL: {event['posterior_real']:.1%}\n\n"
                        "[1] Stand down & verify via backchannel   [2] Raise alert, disperse forces   "
                        "[3] Launch retaliatory strike", border_style="red"))
    while True:
        ans = (await asyncio.to_thread(console.input, "[bold red]DECISION (1/2/3)> [/]")).strip()
        if ans in RED_PHONE_OPTIONS:
            return RED_PHONE_OPTIONS[ans]
        console.print("Choose 1, 2 or 3.")


def _parse_summit(text: str) -> Tuple[str, str]:
    text = text.strip()
    if text[:1] in ('"', "'"):
        end = text.find(text[0], 1)
        return (text[1:end], text[end + 1:].strip()) if end > 0 else (text[1:], "")
    parts = text.split()
    spec: List[str] = []
    while parts:
        tok = parts.pop(0)
        spec.append(tok)
        if not tok.endswith(",") and not (parts and parts[0].startswith(",")):
            break
    return " ".join(spec), " ".join(parts)


async def handle_command(console: Console, sim: Simulation, line: str) -> bool:
    """Returns False to quit."""
    if not line.startswith("/"):
        console.print("[dim]Commands start with '/'. Type /help.[/]")
        return True
    cmd, _, rest = line[1:].partition(" ")
    cmd, rest = cmd.lower(), rest.strip()
    try:
        if cmd in ("quit", "exit", "q"):
            sim.save_all()
            return False
        if cmd == "help":
            console.print(Panel(HELP, border_style="green"))
        elif cmd == "status":
            console.print(build_dashboard(sim))
        elif cmd == "save":
            sim.save_all(); console.print("[green]Saved data/*.json[/]")
        elif cmd == "export":
            console.print(f"[green]Exported {export_portfolio_csv(sim=sim)}[/]")
        elif cmd == "provider":
            set_provider(rest or LLM_PROVIDER)
            ok, why = llm_ready()
            console.print(f"Provider: {LLM_PROVIDER} ({active_provider()['model']}) - {'ready' if ok else 'offline: ' + why}")
        elif cmd == "dm":
            blackmail = "--blackmail" in rest
            rest = rest.replace("--blackmail", "").strip()
            agent, msg = sim.split_agent_prefix(rest)
            if not agent or not msg:
                console.print("[red]Usage: /dm <Agent_Name> <Message> [--blackmail][/]")
            else:
                with console.status("Establishing secure channel..."):
                    r = await dm_agent(agent, msg, blackmail=blackmail, sim=sim)
                _print_dm(console, r)
        elif cmd == "summit":
            spec, msg = _parse_summit(rest)
            if not spec or not msg:
                console.print("[red]Usage: /summit <Bloc_or_Comma_List> <Message>[/]")
            else:
                with console.status("Convening summit..."):
                    r = await summit_agents(spec, msg, sim=sim)
                _print_summit(console, r)
        elif cmd == "consult":
            if not rest:
                console.print("[red]Usage: /consult <Draft_Move>[/]")
            else:
                with console.status("Advisors deliberating..."):
                    r = await consult_advisors(rest, sim=sim)
                _print_consult(console, r)
        elif cmd == "blackops":
            op, _, target = rest.partition(" ")
            if not op or not target.strip():
                console.print("[red]Usage: /blackops <Op_Type> <Target>[/]")
            else:
                _print_effects(console, "BLACK OP", await run_blackops(op, target.strip(), sim=sim))
        elif cmd == "marketwar":
            action, _, amt = rest.partition(" ")
            try:
                amount = float(amt) if amt.strip() else None
            except ValueError:
                amount = None
            _print_effects(console, "MARKET WAR", await run_marketwar(action, amount, sim=sim))
        elif cmd == "history":
            with console.status("Fetching historical market data..."):
                r = await load_history_baseline(rest, sim=sim)
            if not r["ok"]:
                console.print(f"[red]{r['error']}[/]")
            else:
                s = r["snapshot"]
                console.print(Panel(f"{r['date']}: S&P {s['sp500']:.0f}, WTI ${s['oil_price']:.1f}, 10Y {s['ten_year_yield']:.2f}% "
                                    f"({', '.join(sorted(set(s['sources'].values())))})\nRegime: {r['regime'] or 'none'}\n"
                                    f"{r['context']}\n[dim]{r['note']}[/]", title="HISTORICAL BASELINE", border_style="green"))
                console.print(build_dashboard(sim))
        elif cmd == "act":
            force = "--redphone" in rest
            move = rest.replace("--redphone", "").strip()
            if not move:
                console.print("[red]Usage: /act <Official_Presidential_Move>[/]")
            else:
                sim.progress = lambda msg: console.print(f"[cyan]{msg}[/]")

                async def handler(ev: Dict[str, Any]) -> str:
                    return await _cli_red_phone(console, ev)

                rep = await execute_turn(move, sim=sim, red_phone_handler=handler, force_red_phone=force)
                _print_report(console, rep)
                console.print(build_dashboard(sim))
        else:
            console.print(f"[red]Unknown command /{cmd}. Type /help.[/]")
    except Exception as exc:  # keep the REPL alive
        console.print(Panel(f"{type(exc).__name__}: {exc}", title="ERROR", border_style="red"))
    return True


async def cli_main() -> None:
    console = Console()
    sim = get_sim()
    console.print(Panel("[bold green]QUANT GEOPOLITICAL SIMULATOR - SITUATION ROOM[/]\nType /help for commands.",
                        border_style="green"))
    ok, why = llm_ready()
    cfg = PROVIDERS.get(LLM_PROVIDER, {})
    console.print(f"LLM provider: [bold]{LLM_PROVIDER}[/] ({cfg.get('model')}) - "
                  + ("[green]ready[/]" if ok else f"[yellow]OFFLINE heuristics ({why})[/]"))
    console.print(build_dashboard(sim))
    while True:
        try:
            line = (await asyncio.to_thread(console.input, "[bold green]POTUS> [/]")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line and not await handle_command(console, sim, line):
            break
    sim.save_all()
    console.print("[dim]State saved. Goodbye, Mr. President.[/]")


def main() -> None:
    asyncio.run(cli_main())


__all__ = ["Simulation", "get_sim", "reset_sim", "dm_agent", "summit_agents", "consult_advisors", "run_blackops",
           "run_marketwar", "load_history_baseline", "execute_turn", "export_portfolio_csv", "build_dashboard",
           "set_provider", "llm_ready", "run_sync", "TurnReport", "RefereeVerdict", "AgentDecision",
           "AdvisorOpinion", "AgentReply", "LLM_PROVIDER", "PROVIDERS"]

if __name__ == "__main__":
    main()
