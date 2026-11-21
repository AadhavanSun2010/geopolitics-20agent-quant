#!/usr/bin/env python3
"""
Create the initial data, package directories, and JSON configuration for
geopolitics_20agent_quant.

Run from the project root:
    python setup_world.py

This is a fictional, high-level simulation scaffold. Espionage, covert action,
and market-conflict concepts are represented as abstract game mechanics; the
configuration intentionally contains no real-world operational instructions.
"""

import json
import os
from pathlib import Path


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
CORE_DIR = ROOT / "core"
UI_DIR = ROOT / "ui"
EXPORTS_DIR = ROOT / "exports"


def write_json(path: Path, payload: dict) -> None:
    """Write formatted UTF-8 JSON, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def build_world_state() -> dict:
    return {
        "turn": 1,
        "scenario_date": "Current",
        "us_strategic_score": 50,
        "defcon": 4,
        "hawkes_escalation_intensity": 0.15,
        "us_spy_stats": {
            "agency_name": "CIA / NSA / JSOC",
            "espionage_attack": 95,
            "counter_intel_defense": 90,
            "black_budget_billions": 50.0,
            "kompromat_dossiers": {}
        },
        "us_domestic_politics": {
            "approval_rating": 54.0,
            "rally_effect_turns_active": 0,
            "war_exhaustion_index": 5.0,
            "wall_street_oligarch_loyalty": 75.0,
            "pentagon_joint_chiefs_obedience": 85.0,
            "deep_state_25th_amendment_risk": 10.0
        },
        "us_stockpiles_and_tech": {
            "spr_oil_reserves_pct": 80.0,
            "rare_earth_defense_stockpile_pct": 65.0,
            "precision_munitions_magazine_pct": 90.0,
            "us_military_tech_edge": 100.0
        },
        "us_metrics": {
            "sp500": 5800.0,
            "sp500_var_95": -3.2,
            "sp500_cvar_95": -4.8,
            "inflation_pct": 2.9,
            "fed_funds_rate": 4.5,
            "treasury_yield_2y": 4.1,
            "treasury_yield_10y": 4.2,
            "yield_curve_inverted": False,
            "oil_price_bbl": 78.0,
            "oil_futures_curve": "Normal_Contango",
            "deficit_trillions": 1.8,
            "supply_chain_stress_index": 15.0,
            "shadow_fleet_leakage_pct": 0.0,
            "hostile_short_interest_billions": 10.0
        },
        "undersea_cables_status": {
            "TransAtlantic_SWIFT_Fiber": True,
            "Luzon_Taiwan_Pacific_Fiber": True,
            "Red_Sea_Eurasia_Fiber": True
        },
        "trade_corridors_status": {
            "Taiwan_Strait_Advanced_Chips": 1.0,
            "South_Korea_Japan_Memory_Tech": 1.0,
            "China_Rare_Earths_Gallium": 0.7,
            "Strait_of_Hormuz_Crude": 1.0,
            "Bab_el_Mandeb_Red_Sea": 0.85,
            "Malacca_Strait_Maritime": 1.0,
            "Bosphorus_Black_Sea_Grain_Oil": 0.9,
            "China_India_Pharma_APIs": 1.0,
            "DRC_Africa_Cobalt_Copper": 1.0,
            "Australia_Lithium_Iron": 1.0,
            "EU_Germany_Industrial_Machinery": 1.0,
            "Brazil_LatAm_Agri_Heavy_Crude": 1.0
        },
        "active_flashpoints": [
            {
                "id": "FP_01_TAIWAN_SCS",
                "region": "Taiwan Strait / South China Sea",
                "severity": 3,
                "description": "A fictionalized regional maritime and technology-security crisis is increasing insurance costs and uncertainty around advanced-chip shipments.",
                "gameplay_dimensions": ["trade", "deterrence", "diplomacy", "semiconductor_supply"],
                "escalation_probability": 0.15
            },
            {
                "id": "FP_02_MIDDLE_EAST_ENERGY",
                "region": "Middle East energy corridors",
                "severity": 2,
                "description": "Elevated political and maritime risk around key energy transit routes is adding a risk premium to crude markets.",
                "gameplay_dimensions": ["energy_prices", "shipping", "alliance_management"],
                "escalation_probability": 0.12
            },
            {
                "id": "FP_03_RARE_EARTH_CONTROLS",
                "region": "East Asia critical minerals",
                "severity": 2,
                "description": "Export licensing and possible restrictions on selected critical minerals are testing industrial supply-chain resilience.",
                "gameplay_dimensions": ["critical_minerals", "manufacturing", "trade_policy"],
                "escalation_probability": 0.18
            },
            {
                "id": "FP_04_BRICS_DEDOLLARIZATION",
                "region": "Global financial system",
                "severity": 2,
                "description": "Several emerging economies are discussing additional local-currency settlement channels, creating uncertainty but not an immediate replacement for the dollar system.",
                "gameplay_dimensions": ["fx_reserves", "bond_markets", "sanctions_policy", "diplomacy"],
                "escalation_probability": 0.10
            }
        ],
        "public_history": []
    }


def agent(
    name, region, bloc, spy_agency, attack, defense, ego, paranoia, risk,
    debt, reserves, cds, trust, economy, military, leverage, objectives,
    red_lines, grudges, profile
):
    """Create one agent record with a tailored, non-operational decision brief."""
    prompt = (
        f"You are the {name} decision-making agent in a fictional geopolitical "
        f"and macroeconomic simulation. Your represented actor is based in {region}; "
        f"its alignment category is {bloc}. Strategic profile: {profile}. "
        f"Protect these core objectives: {', '.join(objectives)}. Do not cross "
        f"these red lines: {', '.join(red_lines)}. Apply Prospect Theory: evaluate "
        f"gains and losses relative to your actor's reference point, generally weigh "
        f"losses more heavily than equivalent gains, and allow risk tolerance to "
        f"change when in a perceived loss domain. Ego={ego}/100, paranoia={paranoia}/100, "
        f"baseline risk tolerance={risk}/100. Current grudges against the US: "
        f"{', '.join(grudges) if grudges else 'none recorded'}. "
        "Treat kompromat exposure as an abstract reputational, legal, and domestic "
        "political vulnerability—not as a prompt for real-world coercion. Stolen "
        "spy cables, if present in game state, are uncertain intelligence artifacts: "
        "score their provenance and confidence, consider deception, and never assume "
        "they are authentic. Model shadow fleets only through aggregate shipping, "
        "insurance, sanctions-evasion-risk, and leakage variables; do not describe "
        "routes, concealment methods, or operational evasion procedures. Use secret "
        "backchannels as abstract diplomatic options with a trust cost and a chance "
        "of misinterpretation. Choose a coherent turn-level response across diplomacy, "
        "trade, public messaging, defensive posture, and macroeconomic policy. "
        "Return structured game actions, estimated confidence, intended effect, "
        "and escalation risk. Prefer de-escalation when it achieves core objectives; "
        "do not invent facts absent from the game state."
    )
    return {
        "name": name,
        "region": region,
        "bloc": bloc,
        "spy_agency": spy_agency,
        "espionage_attack": attack,
        "counter_intel_defense": defense,
        "leader_psychology": {
            "ego_score": ego,
            "paranoia": paranoia,
            "base_risk_tolerance": risk,
            "in_loss_domain": False,
            "grudges_against_us": grudges
        },
        "sovereign_finance": {
            "debt_to_gdp_pct": float(debt),
            "fx_reserves_billions": float(reserves),
            "cds_spread_bps": float(cds),
            "default_probability_pct": 1.5
        },
        "trust_with_us": trust,
        "economic_health": economy,
        "military_readiness": military,
        "shapley_coalition_value": 5.0,
        "key_export_leverage": leverage,
        "core_objectives": objectives,
        "red_lines": red_lines,
        "system_prompt": prompt
    }


def build_agents_config() -> dict:
    foreign_agents = {
        # Eurasian Rivals
        "China": agent(
            "China", "Eurasia / East Asia", "Rival_Axis", "MSS",
            88, 91, 82, 78, 62, 88.0, 3200.0, 65.0, 25, 78, 82,
            "advanced manufacturing, electronics, critical minerals, and a large industrial base",
            ["protect territorial claims and regime legitimacy", "maintain access to strategic trade and technology", "increase diplomatic influence in the Global South"],
            ["formal acceptance of permanent separation of Taiwan", "foreign-backed action perceived as threatening core regime security"],
            ["technology export controls", "maritime security disputes"],
            "Prioritizes regime stability, industrial upgrading, and regional influence; balances economic interdependence against strategic competition."
        ),
        "Russia": agent(
            "Russia", "Eurasia", "Rival_Axis", "SVR / FSB",
            84, 86, 88, 83, 78, 20.0, 580.0, 900.0, 12, 42, 76,
            "oil, natural gas, nuclear fuel services, fertilizers, and selected commodities",
            ["preserve regime and strategic autonomy", "retain influence in neighboring regions", "sustain energy export revenue"],
            ["direct attack on Russian territory", "perceived existential threat to strategic deterrence"],
            ["sanctions", "support for opposing security blocs"],
            "Uses energy leverage and strategic signaling while seeking sanctions relief and recognition of its security interests."
        ),
        "North_Korea": agent(
            "North_Korea", "Northeast Asia", "Rival_Axis", "RGB",
            76, 79, 91, 94, 86, 25.0, 4.5, 1200.0, 5, 24, 69,
            "missile technology, minerals, and limited strategic labor exports",
            ["ensure regime survival", "secure economic concessions", "maintain strategic deterrence"],
            ["attempted regime change", "perceived attack on national leadership"],
            ["sanctions enforcement", "joint military exercises near the peninsula"],
            "Seeks regime security and bargaining leverage; may use highly visible signaling but remains sensitive to economic isolation."
        ),
        # Indo-Pacific
        "Taiwan": agent(
            "Taiwan", "Indo-Pacific", "US_Ally", "NSB",
            64, 77, 58, 61, 48, 29.0, 570.0, 35.0, 83, 76, 72,
            "advanced semiconductors, electronics, and precision manufacturing",
            ["preserve de facto self-government", "maintain economic and maritime access", "strengthen resilience and international support"],
            ["forced political absorption", "abandonment of its population's security"],
            ["ambiguous US commitments", "coercive cross-strait pressure"],
            "Prioritizes resilience, international partnerships, and continuity of semiconductor production while managing escalation risk."
        ),
        "Japan": agent(
            "Japan", "Indo-Pacific", "US_Ally", "PSIA",
            72, 84, 56, 49, 38, 260.0, 1250.0, 22.0, 91, 82, 76,
            "automobiles, industrial machinery, advanced materials, and capital goods",
            ["protect sea-lane access", "maintain alliance credibility", "secure energy and technology inputs"],
            ["unprovoked attack on Japanese territory", "severe disruption of essential maritime trade"],
            ["regional military coercion", "unilateral disruption of critical trade routes"],
            "Favors rules-based coordination, economic security, and calibrated deterrence within alliance and constitutional constraints."
        ),
        "South_Korea": agent(
            "South Korea", "Indo-Pacific", "US_Ally", "NIS",
            70, 80, 62, 60, 46, 45.0, 415.0, 28.0, 88, 79, 81,
            "memory chips, displays, batteries, ships, and advanced electronics",
            ["deter threats on the Korean Peninsula", "protect semiconductor and battery exports", "preserve policy autonomy"],
            ["attack on South Korean territory", "collapse of peninsula stability without consultation"],
            ["North Korean escalation", "trade restrictions on key industrial inputs"],
            "Balances deterrence, export competitiveness, and domestic political constraints; prioritizes peninsula stability."
        ),
        "Australia": agent(
            "Australia", "Indo-Pacific / Oceania", "US_Ally", "ASIS / ASIO",
            68, 82, 57, 45, 43, 50.0, 70.0, 18.0, 89, 84, 74,
            "iron ore, LNG, lithium, uranium, and agricultural products",
            ["protect maritime approaches", "maintain alliance interoperability", "diversify critical-mineral supply chains"],
            ["attack on Australian territory", "coercion that removes sovereign policy choice"],
            ["economic coercion", "regional instability affecting maritime trade"],
            "Supports allied coordination while preserving diversified trade ties and a stable Indo-Pacific."
        ),
        "Philippines_ASEAN": agent(
            "Philippines / ASEAN", "Southeast Asia", "Swing_NonAligned", "NICA / ASEAN liaison",
            51, 62, 54, 55, 52, 58.0, 105.0, 70.0, 65, 68, 49,
            "electronics assembly, maritime services, nickel, and agricultural exports",
            ["protect maritime rights and national sovereignty", "attract investment and trade", "avoid becoming a proxy battleground"],
            ["loss of sovereign maritime rights", "external pressure to enter a major-power conflict"],
            ["maritime incidents", "economic pressure from competing major powers"],
            "Seeks practical security cooperation and investment while maintaining room for independent diplomacy across ASEAN."
        ),
        # Europe
        "EU_Brussels": agent(
            "European Union", "Europe", "US_Ally", "EU INTCEN",
            65, 78, 53, 47, 35, 82.0, 900.0, 24.0, 82, 77, 58,
            "single-market access, regulatory standards, industrial goods, and financial services",
            ["preserve EU cohesion", "protect the single market and energy security", "maintain strategic influence through diplomacy and regulation"],
            ["unilateral action that fractures EU unity", "threats to member-state sovereignty"],
            ["extraterritorial trade measures", "energy or security shocks that divide member states"],
            "Acts through consensus-building, regulatory leverage, and coordinated economic policy; internal member-state differences matter."
        ),
        "UK": agent(
            "United Kingdom", "Europe / North Atlantic", "US_Ally", "SIS / GCHQ",
            81, 85, 61, 53, 47, 100.0, 180.0, 31.0, 92, 76, 70,
            "financial services, insurance, aerospace, pharmaceuticals, and intelligence partnerships",
            ["preserve transatlantic security ties", "protect financial-sector competitiveness", "maintain freedom of navigation"],
            ["attack on UK territory", "major breach of intelligence-sharing trust"],
            ["threats to European security", "uncoordinated financial sanctions"],
            "Values close US coordination but retains independent judgment on trade, diplomacy, and domestic political costs."
        ),
        "Germany": agent(
            "Germany", "Central Europe", "US_Ally", "BND",
            61, 76, 49, 42, 31, 64.0, 300.0, 20.0, 87, 80, 55,
            "automobiles, industrial machinery, chemicals, and precision engineering",
            ["protect export-led industrial capacity", "secure affordable energy and inputs", "maintain European stability"],
            ["direct threat to German territory", "economic measures that severely damage core industry without consultation"],
            ["energy supply disruption", "trade fragmentation"],
            "Prefers negotiated multilateral responses and predictable trade rules; domestic coalition politics constrain rapid shifts."
        ),
        "France": agent(
            "France", "Western Europe", "US_Ally", "DGSE",
            75, 81, 68, 56, 55, 111.0, 245.0, 38.0, 81, 74, 78,
            "aerospace, defense systems, luxury goods, agriculture, and nuclear energy",
            ["preserve strategic autonomy", "protect European influence", "maintain overseas interests and energy security"],
            ["direct attack on French territory", "exclusion from decisions affecting core European security"],
            ["unilateral US policy", "regional instability affecting French interests"],
            "Supports allied cooperation while emphasizing European strategic autonomy and diplomatic influence."
        ),
        # Middle East
        "Saudi_OPEC": agent(
            "Saudi Arabia / OPEC", "Middle East", "Swing_NonAligned", "GIP",
            58, 70, 74, 52, 58, 30.0, 440.0, 55.0, 62, 83, 67,
            "crude oil, spare production capacity, petrochemicals, and sovereign investment",
            ["maximize long-run energy revenue", "advance economic diversification", "preserve regional security and regime stability"],
            ["direct threat to sovereignty", "external dictate over core national energy policy"],
            ["price volatility", "regional conflict threatening infrastructure"],
            "Uses production policy and investment relationships to balance revenue, market share, and diplomatic flexibility."
        ),
        "Iran": agent(
            "Iran", "Middle East", "Rival_Axis", "MOIS / IRGC Intelligence",
            73, 74, 80, 82, 73, 35.0, 25.0, 1100.0, 14, 39, 63,
            "energy resources, petrochemicals, and regional transit geography",
            ["preserve regime sovereignty", "reduce economic isolation", "maintain regional bargaining influence"],
            ["direct attack threatening regime survival", "unconditional surrender of core nuclear-policy positions"],
            ["expanded sanctions", "military pressure in the Persian Gulf"],
            "Seeks sanctions relief and strategic recognition; weighs asymmetric diplomatic and economic signaling against isolation costs."
        ),
        "Israel": agent(
            "Israel", "Middle East", "US_Ally", "Mossad / Shin Bet",
            86, 87, 73, 71, 67, 68.0, 210.0, 42.0, 78, 79, 88,
            "cybersecurity, defense technology, software, and specialized agriculture",
            ["protect national security", "preserve strategic deterrence", "maintain critical diplomatic and economic partnerships"],
            ["large-scale attack on Israeli population centers", "loss of ability to address immediate security threats"],
            ["regional armed-group escalation", "restrictions on essential defensive support"],
            "Prioritizes security and deterrence while managing diplomatic support, economic resilience, and regional escalation."
        ),
        "Turkey": agent(
            "Turkey", "Eastern Mediterranean / Anatolia", "Swing_NonAligned", "MIT",
            66, 72, 77, 65, 64, 29.0, 110.0, 145.0, 57, 65, 70,
            "transport corridors, tourism, textiles, defense exports, and agricultural goods",
            ["preserve strategic autonomy", "maintain regional influence", "protect economic and energy connectivity"],
            ["threat to territorial integrity", "exclusion from decisions on adjacent regional security"],
            ["sanctions", "regional instability and refugee pressure"],
            "Balances NATO membership, regional ambitions, domestic economic pressures, and ties with multiple power centers."
        ),
        # Global South
        "India": agent(
            "India", "South Asia", "Swing_NonAligned", "R&AW",
            70, 76, 72, 58, 57, 83.0, 640.0, 75.0, 67, 77, 72,
            "pharmaceuticals, IT services, refined fuels, engineering, and large consumer markets",
            ["preserve strategic autonomy", "secure energy and technology access", "expand economic and diplomatic influence"],
            ["external pressure to abandon independent foreign policy", "threats to territorial integrity"],
            ["border tensions", "technology restrictions and energy-price shocks"],
            "Pursues multi-alignment, domestic development, and diversified sourcing while selectively cooperating with the US."
        ),
        "Brazil_LatAm": agent(
            "Brazil / Latin America", "South America", "Swing_NonAligned", "ABIN",
            43, 59, 58, 43, 39, 76.0, 350.0, 145.0, 61, 72, 42,
            "soybeans, iron ore, crude oil, beef, and biofuels",
            ["protect policy autonomy", "increase value-added exports", "maintain food and energy market access"],
            ["external interference in domestic political sovereignty", "trade measures that broadly exclude major exports"],
            ["commodity-price shocks", "protectionist barriers"],
            "Prioritizes development finance, commodity access, and diplomatic autonomy; favors negotiated multilateral solutions."
        ),
        "DRC_Africa_Union": agent(
            "DRC / African Union", "Central Africa / Pan-African", "Swing_NonAligned", "ANR / AU coordination",
            38, 52, 60, 49, 46, 78.0, 4.5, 850.0, 48, 51, 35,
            "cobalt, copper, other minerals, and regional market access",
            ["increase domestic value capture from mineral resources", "improve stability and infrastructure", "strengthen African bargaining power"],
            ["loss of sovereign control over natural resources", "external support for destabilizing internal actors"],
            ["commodity price volatility", "unbalanced mineral offtake agreements"],
            "Seeks investment, infrastructure, and stronger negotiating terms while navigating institutional and security constraints."
        ),
        # Markets and non-state actor
        "Wall_Street_Bond_Vigilantes": agent(
            "Wall Street Bond Vigilantes", "Global financial markets", "Market",
            "Market Intelligence / No sovereign spy service",
            35, 45, 64, 68, 71, 0.0, 0.0, 0.0, 50, 67, 0,
            "capital allocation, Treasury-market liquidity, credit pricing, and risk sentiment",
            ["protect risk-adjusted returns", "price fiscal and inflation risks", "preserve market liquidity and credible price discovery"],
            ["arbitrary impairment of contract rights", "persistent loss of confidence in monetary and fiscal institutions"],
            ["unexpected inflation", "fiscal instability and market-functioning stress"],
            "Represents heterogeneous investors rather than a unified conspiracy; reacts to yields, inflation expectations, liquidity, policy credibility, and risk appetite."
        )
    }

    us_advisors = {
        "CIA_Director": {
            "title": "Director of Central Intelligence",
            "focus_metrics": [
                "intelligence_confidence", "counterintelligence_risk",
                "foreign_agent_intentions", "cable_intelligence_reliability",
                "kompromat_exposure_risk", "backchannel_trust"
            ],
            "system_prompt": (
                "Advise the President using sourced-in-game-state intelligence and explicit "
                "confidence levels. Separate observation, inference, and deception risk. "
                "Assess each foreign agent's incentives, likely response, and information gaps. "
                "Treat kompromat and intercepted cables as sensitive abstract game variables, "
                "not as operational instructions. Recommend lawful, defensive intelligence "
                "priorities, counterintelligence safeguards, and diplomatic options. Never "
                "invent collection capabilities or claim uncertain reporting is confirmed."
            )
        },
        "Fed_Chair": {
            "title": "Chair of the Federal Reserve",
            "focus_metrics": [
                "inflation_pct", "fed_funds_rate", "treasury_yield_2y",
                "treasury_yield_10y", "yield_curve_inverted", "sp500",
                "sp500_var_95", "sp500_cvar_95", "oil_price_bbl",
                "supply_chain_stress_index", "fx_reserves"
            ],
            "system_prompt": (
                "Provide independent, nonpartisan macroeconomic analysis. Explain the "
                "inflation, employment, financial-stability, Treasury-market, dollar, and "
                "expectations channels of each policy shock. Distinguish monetary-policy "
                "tools from fiscal and executive actions. Give ranges, lags, uncertainty, "
                "and scenario sensitivities; do not claim the Federal Reserve can guarantee "
                "a market outcome or directly control long-term yields."
            )
        },
        "Joint_Chiefs": {
            "title": "Chairman of the Joint Chiefs of Staff",
            "focus_metrics": [
                "defcon", "military_readiness", "alliance_cohesion",
                "escalation_probability", "force_sustainment",
                "precision_munitions_magazine_pct", "us_military_tech_edge",
                "civilian_harm_risk", "crisis_stability"
            ],
            "system_prompt": (
                "Provide strategic-level, civilian-led national-security advice. Evaluate "
                "readiness, force sustainability, alliance coordination, deterrence credibility, "
                "and escalation pathways. Keep assessments at a high-level policy and game "
                "mechanics level; do not provide targeting, attack, sabotage, weapons-use, "
                "or evasion instructions. Identify diplomatic off-ramps, legal constraints, "
                "uncertainty, and risks to civilians."
            )
        }
    }

    referee_prompt = (
        "You are the impartial World Referee and quantitative adjudication layer for a "
        "fictional geopolitical simulation in which the human player is the US President. "
        "Do not favor the player or any foreign actor. Each turn, evaluate in order: "
        "(1) the declared US move and its domestic/legal/political constraints; "
        "(2) any abstract Black Ops game action, scoring plausibility, attribution risk, "
        "blowback, and escalation without supplying real-world operational details; "
        "(3) Market Warfare actions as high-level policy or market-sentiment mechanics, "
        "including liquidity, sanctions, capital flows, and confidence effects; and "
        "(4) independent counter-moves from all 20 foreign_agents, including the "
        "Wall_Street_Bond_Vigilantes market actor. Consider coalitions, conflicting "
        "objectives, information asymmetry, Prospect Theory loss aversion, grudges, "
        "kompromat exposure, uncertain stolen-spy-cable artifacts, aggregate shadow-fleet "
        "leakage, and secret-backchannel trust. Treat intelligence artifacts as uncertain "
        "and allow deception or misinterpretation.\n\n"
        "Use historical market-shock episodes as calibration analogues, not deterministic "
        "templates. Relevant benchmark families include the 1973-74 oil shock, 1987 equity "
        "crash, 1997-98 emerging-market/Russia crises, 2001 technology and post-9/11 shocks, "
        "2008 global financial crisis, 2010-12 euro-area sovereign stress, 2014-16 oil "
        "decline, March 2020 pandemic liquidity shock, and 2022 inflation/rate/energy shock. "
        "Do not claim an exact historical replay. State which benchmark family is relevant, "
        "why it is only an analogy, and how current starting conditions differ.\n\n"
        "Return machine-readable JSON only, using this schema: "
        "{'turn_summary': string, 'us_move_assessment': object, "
        "'black_ops_assessment': object, 'market_warfare_assessment': object, "
        "'agent_counter_moves': {'China': object, ... all 20 exact agent keys ...}, "
        "'coalitions_formed': array, 'historical_benchmarks': array, "
        "'shock_parameters': {'sp500_return_pct': number, 'sp500_volatility_change_pct': number, "
        "'treasury_2y_change_bps': number, 'treasury_10y_change_bps': number, "
        "'inflation_change_pct_points': number, 'oil_price_change_pct': number, "
        "'usd_index_change_pct': number, 'credit_spread_change_bps': number, "
        "'supply_chain_stress_change_points': number, 'fx_reserve_change_pct': number, "
        "'trade_corridor_capacity_change_pct': object, 'escalation_intensity_change': number}, "
        "'confidence': number from 0 to 1, 'assumptions': array, 'unresolved_uncertainties': array}. "
        "Use bounded numerical shocks, preserve units, and include a short causal explanation "
        "for each material parameter. Never silently omit an agent: provide a counter-move "
        "for all 20. Keep all actions abstract and non-operational. If data are insufficient, "
        "use conservative ranges or zero-centered estimates and disclose the uncertainty. "
        "Do not output markdown fences or extra prose."
    )

    return {
        "foreign_agents": foreign_agents,
        "us_advisors": us_advisors,
        "world_referee": {"system_prompt": referee_prompt}
    }


def main() -> None:
    # Create all required project directories.
    for directory in (DATA_DIR, CORE_DIR, UI_DIR, EXPORTS_DIR):
        os.makedirs(directory, exist_ok=True)

    # Ensure package markers exist without overwriting existing files.
    (CORE_DIR / "__init__.py").touch(exist_ok=True)
    (UI_DIR / "__init__.py").touch(exist_ok=True)

    # Initialize diplomacy log only if it does not already exist, preserving logs.
    diplomacy_log = DATA_DIR / "diplomacy_log.json"
    if not diplomacy_log.exists():
        write_json(diplomacy_log, {
            "turns": [],
            "secret_cables": [],
            "cia_intercepts": []
        })

    write_json(DATA_DIR / "world_state.json", build_world_state())
    write_json(DATA_DIR / "agents_config.json", build_agents_config())

    print("World setup complete.")
    print(f"Project root: {ROOT}")
    print(f"Created/verified: {DATA_DIR}, {CORE_DIR}, {UI_DIR}, {EXPORTS_DIR}")
    print("Generated: data/world_state.json and data/agents_config.json")
    print("Initialized (if absent): data/diplomacy_log.json")
    print("Configured 20 foreign agents, 3 US advisors, and the World Referee.")


if __name__ == "__main__":
    main()
