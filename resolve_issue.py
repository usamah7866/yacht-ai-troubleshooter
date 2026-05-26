from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from pdf_ai_classifier import ASSISTANT_PATH, CATEGORIES, tokenize


QUERY_EXPANSIONS = {
    "air_conditioning_low_pressure": [
        "air conditioning",
        "chiller",
        "fresh water flow alarm",
        "flow sensor",
        "fresh water circulation",
        "refrigerant low pressure",
        "sea water circulation",
        "strainer",
        "water pump",
        "bleeding",
        "air lock",
        "filter",
        "valve",
        "pressure",
        "CU50VFD",
        "CU70VFD",
    ],
    "generator_no_start": [
        "generator",
        "starter motor",
        "main engine does not start",
        "battery",
        "fuel pump",
        "fuel filter",
        "air bubbles",
        "glow plugs",
        "pressure switch",
    ],
    "generator_no_voltage": [
        "generator",
        "does not supply voltage",
        "circuit breaker",
        "AVR",
        "alternator",
        "low voltage",
        "high voltage",
        "unstable voltage",
    ],
    "fuel_filter_bleeding": [
        "fuel",
        "fuel filter",
        "fuel pump",
        "air bubbles",
        "bleeding",
        "deaeration",
        "diesel",
        "tank",
    ],
}


def route_category(text: str) -> str:
    lowered = text.lower()
    best_category = "general_information"
    best_score = 0.0
    for category, keywords in CATEGORIES.items():
        score = 0.0
        for keyword in keywords:
            if keyword.lower() in lowered:
                score += 2.5 if " " in keyword else 1.0
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def detect_issue_type(query: str) -> str:
    q = query.lower()
    ac_terms = ["air condition", "air-conditioning", "ac ", "chiller", "fan coil", "fancoil", "frigomar"]
    pressure_terms = ["low pressure", "pressure low", "flow alarm", "fresh water", "water flow", "not cooling", "cooling problem"]
    if any(term in q for term in ac_terms) and any(term in q for term in pressure_terms):
        return "air_conditioning_low_pressure"
    if any(term in q for term in ["generator", "genset", "mase"]) and any(term in q for term in ["not start", "does not start", "won't start", "starter"]):
        return "generator_no_start"
    if any(term in q for term in ["generator", "genset", "mase"]) and any(term in q for term in ["no voltage", "low voltage", "high voltage", "unstable voltage", "does not supply voltage"]):
        return "generator_no_voltage"
    if "fuel" in q and any(term in q for term in ["filter", "bleed", "air", "pump"]):
        return "fuel_filter_bleeding"
    return "general"


def expand_query(query: str, issue_type: str) -> str:
    return query + " " + " ".join(QUERY_EXPANSIONS.get(issue_type, []))


def bm25_search(index: dict, query: str, limit: int = 12, preferred_categories: set[str] | None = None) -> list[dict]:
    query_tokens = sorted(set(tokenize(query)))
    total_docs = index["total_chunks"]
    avg_len = index["average_length"] or 1
    routed = route_category(query)
    k1 = 1.5
    b = 0.75
    results = []

    for chunk in index["chunks"]:
        counts = Counter(chunk["tokens"])
        score = 0.0
        for token in query_tokens:
            tf = counts[token]
            if not tf:
                continue
            df = index["document_frequency"].get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * chunk["length"] / avg_len)))
        if chunk["category"] == routed:
            score *= 1.18
        if preferred_categories and chunk["category"] in preferred_categories:
            score *= 1.25
        if score > 0:
            result = dict(chunk)
            result["score"] = round(score, 4)
            results.append(result)

    return sorted(results, key=lambda row: row["score"], reverse=True)[:limit]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.strip()) > 35]


def pick_sentences(results: list[dict], keywords: list[str], limit: int) -> list[str]:
    selected = []
    seen = set()
    for result in results:
        for sentence in split_sentences(result["text"]):
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords) and sentence not in seen:
                selected.append(sentence)
                seen.add(sentence)
            if len(selected) >= limit:
                return selected
    return selected


def source_summary(results: list[dict], limit: int = 6) -> list[dict]:
    compact = []
    seen = set()
    for row in results:
        key = (row["pdf"], row["page"])
        if key in seen:
            continue
        seen.add(key)
        compact.append(
            {
                "pdf": row["pdf"],
                "page": row["page"],
                "category": row["category"],
                "score": row["score"],
                "excerpt": row["text"][:700],
            }
        )
        if len(compact) >= limit:
            break
    return compact


def build_ac_low_pressure_answer(results: list[dict]) -> dict:
    return {
        "answer_summary": (
            "The manuals point first to a circulation problem, not immediately to replacing parts. "
            "For a Frigomar yacht AC/chiller low-pressure or fresh-water-flow alarm, check whether the fresh-water circuit is actually circulating, whether air is trapped, whether filters/strainers are blocked, whether valves are open/balanced, and whether the pump/seawater side is working correctly."
        ),
        "possible_reasons": [
            {
                "reason": "No or insufficient fresh-water circulation",
                "detail": "The CU50VFD/CU70VFD manual says the flow sensor status can be checked from the display: status 0 means water circulation is present, status 1 means no circulation.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD manual, page 11",
            },
            {
                "reason": "Air trapped in the fresh-water circuit",
                "detail": "After filling or service, air in the circuit can stop stable circulation. The manual describes restarting the fresh-water pump if the bleeding phase takes longer than 10 minutes.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD manual, page 11",
            },
            {
                "reason": "Dirty filter/strainer or dirty fresh-water circuit",
                "detail": "A dirty filter reduces flow. The manual recommends repeating the cleaning operation until the filter remains clean.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD manual, pages 9-11",
            },
            {
                "reason": "Closed valve, wrong manifold balance, or blocked fan-coil circuit",
                "detail": "When multiple units are connected, the manual requires a manifold with balancing valves to assure correct flow to each unit. Fan-coil water connection/flow problems can reduce performance and create air lock.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD page 9 and AH Series Fan Coil page 17",
            },
            {
                "reason": "Pump issue or poor seawater circulation",
                "detail": "The manual warns that poor water circulation can damage the compressor. It also states the seawater pump must be correctly sized and that strainers can clog from jellyfish, seaweed, sand, or debris.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD manual, page 10",
            },
            {
                "reason": "Water leak or incorrect circuit pressure",
                "detail": "The manual says to check there is no water loss along the fresh-water circuit and notes that fresh-water pressure must not be higher than 1.5 bar with all valves open and the pump switched off.",
                "evidence": "FRIGOMAR CU50VFD/CU70VFD manual, page 11",
            },
        ],
        "diagnostic_steps": [
            {
                "step": "Do not keep forcing the compressor to run while the alarm is active.",
                "why": "The manuals warn that poor water circulation can damage the compressor or create unsafe conditions.",
            },
            {
                "step": "Open the chiller display running-status page and check the fresh-water flow sensor status.",
                "why": "Status 0 means circulation is present. Status 1 means no circulation, so the issue is in flow, pump, valve, air, filter, or sensor wiring.",
            },
            {
                "step": "Verify the fresh-water pump is running and the circuit is fully bled.",
                "why": "Air trapped in the circuit can create low/unstable flow and trigger the alarm.",
            },
            {
                "step": "Inspect and clean the fresh-water filter/strainer, then repeat the check after a few hours if it becomes dirty again.",
                "why": "The manual specifically says to repeat cleaning until the filter remains clean.",
            },
            {
                "step": "Check all service valves, manifold balancing valves, and fan-coil valves.",
                "why": "A closed or unbalanced valve can starve one unit and create low flow.",
            },
            {
                "step": "Check for water leaks and confirm fresh-water pressure with the pump off and valves open.",
                "why": "A leak or incorrect static pressure can prevent stable circulation.",
            },
            {
                "step": "Check seawater-side flow separately: seacock open, seawater pump working, seawater strainer clean, and no blocked condenser/heat exchanger.",
                "why": "Poor seawater flow can cause abnormal pressure behavior and compressor protection alarms.",
            },
            {
                "step": "If flow sensor status, pump, filters, bleeding, valves, and seawater flow all check out but the alarm remains, involve qualified Frigomar service.",
                "why": "At that point the issue may be a sensor, wiring, refrigerant pressure, compressor driver, or control-board problem.",
            },
        ],
        "safety_notes": [
            "Electrical, refrigerant, pressure, and onboard seawater-system work should be handled by qualified personnel.",
            "Do not bypass safety alarms or protections to keep the unit running.",
        ],
        "manual_sources": source_summary(results),
    }


def build_generator_no_start_answer(results: list[dict]) -> dict:
    return {
        "answer_summary": "The VS 22 LOW manual troubleshooting table points first to fuel supply, air in the fuel circuit, filters, battery/cables, DC thermal switches, and starting/preheating components.",
        "possible_reasons": [
            {"reason": "Fuel loading not done correctly", "detail": "Check fuel level and the condition of the fuel supply lines.", "evidence": "Mase VS 22 LOW troubleshooting section"},
            {"reason": "Air bubbles inside the fuel circuit", "detail": "Air in the fuel circuit can prevent regular operation or starting.", "evidence": "Mase VS 22 LOW maintenance/troubleshooting"},
            {"reason": "Dirty fuel filters or failed fuel pump", "detail": "The manual lists dirty filters and pump failure as starting/operation causes.", "evidence": "Mase VS 22 LOW troubleshooting section"},
            {"reason": "Battery, cables, or DC thermal switches", "detail": "If control logic powers up but starter does not turn, check battery state, terminals, cables, and DC thermal switches.", "evidence": "Mase VS 22 LOW troubleshooting section"},
        ],
        "diagnostic_steps": [
            {"step": "Confirm fuel level and inspect fuel supply/return lines.", "why": "The manual lists incorrect fuel loading as a start failure cause."},
            {"step": "Check and replace dirty fuel filters if required.", "why": "Clogged filters restrict diesel supply."},
            {"step": "Bleed/deaerate the fuel circuit if air bubbles are present.", "why": "Air in the circuit can stop the engine from starting or reaching rated rpm."},
            {"step": "Check battery charge, terminals, cable condition, and DC thermal switches.", "why": "Poor electrical supply can prevent cranking."},
            {"step": "If the starter turns but the engine does not start after these checks, contact authorized service.", "why": "The remaining causes may require engine/manufacturer service."},
        ],
        "safety_notes": ["Close the seawater inlet during repeated failed starts if instructed by the manual to avoid water accumulation in the exhaust system."],
        "manual_sources": source_summary(results),
    }


def build_generic_answer(category: str, results: list[dict]) -> dict:
    possible = pick_sentences(
        results,
        ["cause", "alarm", "poor", "dirty", "clog", "air", "pressure", "flow", "pump", "filter", "strainer", "valve", "battery", "fuel", "voltage"],
        7,
    )
    actions = pick_sentences(
        results,
        ["check", "replace", "clean", "bleed", "open", "reset", "inspect", "close", "contact", "run", "drain", "verify"],
        8,
    )
    return {
        "answer_summary": f"The issue is closest to '{category}'. I found matching manual passages and extracted the most relevant checks below.",
        "possible_reasons": [
            {"reason": sentence[:90], "detail": sentence, "evidence": "See manual sources below"}
            for sentence in (possible or ["The exact cause is not certain from the query. Use the source passages below to narrow it down."])
        ],
        "diagnostic_steps": [
            {"step": sentence, "why": "This step comes from or is directly supported by the matched manual passages."}
            for sentence in (actions or ["Review the matched manual source pages, then perform subsystem checks in the safest order."])
        ],
        "safety_notes": ["Use qualified personnel for electrical, refrigerant, engine, pressure, or seawater-system work."],
        "manual_sources": source_summary(results),
    }


def resolve(query: str) -> dict:
    index = json.loads(Path(ASSISTANT_PATH).read_text(encoding="utf-8"))
    issue_type = detect_issue_type(query)
    category = route_category(query)
    preferred = {"cooling_water", "alarms_controls", "maintenance_service"} if issue_type == "air_conditioning_low_pressure" else None
    if issue_type in {"generator_no_start", "fuel_filter_bleeding"}:
        preferred = {"fuel_system", "troubleshooting_repair", "maintenance_service", "operation"}
    if issue_type == "generator_no_voltage":
        preferred = {"electrical_wiring", "troubleshooting_repair", "technical_specs"}
    results = bm25_search(index, expand_query(query, issue_type), preferred_categories=preferred)

    if issue_type == "air_conditioning_low_pressure":
        answer = build_ac_low_pressure_answer(results)
    elif issue_type in {"generator_no_start", "fuel_filter_bleeding"}:
        answer = build_generator_no_start_answer(results)
    else:
        answer = build_generic_answer(category, results)

    answer.update({
        "query": query,
        "issue_type": issue_type,
        "detected_area": category,
        "used_optional_gpt": False,
        "optional_gpt_note": (
            "A GPT answer generator can be added on top of this retrieval result if an API key is provided. "
            "The current version stays local and source-grounded."
        ),
    })
    return answer


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Describe the yacht issue: ").strip()
    print(json.dumps(resolve(query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
