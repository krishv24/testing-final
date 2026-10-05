"""
Batch Experiment Runner for IEEE Journal Data Collection.
=========================================================

Runs the Hierarchical AI Meteorologist pipeline across diverse Indian
locations and parameter combinations, recording full results to Supabase.

Usage:
    python batch_runner.py                  # Run all uncompleted experiments
    python batch_runner.py --dry-run        # Test DB connection, no API calls
    python batch_runner.py --limit 10       # Run at most 10 experiments
    python batch_runner.py --delay 8        # 8 seconds between runs (default 5)

Environment variables (in .env):
    GEMINI_API_KEY      Your Gemini API key
    GEONAMES_USERNAME   Your GeoNames username
    SUPABASE_URL        Supabase project URL
    SUPABASE_KEY        Supabase publishable/anon key
    RUNNER_NAME         Your name (e.g. "krish", "amit")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ──
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

import httpx
from supabase import create_client, Client

from app.assistant.context_builder import run_assistant_pipeline
from app.config import get_settings
from app.llm.meteorologist import run_meteorologist
from app.llm.writer import run_writer
from app.schemas import AssistantRequest, ReportParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("batch_runner")

# ── Locations ──
LOCATIONS_FILE = _ROOT / "experiment_locations.json"

# ── Parameter combos to run per location ──
#    53 locations × 4 combos = 212 runs (~200 target)
PARAM_COMBOS = [
    {"tone": "technical",      "domain": "risk_analysis"},
    {"tone": "conversational", "domain": "general_public"},
    {"tone": "official",       "domain": "agriculture"},
    {"tone": "technical",      "domain": "extreme_weather"},
]


# =====================================================================
# Supabase helpers
# =====================================================================

def get_supabase() -> Client:
    """Create and return a Supabase client from env vars."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env\n"
            "  SUPABASE_URL=https://xxxxx.supabase.co\n"
            "  SUPABASE_KEY=your-anon-or-publishable-key"
        )
    return create_client(url, key)


def get_completed_runs(sb: Client) -> set[tuple[str, str, str]]:
    """Return set of (location_name, tone, domain) tuples already completed successfully."""
    try:
        resp = (
            sb.table("experiment_runs")
            .select("location_name, tone, domain")
            .is_("error", "null")
            .execute()
        )
        return {(r["location_name"], r["tone"], r["domain"]) for r in resp.data}
    except Exception as e:
        logger.warning("Could not fetch completed runs: %s", e)
        return set()


def insert_record(sb: Client, record: dict) -> None:
    """Insert a run record into Supabase, logging errors but never crashing."""
    try:
        sb.table("experiment_runs").insert(record).execute()
    except Exception as e:
        logger.error("Supabase insert failed: %s", str(e)[:500])
        # Fallback: save to local file so data isn't lost
        fallback = _ROOT / "experiment_log"
        fallback.mkdir(exist_ok=True)
        ts = int(time.time())
        loc = record.get("location_name", "unknown")
        path = fallback / f"run_{loc}_{ts}.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        logger.info("Saved fallback record to %s", path)


# =====================================================================
# Pipeline execution
# =====================================================================

async def run_single_experiment(location: dict, params: dict) -> dict:
    """
    Run the full pipeline for one (location, params) combination.
    Returns the record dict ready for Supabase insert.
    """
    settings = get_settings()
    start = time.perf_counter()

    # ── Block 1: Assistant (data gathering) ──
    req = AssistantRequest(
        query=location["name"],
        latitude=location["lat"],
        longitude=location["lon"],
        context_style="hierarchical",
        use_cache=False,
    )

    async with httpx.AsyncClient(timeout=120) as client:
        payload, assistant_meta = await run_assistant_pipeline(req, client)

    assistant_ms = int((time.perf_counter() - start) * 1000)

    # ── Block 2: Meteorologist + Verification ──
    met_start = time.perf_counter()
    met_output, met_meta = await run_meteorologist(payload, use_cache=False)
    meteorologist_ms = int((time.perf_counter() - met_start) * 1000)

    # ── Block 3: Writer ──
    writer_start = time.perf_counter()
    report_params = ReportParams(
        tone=params["tone"],
        length="short",
        domain=params["domain"],
    )
    writer_output, writer_meta = await run_writer(
        payload, met_output, report_params, use_cache=False
    )
    writer_ms = int((time.perf_counter() - writer_start) * 1000)

    total_ms = int((time.perf_counter() - start) * 1000)

    # ── Extract verification data ──
    confidence_report = met_output.confidence_report or {}
    component_scores = confidence_report.get("component_scores", {})
    eval_metrics = met_meta.get("eval_metrics", {})
    validation_details = met_meta.get("validation_details", {})

    return {
        # Identity
        "runner_name": os.getenv("RUNNER_NAME", "unknown"),
        "location_name": location["name"],
        "latitude": location["lat"],
        "longitude": location["lon"],
        "climate_zone": location.get("climate", ""),
        # Parameters
        "context_style": "hierarchical",
        "tone": params["tone"],
        "domain": params["domain"],
        "length": "short",
        "gemini_model": settings.gemini_model,
        # Scores
        "overall_score": confidence_report.get("overall_score", 0.0),
        "fact_score": component_scores.get("data_fact", 0.0),
        "rule_score": component_scores.get("met_rule", 0.0),
        "causal_score": component_scores.get("causal_graph", 0.0),
        "temporal_score": component_scores.get("temporal_consistency", 0.0),
        # Evaluation metrics
        "false_claim_rate": eval_metrics.get("false_claim_rate", 0.0),
        "total_claims": eval_metrics.get("total_claims", 0),
        "failed_claims": eval_metrics.get("failed_claims", 0),
        "rule_violations": eval_metrics.get("rule_violations", 0),
        "verified_report": eval_metrics.get("verified_report", False),
        "cross_scale_consistent": eval_metrics.get("cross_scale_consistent", True),
        "correction_needed": eval_metrics.get("correction_needed", False),
        "hallucination_precision": eval_metrics.get("hallucination_precision", 0.0),
        "hallucination_recall": eval_metrics.get("hallucination_recall", 0.0),
        # Timing
        "total_pipeline_ms": total_ms,
        "assistant_ms": assistant_ms,
        "meteorologist_ms": meteorologist_ms,
        "writer_ms": writer_ms,
        # Rich data (JSONB)
        "meteorologist_raw": met_output.model_dump(mode="json"),
        "writer_raw": writer_output.model_dump(mode="json"),
        "fact_validation_details": validation_details.get("fact_results", []),
        "rule_validation_details": validation_details.get("rule_results", {}),
        "causal_validation_details": validation_details.get("causal_results", {}),
        "temporal_validation_details": validation_details.get("temporal_results", {}),
        "confidence_failures": confidence_report.get("failures", []),
        "degradation_codes": payload.degradation_codes,
        # No error
        "error": None,
    }


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is a Gemini quota/rate-limit error."""
    msg = str(exc).lower()
    markers = [
        "429", "quota", "rate limit", "resource exhausted",
        "too many requests", "daily limit",
    ]
    return any(m in msg for m in markers)


def _make_error_record(location: dict, params: dict, error: str) -> dict:
    """Build a minimal error record for Supabase."""
    return {
        "runner_name": os.getenv("RUNNER_NAME", "unknown"),
        "location_name": location["name"],
        "latitude": location["lat"],
        "longitude": location["lon"],
        "climate_zone": location.get("climate", ""),
        "context_style": "hierarchical",
        "tone": params["tone"],
        "domain": params["domain"],
        "length": "short",
        "gemini_model": get_settings().gemini_model,
        "error": error[:2000],
    }


# =====================================================================
# Main loop
# =====================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for IEEE journal data collection"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test Supabase connection and show run queue; no API calls",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum experiments to run (0 = all remaining)",
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Seconds to wait between runs (default: 5)",
    )
    args = parser.parse_args()

    # ── Load locations ──
    if not LOCATIONS_FILE.exists():
        logger.error("Locations file not found: %s", LOCATIONS_FILE)
        sys.exit(1)

    locations = json.loads(LOCATIONS_FILE.read_text(encoding="utf-8"))
    logger.info("Loaded %d locations from %s", len(locations), LOCATIONS_FILE.name)

    # ── Connect to Supabase ──
    sb = get_supabase()

    # ── Build run queue ──
    completed = get_completed_runs(sb)
    queue: list[tuple[dict, dict]] = []
    for loc in locations:
        for params in PARAM_COMBOS:
            key = (loc["name"], params["tone"], params["domain"])
            if key not in completed:
                queue.append((loc, params))

    total_possible = len(locations) * len(PARAM_COMBOS)

    if args.dry_run:
        print()
        print("=== DRY RUN ===")
        print(f"  [OK] Supabase connection OK")
        print(f"  [OK] {len(locations)} locations loaded")
        print(f"  [OK] {len(PARAM_COMBOS)} parameter combos per location")
        print(f"  [OK] Total possible runs: {total_possible}")
        print(f"  [OK] Already completed: {len(completed)} runs")
        print(f"  [OK] Remaining: {len(queue)} runs")
        print(f"  [OK] Runner name: {os.getenv('RUNNER_NAME', 'unknown')}")
        print(f"  [OK] Gemini model: {get_settings().gemini_model}")
        print()
        if queue:
            print("  Next 5 in queue:")
            for loc, params in queue[:5]:
                print(f"    -> {loc['name']} ({params['tone']}/{params['domain']})")
        print()
        return

    if not queue:
        print("✓ All experiments completed! Nothing to do.")
        return

    # Apply limit
    run_limit = args.limit if args.limit > 0 else len(queue)
    queue = queue[:run_limit]

    logger.info(
        "Starting batch: %d runs queued (of %d remaining, %d total completed)",
        len(queue), total_possible - len(completed), len(completed),
    )

    consecutive_quota_failures = 0
    completed_count = 0
    failed_count = 0

    for idx, (loc, params) in enumerate(queue, 1):
        label = f"[{idx}/{len(queue)}] {loc['name']} ({params['tone']}/{params['domain']})"

        try:
            logger.info("> %s", label)
            record = await run_single_experiment(loc, params)
            insert_record(sb, record)

            score = record["overall_score"]
            ms = record["total_pipeline_ms"]
            logger.info(
                "[OK] %s — score: %.2f — %.1fs",
                label, score, ms / 1000,
            )

            consecutive_quota_failures = 0
            completed_count += 1

        except Exception as e:
            if _is_quota_error(e):
                consecutive_quota_failures += 1
                logger.warning(
                    "[WARN] Quota/rate error on %s (consecutive: %d): %s",
                    label, consecutive_quota_failures, str(e)[:300],
                )

                if consecutive_quota_failures >= 3:
                    logger.error(
                        "[FAIL] 3 consecutive quota failures — daily limit likely reached. "
                        "Stopping. Will resume on next run."
                    )
                    break

                # Wait and retry once
                wait_secs = 60 * consecutive_quota_failures  # 60s, 120s
                logger.info("Waiting %ds before retry...", wait_secs)
                await asyncio.sleep(wait_secs)

                try:
                    record = await run_single_experiment(loc, params)
                    insert_record(sb, record)
                    score = record["overall_score"]
                    ms = record["total_pipeline_ms"]
                    logger.info("[OK] %s (retry) — score: %.2f — %.1fs", label, score, ms / 1000)
                    consecutive_quota_failures = 0
                    completed_count += 1
                except Exception as retry_e:
                    logger.error("[FAIL] Retry failed for %s: %s", label, str(retry_e)[:300])
                    insert_record(sb, _make_error_record(loc, params, str(retry_e)))
                    failed_count += 1
                    if _is_quota_error(retry_e):
                        consecutive_quota_failures += 1
                        if consecutive_quota_failures >= 3:
                            logger.error("[FAIL] 3 consecutive quota failures. Stopping.")
                            break

            else:
                # Non-quota error — log and continue
                logger.error("[FAIL] Error on %s: %s", label, str(e)[:500])
                insert_record(sb, _make_error_record(loc, params, str(e)))
                failed_count += 1
                consecutive_quota_failures = 0

        # Delay between runs
        if idx < len(queue):
            await asyncio.sleep(args.delay)

    # ── Summary ──
    print()
    print("=== BATCH COMPLETE ===")
    print(f"  Succeeded: {completed_count}")
    print(f"  Failed:    {failed_count}")
    print(f"  Skipped:   {len(queue) - completed_count - failed_count}")
    print(f"  Total in DB: {len(completed) + completed_count}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
