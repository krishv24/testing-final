"""
Experiment Results Analyzer & CSV Exporter.
============================================

Reads all successful runs from Supabase and produces:
  1. experiment_results.csv   — one row per run (for LaTeX tables)
  2. Console summary          — aggregate statistics

Usage:
    python analyze_results.py               # Print summary + export CSV
    python analyze_results.py --progress    # Show completion progress only
    python analyze_results.py --output results/   # Custom output directory
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from supabase import create_client


def get_supabase():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


def fetch_all_runs(sb, include_errors=False):
    """Fetch all runs from Supabase."""
    query = sb.table("experiment_runs").select("*").order("created_at")
    if not include_errors:
        query = query.is_("error", "null")
    resp = query.execute()
    return resp.data


# ── CSV column order ──
CSV_COLUMNS = [
    "id", "created_at", "runner_name",
    "location_name", "latitude", "longitude", "climate_zone",
    "context_style", "tone", "domain", "length", "gemini_model",
    "overall_score", "fact_score", "rule_score", "causal_score", "temporal_score",
    "false_claim_rate", "total_claims", "failed_claims", "rule_violations",
    "verified_report", "cross_scale_consistent", "correction_needed",
    "hallucination_precision", "hallucination_recall",
    "total_pipeline_ms", "assistant_ms", "meteorologist_ms", "writer_ms",
    "degradation_codes", "error",
]


def export_csv(runs: list[dict], output_dir: Path) -> Path:
    """Export runs to CSV (excluding JSONB blob columns for readability)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_results.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            # Flatten degradation_codes list to string
            row = dict(run)
            if isinstance(row.get("degradation_codes"), list):
                row["degradation_codes"] = "; ".join(row["degradation_codes"])
            writer.writerow(row)

    return path


def export_full_json(runs: list[dict], output_dir: Path) -> Path:
    """Export ALL data including JSONB columns as a full JSON dump."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_results_full.json"
    path.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    return path


def compute_summary(runs: list[dict]) -> dict:
    """Compute aggregate statistics across all runs."""
    n = len(runs)
    if n == 0:
        return {"total_runs": 0}

    def avg(key):
        vals = [r.get(key, 0) or 0 for r in runs]
        return round(sum(vals) / n, 4) if vals else 0

    def rate(key, expected=True):
        count = sum(1 for r in runs if r.get(key) == expected)
        return round(count / n, 4)

    # Per-climate-zone breakdown
    by_climate = defaultdict(list)
    for r in runs:
        cz = r.get("climate_zone", "unknown") or "unknown"
        by_climate[cz].append(r.get("overall_score", 0) or 0)

    climate_summary = {}
    for cz, scores in sorted(by_climate.items()):
        climate_summary[cz] = {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
        }

    # Per-runner breakdown
    by_runner = defaultdict(int)
    for r in runs:
        by_runner[r.get("runner_name", "unknown")] += 1

    # Per-tone breakdown
    by_tone = defaultdict(list)
    for r in runs:
        by_tone[r.get("tone", "unknown")].append(r.get("overall_score", 0) or 0)

    tone_summary = {}
    for tone, scores in sorted(by_tone.items()):
        tone_summary[tone] = {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 4),
        }

    # Per-domain breakdown
    by_domain = defaultdict(list)
    for r in runs:
        by_domain[r.get("domain", "unknown")].append(r.get("overall_score", 0) or 0)

    domain_summary = {}
    for dom, scores in sorted(by_domain.items()):
        domain_summary[dom] = {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 4),
        }

    return {
        "total_runs": n,
        "runs_per_runner": dict(by_runner),
        # Averages
        "avg_overall_score": avg("overall_score"),
        "avg_fact_score": avg("fact_score"),
        "avg_rule_score": avg("rule_score"),
        "avg_causal_score": avg("causal_score"),
        "avg_temporal_score": avg("temporal_score"),
        "avg_false_claim_rate": avg("false_claim_rate"),
        "avg_hallucination_precision": avg("hallucination_precision"),
        "avg_hallucination_recall": avg("hallucination_recall"),
        # Rates
        "verified_report_rate": rate("verified_report", True),
        "cross_scale_consistency_rate": rate("cross_scale_consistent", True),
        "correction_rate": rate("correction_needed", True),
        # Timing
        "avg_pipeline_ms": avg("total_pipeline_ms"),
        # Breakdowns
        "by_climate_zone": climate_summary,
        "by_tone": tone_summary,
        "by_domain": domain_summary,
    }


def print_summary(summary: dict) -> None:
    """Pretty-print the summary to console."""
    print()
    print("=" * 65)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total successful runs: {summary['total_runs']}")
    print()

    if summary["total_runs"] == 0:
        print("  No data yet.")
        return

    # Runner contributions
    print("  -- Runs per team member ──")
    for runner, count in summary.get("runs_per_runner", {}).items():
        print(f"    {runner}: {count}")
    print()

    # Key metrics
    print("  -- Key Metrics (averages) ──")
    print(f"    Overall Score:            {summary['avg_overall_score']:.4f}")
    print(f"    Fact Validation Score:    {summary['avg_fact_score']:.4f}")
    print(f"    Rule Consistency Score:   {summary['avg_rule_score']:.4f}")
    print(f"    Causal Graph Score:       {summary['avg_causal_score']:.4f}")
    print(f"    Temporal Consistency:     {summary['avg_temporal_score']:.4f}")
    print(f"    False Claim Rate:         {summary['avg_false_claim_rate']:.4f}")
    print(f"    Hallucination Precision:  {summary['avg_hallucination_precision']:.4f}")
    print(f"    Hallucination Recall:     {summary['avg_hallucination_recall']:.4f}")
    print()
    print(f"    Verified Report Rate:     {summary['verified_report_rate']:.1%}")
    print(f"    Cross-Scale Consistency:  {summary['cross_scale_consistency_rate']:.1%}")
    print(f"    Correction Needed Rate:   {summary['correction_rate']:.1%}")
    print()
    print(f"    Avg Pipeline Time:        {summary['avg_pipeline_ms']:.0f} ms")
    print()

    # By tone
    print("  -- By Tone ──")
    for tone, data in summary.get("by_tone", {}).items():
        print(f"    {tone:20s}  n={data['count']:3d}  avg_score={data['avg_score']:.4f}")
    print()

    # By domain
    print("  -- By Domain ──")
    for dom, data in summary.get("by_domain", {}).items():
        print(f"    {dom:20s}  n={data['count']:3d}  avg_score={data['avg_score']:.4f}")
    print()

    # By climate zone (top 10 by count)
    print("  -- By Climate Zone (top 10) ──")
    climate = summary.get("by_climate_zone", {})
    sorted_climate = sorted(climate.items(), key=lambda x: x[1]["count"], reverse=True)
    for cz, data in sorted_climate[:10]:
        print(
            f"    {cz:35s}  n={data['count']:3d}  "
            f"avg={data['avg_score']:.3f}  "
            f"[{data['min_score']:.3f} – {data['max_score']:.3f}]"
        )
    print()
    print("=" * 65)


def show_progress(sb) -> None:
    """Show completion progress."""
    total_resp = sb.table("experiment_runs").select("id", count="exact").execute()
    success_resp = sb.table("experiment_runs").select("id", count="exact").is_("error", "null").execute()
    error_resp = sb.table("experiment_runs").select("id", count="exact").not_.is_("error", "null").execute()

    total = total_resp.count or len(total_resp.data)
    success = success_resp.count or len(success_resp.data)
    errors = error_resp.count or len(error_resp.data)

    # Load locations to compute target
    locations_file = Path(__file__).resolve().parent / "experiment_locations.json"
    if locations_file.exists():
        locations = json.loads(locations_file.read_text(encoding="utf-8"))
        target = len(locations) * 4  # 4 param combos per location
    else:
        target = 212

    pct = (success / target * 100) if target > 0 else 0

    print()
    print(f"  Progress: {success}/{target} ({pct:.1f}%)")
    print(f"  |-- Successful: {success}")
    print(f"  |-- Errors:     {errors}")
    print(f"  +-- Remaining:  {target - success}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--progress", action="store_true", help="Show completion progress only")
    parser.add_argument("--output", type=str, default="results", help="Output directory (default: results/)")
    args = parser.parse_args()

    sb = get_supabase()

    if args.progress:
        show_progress(sb)
        return

    # Fetch all successful runs
    runs = fetch_all_runs(sb, include_errors=False)
    print(f"Fetched {len(runs)} successful runs from Supabase")

    if not runs:
        print("No successful runs found. Run batch_runner.py first.")
        return

    output_dir = Path(args.output)

    # Export CSV
    csv_path = export_csv(runs, output_dir)
    print(f"✓ CSV exported to {csv_path}")

    # Export full JSON
    json_path = export_full_json(runs, output_dir)
    print(f"✓ Full JSON exported to {json_path}")

    # Export summary
    summary = compute_summary(runs)
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"✓ Summary exported to {summary_path}")

    # Print to console
    print_summary(summary)


if __name__ == "__main__":
    main()
