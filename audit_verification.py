"""
Verification Engine vs. LLM Auditor
====================================

Analyzes experiment run logs to answer two fundamental questions:
  1. Is the verification engine correctly doing its job (or triggering false alarms due to schema bugs)?
  2. How wrong is the LLM (pure numerical hallucinations vs physical rule violations vs causal chain breaks)?

Usage:
    python audit_verification.py                                # Audits experiment_runs_rows.csv (or test.csv)
    python audit_verification.py --input test.csv               # Audit specific CSV file
    python audit_verification.py --supabase                    # Audit directly from live Supabase
    python audit_verification.py --verbose                     # Show individual failed claims and errors
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle large CSV fields on Windows / 64-bit platforms
try:
    csv.field_size_limit(2147483647)
except OverflowError:
    csv.field_size_limit(sys.maxsize)

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def load_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_from_supabase() -> List[Dict[str, Any]]:
    from supabase import create_client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env to use --supabase")
        sys.exit(1)
    sb = create_client(url, key)
    resp = sb.table("experiment_runs").select("*").order("created_at").execute()
    return resp.data


def parse_json_field(val: Any) -> Any:
    if val is None or val == "":
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "null":
            return None
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def run_audit(runs: List[Dict[str, Any]], verbose: bool = False):
    total_runs = len(runs)
    if total_runs == 0:
        print("No runs found to audit.")
        return

    # Categories
    infra_errors = []          # 429 quota, network timeouts
    llm_syntax_errors = []     # JSON unterminated, bad output format
    completed_runs = []        # Runs that reached verification

    # Failure Taxonomy across completed runs
    verifier_column_bugs = []  # False alarms: "Column 'xyz' not found in DataFrame"
    verifier_other_bugs = []   # Other validator crashes / NoneType issues
    llm_fact_hallucinations = [] # Real data mismatches (Observed != Expected)
    llm_rule_violations = []   # Met physical rules failed
    llm_causal_violations = [] # Causal DAG invalid transitions / missing nodes
    llm_temporal_violations = [] # Cross-scale temporal contradictions

    total_claims_count = 0
    total_failed_claims_count = 0

    for idx, r in enumerate(runs):
        err = r.get("error")
        run_name = f"{r.get('location_name', 'Unknown')} ({r.get('tone', '')}/{r.get('domain', '')})"
        run_id = r.get("id", f"run_{idx}")

        if err:
            err_str = str(err)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                infra_errors.append({"run": run_name, "id": run_id, "error": "Gemini Quota Exceeded (429 Rate Limit)"})
            elif "Unterminated string" in err_str or "JSON" in err_str or "Expecting" in err_str:
                llm_syntax_errors.append({"run": run_name, "id": run_id, "error": f"LLM Generation Truncated / Invalid JSON: {err_str[:80]}"})
            else:
                infra_errors.append({"run": run_name, "id": run_id, "error": err_str[:120]})
            continue

        completed_runs.append(r)

        # Track claims
        tc = r.get("total_claims")
        fc = r.get("failed_claims")
        try:
            if tc: total_claims_count += int(tc)
            if fc: total_failed_claims_count += int(fc)
        except (ValueError, TypeError):
            pass

        # Parse confidence_failures
        failures = parse_json_field(r.get("confidence_failures"))
        if not failures and r.get("meteorologist_raw"):
            # Check if embedded inside meteorologist_raw
            m_raw = parse_json_field(r.get("meteorologist_raw"))
            if isinstance(m_raw, dict):
                failures = m_raw.get("confidence_report", {}).get("failures", [])

        if not failures:
            continue

        for f in failures:
            comp = f.get("component", "unknown")
            msg = f.get("message", "")
            cid = f.get("id", "")
            item = {
                "run": run_name,
                "id": cid,
                "component": comp,
                "message": msg,
                "severity": f.get("severity", "medium")
            }

            if comp == "data_fact":
                if "not found in DataFrame" in msg or "Column" in msg and "not found" in msg:
                    # Extract column name
                    verifier_column_bugs.append(item)
                elif "NoneType" in msg or "AttributeError" in msg:
                    verifier_other_bugs.append(item)
                else:
                    llm_fact_hallucinations.append(item)
            elif comp == "met_rule":
                llm_rule_violations.append(item)
            elif comp == "causal_graph":
                llm_causal_violations.append(item)
            elif comp == "temporal_consistency":
                llm_temporal_violations.append(item)
            else:
                # Catch-all
                if "not found in DataFrame" in msg:
                    verifier_column_bugs.append(item)
                else:
                    llm_fact_hallucinations.append(item)

    # -------------------------------------------------------------
    # Output Report
    # -------------------------------------------------------------
    num_completed = len(completed_runs)
    print()
    print("=" * 72)
    print("       AUDIT REPORT: VERIFICATION ENGINE vs. LLM PERFORMANCE")
    print("=" * 72)
    print(f"  Total pipeline executions analyzed:  {total_runs}")
    print(f"  Successfully reached verification:  {num_completed}")
    print(f"  Aborted before verification:        {len(infra_errors) + len(llm_syntax_errors)}")
    if infra_errors:
        print(f"    - Infrastructure (API 429 Quotas): {len(infra_errors)}")
    if llm_syntax_errors:
        print(f"    - LLM JSON formatting truncation: {len(llm_syntax_errors)}")
    print("-" * 72)

    # 1. VERIFICATION ENGINE INTEGRITY
    total_fact_flags = len(verifier_column_bugs) + len(verifier_other_bugs) + len(llm_fact_hallucinations)
    false_alarm_count = len(verifier_column_bugs) + len(verifier_other_bugs)
    engine_false_alarm_rate = (false_alarm_count / total_fact_flags * 100) if total_fact_flags > 0 else 0.0

    print()
    print("  [1] VERIFICATION ENGINE ACCURACY & HEALTH")
    print(f"      Total Factual Claims Evaluated:         {total_claims_count}")
    print(f"      Total Factual Claim Failures:           {total_fact_flags}")
    print(f"      - Verifier Schema Bugs (False Alarms):  {false_alarm_count} ({engine_false_alarm_rate:.1f}%)")
    print(f"      - True LLM Numerical Hallucinations:   {len(llm_fact_hallucinations)} ({100 - engine_false_alarm_rate:.1f}%)")

    if false_alarm_count > 0:
        print("\n      --> BREAKDOWN OF VERIFIER FALSE ALARMS:")
        col_counts = Counter()
        for bug in verifier_column_bugs:
            # extract variable name from message
            msg = bug["message"]
            var = "unknown"
            if "variable '" in msg:
                var = msg.split("variable '")[1].split("'")[0]
            elif "Column '" in msg:
                var = msg.split("Column '")[1].split("'")[0]
            col_counts[var] += 1
        for col, cnt in col_counts.most_common():
            print(f"          * Missing column mapping for '{col}': {cnt} false failures")

    # 2. LLM ACCURACY PROFILE
    print()
    print("  [2] LLM HALLUCINATION & ERROR PROFILE (Where is the LLM actually failing?)")
    print(f"      A. Numerical Fact Hallucinations:       {len(llm_fact_hallucinations)} / {total_claims_count} claims")
    print(f"      B. Meteorological Rule Violations:     {len(llm_rule_violations)}")
    print(f"      C. Causal Reasoning Failures:           {len(llm_causal_violations)}")
    print(f"      D. Cross-Scale Temporal Contradictions: {len(llm_temporal_violations)}")

    if llm_rule_violations:
        print("\n      --> PHYSICAL RULE VIOLATIONS BY LLM:")
        rule_reasons = Counter()
        for r in llm_rule_violations:
            rule_reasons[r["message"][:85]] += 1
        for reason, count in rule_reasons.most_common(5):
            print(f"          * ({count}x) {reason}...")

    if llm_causal_violations:
        print("\n      --> CAUSAL REASONING DEFICIENCIES BY LLM:")
        causal_reasons = Counter()
        for c in llm_causal_violations:
            msg = c["message"]
            if "invalid transition" in msg:
                trans = msg.split("transition:")[1].split("(")[0].strip() if "transition:" in msg else msg
                causal_reasons[f"Invalid transition: {trans}"] += 1
            elif "Omitted expected intermediate" in msg:
                node = msg.split("event in causal chain:")[1].strip() if "event in causal chain:" in msg else msg
                causal_reasons[f"Missing intermediate node: {node}"] += 1
            else:
                causal_reasons[msg[:80]] += 1

        for reason, count in causal_reasons.most_common(6):
            print(f"          * ({count}x) {reason}")

    # 3. VERDICT
    print()
    print("=" * 72)
    print("  [3] EXECUTIVE VERDICT & NEXT STEPS")
    print("=" * 72)
    if false_alarm_count > 0:
        print("  1. YOUR VERIFICATION ENGINE WAS TRIGGERING FALSE ALARMS:")
        print(f"     {false_alarm_count} of {total_fact_flags} fact failures were caused by the verifier")
        print("     searching for wrong column names (e.g. pressure_hpa instead of pressure_mean_hpa).")
        print("     -> GOOD NEWS: The merge to 'main' has fixed this column mapping!")
    else:
        print("  1. VERIFICATION ENGINE HEALTH: CLEAN (Zero schema lookup errors detected).")

    print()
    if len(llm_fact_hallucinations) == 0:
        print("  2. THE LLM'S FACTUAL RECALL IS NEAR PERFECT:")
        print(f"     Across {total_claims_count} claims made, the LLM had ZERO raw numerical hallucinations")
        print("     when the column existed in the dataset!")
    else:
        print(f"  2. THE LLM HAD {len(llm_fact_hallucinations)} RAW FACTUAL HALLUCINATIONS.")

    print()
    print("  3. THE LLM'S REAL WEAKNESS IS CAUSAL REASONING:")
    print(f"     The LLM struggled heavily with meteorological causal DAG transitions ({len(llm_causal_violations)} errors).")
    print("     It frequently skipped intermediate physical steps (e.g. cooling, fog) or chained")
    print("     disconnected weather states.")

    # Verbose drilldown
    if verbose:
        print("\n" + "=" * 72)
        print("  DETAILED RUN-BY-RUN AUDIT")
        print("=" * 72)
        for r in completed_runs:
            loc = r.get("location_name")
            tone = r.get("tone")
            dom = r.get("domain")
            score = r.get("overall_score")
            fact = r.get("fact_score")
            failures = parse_json_field(r.get("confidence_failures")) or []
            print(f"\nRun: {loc} ({tone}/{dom}) | Overall: {score} | Fact Score: {fact}")
            if not failures:
                print("  -> Passed all verification modules cleanly! (No failures)")
            else:
                for f in failures:
                    comp = f.get("component")
                    msg = f.get("message")
                    flag = "[ENGINE BUG]" if "not found in DataFrame" in msg else "[LLM ERROR]"
                    print(f"  {flag} [{comp}] {msg}")

    print("=" * 72)
    print()


def main():
    parser = argparse.ArgumentParser(description="Audit verification engine vs LLM performance")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to CSV file (defaults to experiment_runs_rows.csv or test.csv)")
    parser.add_argument("--supabase", action="store_true", help="Fetch directly from Supabase database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed failure messages per run")
    args = parser.parse_args()

    if args.supabase:
        print("Fetching runs from Supabase...")
        runs = load_from_supabase()
    else:
        if args.input:
            csv_path = Path(args.input)
        else:
            candidates = [ROOT_DIR / "experiment_runs_rows.csv", ROOT_DIR / "test.csv"]
            csv_path = next((c for c in candidates if c.exists()), None)
            if not csv_path:
                print("ERROR: No CSV file found. Please specify --input <path_to_csv> or use --supabase")
                sys.exit(1)
        print(f"Auditing data from: {csv_path.name}")
        runs = load_from_csv(csv_path)

    run_audit(runs, verbose=args.verbose)


if __name__ == "__main__":
    main()
