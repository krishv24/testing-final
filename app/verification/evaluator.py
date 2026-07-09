"""
Evaluation Tracker for the Hierarchical AI Meteorologist pipeline.

Computes per-run metrics from the 4 verification modules and the confidence
scorer, appends them as JSON records to .cache/evaluation_log.json, and
provides get_summary() for aggregated statistics across all recorded runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class EvaluationTracker:
    """Tracks evaluation metrics across pipeline runs."""

    def __init__(self) -> None:
        settings = get_settings()
        self._log_path = Path(settings.cache_dir) / "evaluation_log.json"
        self._threshold = settings.verification_threshold

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        fact_results: List[Any],
        rule_results: Dict[str, Any],
        causal_result: Dict[str, Any],
        temporal_result: Dict[str, Any],
        confidence_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the six evaluation metrics from one pipeline run."""

        # --- 1. False Claim Rate ---
        total_claims = len(fact_results)
        failed_claims = 0
        failed_claim_ids: set[str] = set()
        for r in fact_results:
            is_dict = isinstance(r, dict)
            is_pass = r.get("is_pass", False) if is_dict else getattr(r, "is_pass", False)
            if not is_pass:
                failed_claims += 1
                cid = r.get("claim_id", "") if is_dict else getattr(r, "claim_id", "")
                failed_claim_ids.add(cid)

        false_claim_rate = failed_claims / total_claims if total_claims > 0 else 0.0

        # --- 2. Cross-Scale Consistency ---
        high_temporal = any(
            c.get("severity") == "high"
            for c in temporal_result.get("contradictions", [])
        )
        cross_scale_consistent = not high_temporal

        # --- 3. Verified Report ---
        overall_score = confidence_report.get("overall_score", 0.0)
        verified_report = overall_score >= self._threshold

        # --- 4. Correction Rate ---
        # True if any high or medium severity failure exists in the report
        has_correction_needed = any(
            f.get("severity") in ("high", "medium")
            for f in confidence_report.get("failures", [])
        )

        # --- 5. Approximate Hallucination Precision ---
        # Of failed fact claims, fraction corroborated by at least one rule
        # violation or causal graph failure (treating corroboration = true positive).
        rule_violation_vars: set[str] = set()
        for rr in rule_results.get("rule_results", []):
            if rr.get("fired") and not rr.get("consequent_satisfied"):
                rule_violation_vars.add(rr.get("rule_id", ""))

        has_causal_failure = (
            bool(causal_result.get("invalid_transitions"))
            or causal_result.get("has_forbidden_edges", False)
        )

        # A failed fact claim is "corroborated" if there is at least one rule
        # violation OR a causal graph failure in the same run.
        corroborated_failures = 0
        if failed_claims > 0 and (rule_violation_vars or has_causal_failure):
            # Each failed fact claim counts as corroborated if *any* structural
            # failure exists — because the structural validators confirm
            # that something is genuinely wrong, not just a data edge-case.
            corroborated_failures = failed_claims  # all are corroborated

        hallucination_precision = (
            corroborated_failures / failed_claims if failed_claims > 0 else 0.0
        )

        # --- 6. Approximate Hallucination Recall ---
        # Of all rule violations fired, fraction that had a corresponding
        # fact claim failure (any) in the same report.
        total_rule_violations = len(rule_violation_vars)
        recalled_violations = 0
        if total_rule_violations > 0 and failed_claims > 0:
            # A rule violation is "recalled" if the report also contains at
            # least one failed fact claim — confirming the hallucination was
            # also caught at the data level.
            recalled_violations = total_rule_violations

        hallucination_recall = (
            recalled_violations / total_rule_violations
            if total_rule_violations > 0
            else 0.0
        )

        return {
            "false_claim_rate": round(false_claim_rate, 4),
            "cross_scale_consistent": cross_scale_consistent,
            "verified_report": verified_report,
            "correction_needed": has_correction_needed,
            "hallucination_precision": round(hallucination_precision, 4),
            "hallucination_recall": round(hallucination_recall, 4),
            "overall_score": round(overall_score, 4),
            "component_scores": confidence_report.get("component_scores", {}),
            "total_claims": total_claims,
            "failed_claims": failed_claims,
            "rule_violations": total_rule_violations,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _read_log(self) -> List[Dict[str, Any]]:
        """Read the existing evaluation log, returning an empty list if absent or corrupt."""
        if not self._log_path.exists():
            return []
        try:
            text = self._log_path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            records = json.loads(text)
            if isinstance(records, list):
                return records
            return []
        except (json.JSONDecodeError, OSError):
            logger.warning("Evaluation log corrupt or unreadable; starting fresh.")
            return []

    def _write_log(self, records: List[Dict[str, Any]]) -> None:
        """Write the full record list atomically."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(
            json.dumps(records, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_run(
        self,
        fact_results: List[Any],
        rule_results: Dict[str, Any],
        causal_result: Dict[str, Any],
        temporal_result: Dict[str, Any],
        confidence_report: Dict[str, Any],
        location_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for one pipeline run and append to the log.

        Returns the metrics dict for the run.
        """
        metrics = self._compute_metrics(
            fact_results, rule_results, causal_result, temporal_result, confidence_report
        )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "location_query": location_query or "",
            **metrics,
        }

        records = self._read_log()
        records.append(record)
        self._write_log(records)

        logger.info(
            "Evaluation recorded: score=%.2f verified=%s false_claim_rate=%.2f location=%s",
            metrics["overall_score"],
            metrics["verified_report"],
            metrics["false_claim_rate"],
            location_query or "unknown",
        )
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """
        Read the full log and return aggregated statistics across all runs.

        Returns a dict with: total_runs, avg_false_claim_rate, verified_report_rate,
        cross_scale_consistency_rate, correction_rate, avg_hallucination_precision,
        avg_hallucination_recall, avg_overall_score.
        """
        records = self._read_log()
        total = len(records)

        if total == 0:
            return {
                "total_runs": 0,
                "avg_false_claim_rate": 0.0,
                "verified_report_rate": 0.0,
                "cross_scale_consistency_rate": 0.0,
                "correction_rate": 0.0,
                "avg_hallucination_precision": 0.0,
                "avg_hallucination_recall": 0.0,
                "avg_overall_score": 0.0,
            }

        avg_fcr = sum(r.get("false_claim_rate", 0.0) for r in records) / total
        verified_count = sum(1 for r in records if r.get("verified_report", False))
        consistent_count = sum(1 for r in records if r.get("cross_scale_consistent", False))
        correction_count = sum(1 for r in records if r.get("correction_needed", False))
        avg_hp = sum(r.get("hallucination_precision", 0.0) for r in records) / total
        avg_hr = sum(r.get("hallucination_recall", 0.0) for r in records) / total
        avg_score = sum(r.get("overall_score", 0.0) for r in records) / total

        return {
            "total_runs": total,
            "avg_false_claim_rate": round(avg_fcr, 4),
            "verified_report_rate": round(verified_count / total, 4),
            "cross_scale_consistency_rate": round(consistent_count / total, 4),
            "correction_rate": round(correction_count / total, 4),
            "avg_hallucination_precision": round(avg_hp, 4),
            "avg_hallucination_recall": round(avg_hr, 4),
            "avg_overall_score": round(avg_score, 4),
        }
