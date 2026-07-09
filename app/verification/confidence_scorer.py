import os
import yaml
from typing import List, Dict, Any, Tuple

class ConfidenceScorer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.weights = {
            "data_fact": 0.35,
            "met_rule": 0.10,
            "causal_graph": 0.20,
            "temporal_consistency": 0.20
        }
        self._load_weights()

    def _load_weights(self):
        if not os.path.exists(self.config_path):
            return # fallback to default weights
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            custom_weights = config.get("weights", {})
            if custom_weights:
                self.weights.update(custom_weights)
        except Exception:
            pass

    def compute_report(
        self,
        fact_results: List[Any],
        rule_results: Dict[str, Any],
        causal_result: Dict[str, Any],
        temporal_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes the confidence report by aggregating the outputs of the 4 validation modules.
        Uses a weighted harmonic mean to ensure critical failures drop the overall score to 0.0.
        
        Args:
            fact_results: List of ValidationResult objects/dicts from DataFactValidator.
            rule_results: Output dict from MeteorologicalRuleEngine.
            causal_result: Output dict from CausalGraph.
            temporal_result: Output dict from TemporalConsistencyChecker.
            
        Returns:
            Dict containing the ConfidenceReport fields.
        """
        # 1. Compute individual component scores
        # Data Fact Score: Fraction of claims passed
        passed_facts = 0.0
        for r in fact_results:
            is_dict = isinstance(r, dict)
            if r.get("is_pass", False) if is_dict else getattr(r, "is_pass", False):
                passed_facts += 1.0
        fact_score = passed_facts / len(fact_results) if fact_results else 1.0
        
        # Rule Engine Score
        met_rule_score = rule_results.get("rule_consistency_score", 1.0)
        
        # Causal Graph Score
        causal_score = causal_result.get("validity_score", 1.0)
        
        # Temporal Checker Score
        temporal_score = temporal_result.get("consistency_score", 1.0)
        
        scores = {
            "data_fact": float(fact_score),
            "met_rule": float(met_rule_score),
            "causal_graph": float(causal_score),
            "temporal_consistency": float(temporal_score)
        }

        # 2. Compute overall score
        # The rule engine is additive and excluded from the harmonic mean zeroing behavior.
        # The other three components (data_fact, causal_graph, temporal_consistency) are evaluated via weighted harmonic mean.
        non_rule_keys = ["data_fact", "causal_graph", "temporal_consistency"]
        if any(scores[k] == 0.0 for k in non_rule_keys):
            overall_score = 0.0
        else:
            w_sum_non_rule = sum(self.weights[k] for k in non_rule_keys)
            w_denom_non_rule = sum(self.weights[k] / scores[k] for k in non_rule_keys)
            base_harmonic_score = w_sum_non_rule / w_denom_non_rule if w_denom_non_rule > 0 else 1.0
            
            w_rule = self.weights.get("met_rule", 0.10)
            overall_score = (base_harmonic_score * w_sum_non_rule + scores["met_rule"] * w_rule) / (w_sum_non_rule + w_rule)

        # 3. Collect and Rank Failures/Contradictions by Severity
        failures = []
        
        # Fact validation failures (Medium severity)
        for r in fact_results:
            is_dict = isinstance(r, dict)
            is_pass = r.get("is_pass", False) if is_dict else getattr(r, "is_pass", False)
            if not is_pass:
                reason = r.get("reason", "") if is_dict else getattr(r, "reason", "")
                claim_id = r.get("claim_id", "UNKNOWN") if is_dict else getattr(r, "claim_id", "UNKNOWN")
                obs = r.get("observed_value", "None") if is_dict else getattr(r, "observed_value", "None")
                exp = r.get("expected_value", "None") if is_dict else getattr(r, "expected_value", "None")
                failures.append({
                    "component": "data_fact",
                    "id": claim_id,
                    "severity": "medium",
                    "message": f"Claim failed fact-checking: {reason} (Observed: {obs}, Expected: {exp})"
                })

        # Meteorological rule violations (High severity)
        for r in rule_results.get("rule_results", []):
            if r["fired"] and not r["consequent_satisfied"]:
                failures.append({
                    "component": "met_rule",
                    "id": r["rule_id"],
                    "severity": "high",
                    "message": f"Meteorological rule violation: {r['explanation']}"
                })

        # Causal Graph invalid transitions (High severity for unlinked/forbidden, Low for missing intermediate)
        for t in causal_result.get("invalid_transitions", []):
            # Check if transition is explicitly forbidden
            is_forbidden = causal_result.get("has_forbidden_edges", False)
            failures.append({
                "component": "causal_graph",
                "id": "FORBIDDEN_TRANSITION" if is_forbidden else "UNLINKED_TRANSITION",
                "severity": "high",
                "message": f"Causal chain invalid transition: {t[0]} -> {t[1]} (Physically forbidden: {is_forbidden})"
            })
            
        for node in causal_result.get("missing_intermediates", []):
            failures.append({
                "component": "causal_graph",
                "id": "MISSING_INTERMEDIATE",
                "severity": "low",
                "message": f"Omitted expected intermediate meteorological event in causal chain: '{node}'"
            })

        # Temporal scale contradictions (High/Medium severity)
        for c in temporal_result.get("contradictions", []):
            failures.append({
                "component": "temporal_consistency",
                "id": f"CONTRA_{c['variable'].upper()}",
                "severity": c["severity"],
                "message": f"Cross-scale consistency clash on '{c['variable']}': {c['reason']}"
            })

        # Sort failures: high -> medium -> low
        severity_map = {"high": 0, "medium": 1, "low": 2}
        sorted_failures = sorted(failures, key=lambda x: severity_map.get(x["severity"], 3))

        # 4. Generate plain English summary paragraph
        summary_parts = []
        if overall_score >= 0.8:
            summary_parts.append("The weather narrative has high meteorological confidence and is physically coherent.")
        elif overall_score >= 0.5:
            summary_parts.append("The weather narrative has moderate meteorological confidence with minor contradictions.")
        else:
            summary_parts.append("The weather narrative has low confidence and contains critical meteorological discrepancies.")

        summary_parts.append(
            f"Factual claims validation scored {scores['data_fact']:.2f}. "
            f"Meteorological rule consistency scored {scores['met_rule']:.2f}. "
            f"Causal reasoning validity scored {scores['causal_graph']:.2f}. "
            f"Cross-scale temporal consistency scored {scores['temporal_consistency']:.2f}."
        )

        high_count = sum(1 for f in failures if f["severity"] == "high")
        medium_count = sum(1 for f in failures if f["severity"] == "medium")
        low_count = sum(1 for f in failures if f["severity"] == "low")
        
        if failures:
            summary_parts.append(
                f"A total of {len(failures)} issues were flagged: "
                f"{high_count} high severity contradictions, {medium_count} medium severity warnings, and {low_count} minor omissions."
            )
        else:
            summary_parts.append("No meteorological anomalies or contradictions were detected in the generated narrative.")

        report_summary = " ".join(summary_parts)

        return {
            "overall_score": float(overall_score),
            "component_scores": scores,
            "failures": sorted_failures,
            "summary": report_summary
        }
