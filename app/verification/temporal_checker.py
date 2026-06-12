import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from app.verification.utils import normalize_to_utc

class TemporalConsistencyChecker:
    def __init__(self, tolerances: Optional[Dict[str, float]] = None):
        # Tolerances left in constructor for future extension/observability
        self.tolerances = tolerances or {}

    def _normalize_variable(self, var: str) -> str:
        var = str(var).lower().strip()
        var_map = {
            "pressure": "pressure",
            "pressure_change": "pressure",
            "relative_humidity": "humidity",
            "humidity": "humidity",
            "temperature": "temperature",
            "temperature_change": "temperature",
            "temp_change": "temperature",
            "temp": "temperature",
            "weather_category": "weather_category",
            "precipitation": "precipitation",
            "wind_speed": "wind_speed",
            "visibility": "visibility",
            "wind_gust": "wind_gust",
            "fog_persistence": "fog_persistence"
        }
        return var_map.get(var, var)

    def _get_claim_direction(self, claim: Dict[str, Any]) -> str:
        assertion = str(claim.get("assertion_type", "")).lower().strip()
        thresh = str(claim.get("threshold_or_delta", "")).lower().strip()
        
        if "decreasing" in assertion or "drop" in thresh or "cold" in thresh or "dry" in thresh or "<" in thresh:
            return "down"
        elif "increasing" in assertion or "rise" in thresh or "heat" in thresh or "strong" in thresh or "severe" in thresh or "heavy" in thresh or "extreme" in thresh or ">" in thresh:
            return "up"
        return "neutral"

    def validate_temporal_consistency(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates directional narrative coherence across daily, 6-hour, and hourly forecast scales.
        
        Args:
            claims: List of claims from the Meteorologist's output.
            
        Returns:
            Dict containing:
            - is_consistent: Boolean indicating if there are no high-severity contradictions.
            - consistency_score: Float between 0.0 and 1.0.
            - contradictions: List of contradictions detected with time windows and severity levels.
            - checks_count: Total number of cross-scale claim pairs evaluated.
            - has_medium_contradictions: Boolean indicating if there are medium contradictions.
        """
        # Separate claims by scale using flexible key lookups
        daily_claims = []
        six_hour_claims = []
        hourly_claims = []
        
        for claim in claims:
            scale = str(claim.get("source_scale") or claim.get("data_scale", "hourly")).lower().strip()
            if scale == "daily":
                daily_claims.append(claim)
            elif scale == "six_hour":
                six_hour_claims.append(claim)
            elif scale in ["hourly", "current"]:
                hourly_claims.append(claim)

        contradictions = []
        checks_count = 0

        # Helper to check directional coherence between two claims
        def check_directional_contradiction(c1: Dict[str, Any], c2: Dict[str, Any], scale1: str, scale2: str):
            nonlocal checks_count
            v1 = self._normalize_variable(c1.get("variable", ""))
            v2 = self._normalize_variable(c2.get("variable", ""))
            if v1 != v2:
                return
                
            try:
                start1 = normalize_to_utc(c1.get("window_start_utc", ""))
                end1 = normalize_to_utc(c1.get("window_end_utc", ""))
                start2 = normalize_to_utc(c2.get("window_start_utc", ""))
                end2 = normalize_to_utc(c2.get("window_end_utc", ""))
            except Exception:
                return
                
            # Check overlap
            if max(start1, start2) <= min(end1, end2):
                checks_count += 1
                dir1 = self._get_claim_direction(c1)
                dir2 = self._get_claim_direction(c2)
                
                if dir1 != "neutral" and dir2 != "neutral" and dir1 != dir2:
                    contradictions.append({
                        "claim_ids": [c1.get("claim_id", "UNKNOWN"), c2.get("claim_id", "UNKNOWN")],
                        "variable": v1,
                        "time_window": (c2.get("window_start_utc"), c2.get("window_end_utc")),
                        "severity": "high",
                        "reason": f"Directional contradiction: {scale1} claim asserts '{dir1}' but overlapping {scale2} claim asserts '{dir2}'."
                    })

        # 1. Compare Daily vs. 6-Hour claims
        for d_claim in daily_claims:
            for s_claim in six_hour_claims:
                check_directional_contradiction(d_claim, s_claim, "daily", "6-hour")

        # 2. Compare 6-Hour vs. Hourly claims
        for s_claim in six_hour_claims:
            for h_claim in hourly_claims:
                check_directional_contradiction(s_claim, h_claim, "6-hour", "hourly")

        # 3. Compare Daily vs. Hourly claims (Problem 3: Direct check skipping 6h)
        for d_claim in daily_claims:
            for h_claim in hourly_claims:
                check_directional_contradiction(d_claim, h_claim, "daily", "hourly")

        # Calculate consistency score
        # High-severity contradictions subtract 1.0, medium-severity subtract 0.5
        if checks_count > 0:
            penalty = 0.0
            for contra in contradictions:
                if contra["severity"] == "high":
                    penalty += 1.0
                elif contra["severity"] == "medium":
                    penalty += 0.5
            consistency_score = max(0.0, 1.0 - (penalty / checks_count))
        else:
            consistency_score = 1.0
            
        is_consistent = not any(contra["severity"] == "high" for contra in contradictions)
        has_medium_contradictions = any(contra["severity"] == "medium" for contra in contradictions)
        
        return {
            "is_consistent": is_consistent,
            "consistency_score": float(consistency_score),
            "contradictions": contradictions,
            "checks_count": checks_count,
            "has_medium_contradictions": has_medium_contradictions
        }
