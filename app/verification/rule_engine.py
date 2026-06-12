import os
import re
import yaml
import itertools
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from app.verification.utils import normalize_to_utc

class MeteorologicalRuleEngine:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules = []
        self._load_rules()
        
        # Event node synonym mapping
        self.node_synonyms = {
            "humidity_rise": "high_humidity",
            "frontal_passage": "cooling",
            "clearing": "no_rain",
            "fog_formation": "low_visibility",
        }

        # Event mapping from causal chain node name to (variable_type, direction)
        self.event_to_var_dir = {
            "pressure_drop": ("pressure_change_6h", "down"),
            "pressure_rise": ("pressure_change_6h", "up"),
            "stable_pressure": ("pressure_change_6h", "neutral"),
            "wind_increase": ("wind_speed", "up"),
            "light_winds": ("wind_speed", "down"),
            "wind_gusts": ("wind_gust", "up"),
            "high_humidity": ("relative_humidity", "up"),
            "dry_air": ("relative_humidity", "down"),
            "rainfall": ("precipitation", "up"),
            "no_rain": ("precipitation", "down"),
            "extreme_heat": ("temperature", "up"),
            "warming": ("temperature_change", "up"),
            "cooling": ("temperature_change", "down"),
            "low_visibility": ("visibility", "down"),
            "fog_persistence": ("fog_persistence", "neutral"),
            "thunderstorm": ("weather_category", "up")
        }

    def _load_rules(self):
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Rules config file not found: {self.rules_path}")
            
        with open(self.rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.rules = config.get("rules", [])

    def _normalize_node(self, node: str) -> str:
        node = str(node).lower().strip().replace(" ", "_")
        return self.node_synonyms.get(node, node)

    def _get_rule_var_direction(self, var: str, cond: str) -> Tuple[str, str]:
        """Maps rule variable and condition to logical variable type and direction (up/down/neutral)."""
        var = str(var).lower().strip()
        cond = str(cond).lower().strip()
        
        if cond in ["less_than", "less_than_or_equal", "decreasing", "drop", "cold", "dry"]:
            direction = "down"
        elif cond in ["greater_than", "greater_than_or_equal", "increasing", "rise", "heat", "strong", "severe", "heavy", "extreme"]:
            direction = "up"
        else:
            direction = "neutral"
            
        var_map = {
            "pressure_change_6h": "pressure_change_6h",
            "pressure_change_3h": "pressure_change_6h",
            "pressure": "pressure_change_6h",
            "relative_humidity": "relative_humidity",
            "humidity": "relative_humidity",
            "temperature": "temperature",
            "temperature_change_3h": "temperature_change",
            "temperature_change_6h": "temperature_change",
            "weather_category": "weather_category",
            "precipitation": "precipitation",
            "wind_speed": "wind_speed",
            "visibility": "visibility",
            "wind_gust": "wind_gust",
            "fog_persistence": "fog_persistence"
        }
        return var_map.get(var, var), direction

    def _get_claim_var_direction(self, claim: Dict[str, Any]) -> Tuple[str, str]:
        """Maps claim parameters to logical variable type and direction (up/down/neutral)."""
        var = claim.get("variable", "")
        assertion = claim.get("assertion_type", "")
        thresh = claim.get("threshold_or_delta", "")
        
        var = str(var).lower().strip()
        assertion = str(assertion).lower().strip()
        thresh = str(thresh).lower().strip()
        
        if "decreasing" in assertion or "drop" in thresh or "cold" in thresh or "dry" in thresh or "<" in thresh:
            direction = "down"
        elif "increasing" in assertion or "rise" in thresh or "heat" in thresh or "strong" in thresh or "severe" in thresh or "heavy" in thresh or "extreme" in thresh or ">" in thresh:
            direction = "up"
        else:
            direction = "neutral"
            
        var_map = {
            "pressure": "pressure_change_6h",
            "pressure_change": "pressure_change_6h",
            "relative_humidity": "relative_humidity",
            "humidity": "relative_humidity",
            "temperature": "temperature",
            "temperature_change": "temperature_change",
            "temp_change": "temperature_change",
            "weather_category": "weather_category",
            "precipitation": "precipitation",
            "wind_speed": "wind_speed",
            "visibility": "visibility",
            "wind_gust": "wind_gust",
            "fog_persistence": "fog_persistence"
        }
        return var_map.get(var, var), direction

    def _get_variable_margin(self, var: str) -> float:
        var_clean = str(var).lower().strip()
        margins = {
            "temperature": 1.0,
            "temperature_change": 1.0,
            "temperature_change_3h": 1.0,
            "temperature_change_6h": 1.0,
            "temp": 1.0,
            "temp_change": 1.0,
            "relative_humidity": 5.0,
            "humidity": 5.0,
            "pressure": 2.0,
            "pressure_change_6h": 0.5,
            "pressure_change_3h": 0.5,
            "wind_speed": 1.0,
            "wind_gust": 1.0,
            "precipitation": 0.5
        }
        return margins.get(var_clean, 0.0)

    def _parse_claim_threshold(self, threshold_str: str) -> Tuple[Optional[str], Optional[float], Optional[Tuple[float, float]]]:
        """Parses operator and value(s) from a claim's threshold_or_delta string."""
        s = str(threshold_str).strip()
        # Range check: e.g. "1.32 to 3.32"
        range_match = re.match(r"^([0-9.-]+)\s*(?:to|-)\s*([0-9.-]+)$", s)
        if range_match:
            try:
                return "range", None, (float(range_match.group(1)), float(range_match.group(2)))
            except ValueError:
                pass
        
        # Numeric check with optional operator (supporting optional leading >, <, =, etc., and trailing %)
        match = re.match(r"^([<>=]*)\s*([-+]?[0-9]*\.?[0-9]+)\s*%?$", s)
        if match:
            op = match.group(1).replace(" ", "")
            try:
                val = float(match.group(2))
                if not op:
                    op = "=="
                return op, val, None
            except ValueError:
                pass
        return None, None, None

    def _is_magnitude_matched(self, rule_ant: Dict[str, Any], claim: Dict[str, Any]) -> bool:
        # Extract rule threshold
        rule_threshold = rule_ant.get("threshold")
        if rule_threshold is None:
            return True  # No threshold to check
            
        # Get claim threshold and operator
        claim_thresh_str = claim.get("threshold_or_delta", "")
        op, claim_val, claim_range = self._parse_claim_threshold(claim_thresh_str)
        
        # If the rule threshold is a string (e.g., "thunderstorm"), do string/categorical match
        if isinstance(rule_threshold, str):
            if claim_val is not None or claim_range is not None:
                return False
            return rule_threshold.lower().strip() in claim_thresh_str.lower().strip()

        # Get margin for variable
        var_name = rule_ant.get("variable", "")
        margin = self._get_variable_margin(var_name)

        # If rule threshold is a list/range (e.g., [min_val, max_val])
        if isinstance(rule_threshold, list) and len(rule_threshold) == 2:
            rule_min = float(rule_threshold[0])
            rule_max = float(rule_threshold[1])
            
            # If claim is a range
            if claim_range is not None:
                c_min, c_max = claim_range
                return (c_min >= rule_min + margin) and (c_max <= rule_max - margin)
            # If claim is a single value
            if claim_val is not None:
                return (claim_val >= rule_min + margin) and (claim_val <= rule_max - margin)
            return False

        # Otherwise, rule threshold is a single numeric value
        try:
            rule_val = float(rule_threshold)
        except (ValueError, TypeError):
            return True # Fallback if we can't parse rule threshold as float

        # If claim has a range
        if claim_range is not None:
            claim_val = sum(claim_range) / 2.0

        if claim_val is None:
            return False

        # Add 50% threshold limit check to prevent rule firing on weak or opposite magnitude matches
        cond = str(rule_ant.get("condition", "")).lower().strip()
        if cond in ["greater_than", "greater_than_or_equal", "increasing", "rise", "heat", "strong", "severe", "heavy", "extreme"]:
            if claim_val < 0.5 * rule_val:
                return False
        elif cond in ["less_than", "less_than_or_equal", "decreasing", "drop", "cold", "dry"]:
            if claim_val > 1.5 * rule_val:
                return False

        if cond in ["greater_than", "greater_than_or_equal", "increasing", "rise", "heat", "strong", "severe", "heavy", "extreme"]:
            return claim_val >= (rule_val + margin)
        elif cond in ["less_than", "less_than_or_equal", "decreasing", "drop", "cold", "dry"]:
            return claim_val <= (rule_val - margin)
        elif cond == "equals":
            return abs(claim_val - rule_val) <= margin
        else:
            return claim_val >= rule_val

    def _claims_match_antecedent(self, rule_ant: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Finds all claims in claims list that match a rule's antecedent criteria."""
        target_var, target_dir = self._get_rule_var_direction(rule_ant.get("variable", ""), rule_ant.get("condition", ""))
        matched = []
        for claim in claims:
            claim_var, claim_dir = self._get_claim_var_direction(claim)
            if claim_var == target_var and (target_dir == "neutral" or claim_dir == target_dir):
                if self._is_magnitude_matched(rule_ant, claim):
                    matched.append(claim)
        return matched

    def validate_rules(self, claims: List[Dict[str, Any]], causal_chain: List[str]) -> Dict[str, Any]:
        """
        Validates the claims list and causal chain against the meteorological rules database.
        
        Args:
            claims: List of claims from the Meteorologist's output.
            causal_chain: List of causal chain nodes (e.g. ["pressure_drop", "wind_increase"]).
            
        Returns:
            Dict containing:
            - rule_results: List of validation results per rule.
            - rule_consistency_score: Float between 0.0 and 1.0.
            - has_violations: Boolean indicating if any rule fired but failed consequent checks.
        """
        normalized_chain = [self._normalize_node(node) for node in causal_chain]
        results = []
        
        for rule in self.rules:
            rule_id = rule.get("rule_id", "UNKNOWN")
            rule_name = rule.get("name", "Unnamed Rule")
            lag = float(rule.get("expected_lag_hours", 0.0))
            weight = float(rule.get("confidence_weight", 1.0))
            
            # Map antecedent list
            rule_antecedents = []
            if "antecedents" in rule:
                rule_antecedents = rule["antecedents"]
            elif "antecedent" in rule:
                rule_antecedents = [rule["antecedent"]]
                
            consequent = rule.get("consequent", {})
            cons_var, cons_dir = self._get_rule_var_direction(consequent.get("variable", ""), consequent.get("condition", ""))
            
            # Check A: Firing via Claims list
            fired_via_claims = False
            claims_consequent_satisfied = True
            claims_violation_reasons = []
            
            # Check if all antecedents are present in claims
            ant_claims_sets = []
            for ant in rule_antecedents:
                matched_ants = self._claims_match_antecedent(ant, claims)
                if matched_ants:
                    ant_claims_sets.append(matched_ants)
                    
            if len(ant_claims_sets) == len(rule_antecedents) and len(rule_antecedents) > 0:
                # Problem 2: Verify that all matched antecedents overlap in time to be valid simultaneously
                # Find combinations of claims (one from each antecedent matched set) that overlap
                valid_ant_combos = []
                for combo in itertools.product(*ant_claims_sets):
                    try:
                        max_start = max(normalize_to_utc(c.get("window_start_utc", "")) for c in combo)
                        min_end = min(normalize_to_utc(c.get("window_end_utc", "")) for c in combo)
                        if max_start <= min_end:
                            valid_ant_combos.append(combo)
                    except Exception:
                        continue
                
                # Rule fires via claims only if there is at least one valid overlapping combination of all antecedents
                if valid_ant_combos:
                    fired_via_claims = True
                    
                    # Verify that consequent claims are present within the expected time lag window
                    # relative to the overlapping window of the antecedents
                    for combo in valid_ant_combos:
                        try:
                            # Use the overlapping window bounds
                            ant_start = max(normalize_to_utc(c.get("window_start_utc", "")) for c in combo)
                            ant_end = min(normalize_to_utc(c.get("window_end_utc", "")) for c in combo)
                        except Exception:
                            continue
                            
                        # Target consequent start window with a tolerance of -3h to +6h
                        target_start = ant_start + pd.Timedelta(hours=lag - 3.0)
                        target_end = ant_end + pd.Timedelta(hours=lag + 6.0)
                        
                        found_satisfied_consequent = False
                        found_contradicting_consequent = False
                        
                        for claim in claims:
                            claim_var, claim_dir = self._get_claim_var_direction(claim)
                            if claim_var == cons_var:
                                try:
                                    claim_start = normalize_to_utc(claim.get("window_start_utc", ""))
                                except Exception:
                                    continue
                                    
                                if target_start <= claim_start <= target_end:
                                    if cons_dir == "neutral" or claim_dir == cons_dir:
                                        found_satisfied_consequent = True
                                    elif claim_dir != cons_dir:
                                        found_contradicting_consequent = True
                                        
                        if found_contradicting_consequent:
                            claims_consequent_satisfied = False
                            claims_violation_reasons.append(f"Consequent '{cons_var}' was explicitly contradicted in claims within lag window.")
                        elif not found_satisfied_consequent:
                            claims_consequent_satisfied = False
                            claims_violation_reasons.append(f"Consequent '{cons_var}' was absent from claims in the expected lag window (+{lag}h).")

            # Check B: Firing via Causal Chain
            fired_via_chain = False
            chain_consequent_satisfied = True
            chain_violation_reasons = []
            
            # Map antecedents to causal chain nodes
            ant_chain_nodes = []
            for ant in rule_antecedents:
                ant_var, ant_dir = self._get_rule_var_direction(ant.get("variable", ""), ant.get("condition", ""))
                # Find matching causal chain event
                for event, (v, d) in self.event_to_var_dir.items():
                    if v == ant_var and (ant_dir == "neutral" or d == ant_dir):
                        ant_chain_nodes.append(event)
                        break
                        
            # Check if all antecedent events are in the chain
            if len(ant_chain_nodes) == len(rule_antecedents) and len(rule_antecedents) > 0:
                # Problem 1: Find all indices for each antecedent node to support duplicates
                ant_indices_lists = []
                for node in ant_chain_nodes:
                    indices_list = [i for i, n in enumerate(normalized_chain) if n == node]
                    if indices_list:
                        ant_indices_lists.append(indices_list)
                        
                if len(ant_indices_lists) == len(rule_antecedents):
                    fired_via_chain = True
                    
                    # Consequent event mapping
                    cons_chain_nodes = []
                    contra_chain_nodes = []
                    for event, (v, d) in self.event_to_var_dir.items():
                        if v == cons_var:
                            if cons_dir == "neutral" or d == cons_dir:
                                cons_chain_nodes.append(event)
                            else:
                                contra_chain_nodes.append(event)
                                
                    # Get all indices of consequents and contradictions in causal chain
                    cons_indices = [i for i, n in enumerate(normalized_chain) if n in cons_chain_nodes]
                    contra_indices = [i for i, n in enumerate(normalized_chain) if n in contra_chain_nodes]
                    
                    # Verify if consequent is satisfied: there must exist a consequent index c
                    # that appears after at least one index of each antecedent
                    found_satisfied_consequent = any(
                        all(any(ant_idx < c for ant_idx in ant_indices) for ant_indices in ant_indices_lists)
                        for c in cons_indices
                    )
                    
                    # Verify if consequent is contradicted later in chain
                    found_contradicting_consequent = any(
                        all(any(ant_idx < c for ant_idx in ant_indices) for ant_indices in ant_indices_lists)
                        for c in contra_indices
                    )
                    
                    if found_contradicting_consequent:
                        chain_consequent_satisfied = False
                        chain_violation_reasons.append(f"Consequent event was contradicted later in causal chain.")
                    elif not found_satisfied_consequent:
                        chain_consequent_satisfied = False
                        chain_violation_reasons.append(f"Expected consequent event was missing from causal chain after antecedent.")

            # Combine Firing and Satisfaction Status
            fired = fired_via_claims or fired_via_chain
            satisfied = True
            
            reasons = []
            if fired_via_claims and not claims_consequent_satisfied:
                satisfied = False
                reasons.extend(claims_violation_reasons)
            if fired_via_chain and not chain_consequent_satisfied:
                satisfied = False
                reasons.extend(chain_violation_reasons)
                
            explanation = ""
            if fired:
                if satisfied:
                    explanation = f"Rule successfully fired: antecedent matched and consequent '{cons_var}' was satisfied."
                    penalty_weight = 0.0
                else:
                    explanation = f"Rule violation: antecedent fired but consequent '{cons_var}' failed checks. Reasons: " + "; ".join(reasons)
                    penalty_weight = weight
            else:
                explanation = "Rule did not fire: antecedent not found in claims or causal chain."
                penalty_weight = 0.0
                
            results.append({
                "rule_id": rule_id,
                "name": rule_name,
                "fired": fired,
                "consequent_satisfied": satisfied if fired else True,
                "penalty_weight": penalty_weight,
                "explanation": explanation,
                "weight": weight
            })
            
        # Overall Consistency Score: weighted average of fired rules
        fired_rules = [r for r in results if r["fired"]]
        if fired_rules:
            total_weight = sum(r["weight"] for r in fired_rules)
            satisfied_weight = sum(r["weight"] for r in fired_rules if r["consequent_satisfied"])
            consistency_score = float(satisfied_weight / total_weight) if total_weight > 0 else 1.0
        else:
            consistency_score = 1.0
            
        has_violations = any(r["fired"] and not r["consequent_satisfied"] for r in results)
        
        return {
            "rule_results": results,
            "rule_consistency_score": consistency_score,
            "has_violations": has_violations
        }
