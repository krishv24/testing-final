import os
import re
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from app.verification.utils import normalize_to_utc

class ValidationResult:
    def __init__(self, claim_id: str, is_pass: bool, confidence_score: float, observed_value: Any, expected_value: Any, reason: str):
        self.claim_id = claim_id
        self.is_pass = is_pass
        self.confidence_score = confidence_score
        self.observed_value = observed_value
        self.expected_value = expected_value
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "is_pass": self.is_pass,
            "confidence_score": self.confidence_score,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "reason": self.reason
        }

class DataFactValidator:
    def __init__(self, thresholds_path: str):
        self.thresholds_path = thresholds_path
        self.thresholds = {}
        self._load_thresholds()

    def _load_thresholds(self):
        if not os.path.exists(self.thresholds_path):
            raise FileNotFoundError(f"Thresholds config file not found: {self.thresholds_path}")
            
        with open(self.thresholds_path, "r", encoding="utf-8") as f:
            self.thresholds = yaml.safe_load(f)

    def _normalize_to_utc(self, series_or_val: Any) -> Any:
        """Ensures both naive and aware datetimes are converted to timezone-aware UTC."""
        return normalize_to_utc(series_or_val)


    def _get_column_name(self, variable: str, scale: str) -> str:
        """Maps user-friendly meteorological variable names to actual DataFrame column names."""
        v = str(variable).lower().strip().replace(" ", "_")
        scale = str(scale).lower().strip()
        
        if scale in ["hourly", "current"]:
            mapping = {
                "temperature": "t_c",
                "temp": "t_c",
                "apparent_temperature": "t_feel_c",
                "dew_point": "td_c",
                "dewpoint": "td_c",
                "relative_humidity": "rh_percent",
                "humidity": "rh_percent",
                "wind_speed": "wind_speed_ms",
                "wind_direction": "wind_direction_deg",
                "wind_gust": "wind_gust_ms",
                "wind_gusts": "wind_gust_ms",
                "wind_gust_speed": "wind_gust_ms",
                "precipitation": "precipitation_mm",
                "visibility": "visibility_m",
                "pressure": "pressure_hpa",
            }
            return mapping.get(v, v)
        else: # six_hour or daily
            mapping = {
                "temperature": "t_mean_c",
                "temp": "t_mean_c",
                "temperature_mean": "t_mean_c",
                "temperature_min": "t_min_c",
                "temperature_max": "t_max_c",
                "apparent_temperature": "t_feel_c",
                "relative_humidity": "rh_mean_percent",
                "humidity": "rh_mean_percent",
                "wind_speed": "wind_speed_mean_ms",
                "wind_gust": "wind_speed_mean_ms",
                "wind_gusts": "wind_speed_mean_ms",
                "wind_gust_speed": "wind_speed_mean_ms",
                "precipitation": "precipitation_sum_mm",
                "dew_point": "td_mean_c",
                "visibility": "visibility_mean_m",
                "pressure": "pressure_mean_hpa",
            }
            return mapping.get(v, v)

    def _lookup_threshold(self, variable: str, value_str: str, assertion_type: str = "") -> Dict[str, Any]:
        """Looks up a threshold value or category range from the config file, distinguishing change vs absolute temperature."""
        var_map = {
            "wind_speed": "wind_speed_ms",
            "wind_gust": "wind_speed_ms",
            "precipitation": "precipitation_mm",
            "humidity": "humidity_percent",
            "relative_humidity": "humidity_percent",
            "temperature_change": "temperature_change_c",
            "temp_change": "temperature_change_c",
            "pressure_change": "pressure_change_6h_hpa",
            "temperature": "temperature_absolute_c",
            "temp": "temperature_absolute_c"
        }
        
        var_lower = variable.lower().strip()
        assertion_lower = assertion_type.lower().strip()
        val_lower = value_str.lower().strip()
        
        config_key = None
        if var_lower in ["temperature", "temp"]:
            # Distinguish change/trend/delta or specific terms from absolute temperature threshold assertions
            if "change" in assertion_lower or "delta" in assertion_lower or "trend" in assertion_lower or "drop" in val_lower or "rise" in val_lower:
                config_key = "temperature_change_c"
            else:
                config_key = "temperature_absolute_c"
        else:
            config_key = var_map.get(var_lower)
            
        if not config_key or config_key not in self.thresholds:
            return {}
        
        category = value_str.lower().strip()
        category_data = self.thresholds[config_key].get(category)
        if category_data is not None:
            if isinstance(category_data, dict):
                return category_data
            elif isinstance(category_data, (int, float)):
                return {"val": float(category_data)}
                
        # Parse numeric expression from category string (e.g. "> 35.0" or "35.0")
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", category)
        if match:
            return {"val": float(match.group(0))}
            
        return {}

    def _parse_numeric_condition(self, cond_str: str) -> Optional[Tuple[str, Any]]:
        """Parses inequality operators and numeric values (e.g. '> 25.0', '<= 10') or ranges (e.g. '1.32 to 3.32')."""
        cond_str = cond_str.strip()
        
        # Check for range: e.g. "1.32 to 3.32" or "1007.18 to 1015.67"
        range_match = re.match(r"^([0-9.-]+)\s*(?:to|-)\s*([0-9.-]+)$", cond_str)
        if range_match:
            return "range", (float(range_match.group(1)), float(range_match.group(2)))
            
        match = re.match(r"^([<>=+\s-]*)\s*([0-9.-]+)$", cond_str)
        if match:
            op = match.group(1).replace(" ", "")
            val = float(match.group(2))
            if op == "":
                op = ">="
            return op, val
        return None

    def _validate_string_claim(self, values: np.ndarray, expected_str: str) -> Tuple[bool, float, str]:
        """Validates a claim against string/categorical observations (e.g. weather conditions)."""
        expected_str_lower = str(expected_str).lower().strip()
        # Split by common separators to support ranges/options
        options = re.split(r'\s+(?:to|or|and|-)\s+|\s*/\s*|\s*,\s*', expected_str_lower)
        options = [o.strip() for o in options if o.strip()]
        
        matches = 0
        observed_list = [str(v).lower().strip() for v in values]
        
        for obs in observed_list:
            # Check if any option is a substring of the observed value, or vice versa
            if any(opt in obs or obs in opt for opt in options):
                matches += 1
                
        confidence = float(matches / len(values)) if len(values) > 0 else 0.0
        # Pass if at least one observation in the window matches the expected condition
        is_pass = matches > 0
        observed_summary = ", ".join(set(observed_list[:5]))
        if len(set(observed_list)) > 5:
            observed_summary += "..."
            
        reason = f"String match checked: {matches}/{len(values)} values match the expected options {options}. Observed: [{observed_summary}]."
        return is_pass, confidence, reason

    def validate_all_claims(
        self,
        claims: List[Any],
        hourly_df: pd.DataFrame,
        six_hour_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        current_df: pd.DataFrame = None
    ) -> List[ValidationResult]:
        """
        Validates all meteorological claims against factual data sources.
        
        Args:
            claims: List of dicts or objects conforming to the claim schema.
            hourly_df: DataFrame of hourly weather records.
            six_hour_df: DataFrame of 6-hour weather aggregates.
            daily_df: DataFrame of daily aggregates.
            current_df: DataFrame of current observation snapshot.
            
        Returns:
            List of ValidationResult objects.
        """
        results = []
        df_map = {
            "hourly": hourly_df,
            "current": current_df if current_df is not None else hourly_df,
            "six_hour": six_hour_df,
            "daily": daily_df,
            "climatology": daily_df
        }

        for claim in claims:
            # 1. Extract claim attributes safely supporting both object and dict formats
            claim_id = getattr(claim, "claim_id", None) or claim.get("claim_id", "UNKNOWN")
            variable = getattr(claim, "variable", None) or claim.get("variable", "")
            assertion_type = getattr(claim, "assertion_type", None) or claim.get("assertion_type", "")
            start_str = getattr(claim, "window_start_utc", None) or claim.get("window_start_utc", "")
            end_str = getattr(claim, "window_end_utc", None) or claim.get("window_end_utc", "")
            threshold_or_delta = getattr(claim, "threshold_or_delta", None) or claim.get("threshold_or_delta", "")
            scale = getattr(claim, "source_scale", None) or getattr(claim, "data_scale", None) or claim.get("source_scale") or claim.get("data_scale", "hourly")

            scale_lower = str(scale).lower().strip()
            df = df_map.get(scale_lower)
            if df is None or df.empty:
                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=False,
                    confidence_score=0.0,
                    observed_value=None,
                    expected_value=threshold_or_delta,
                    reason=f"Factual DataFrame for scale '{scale}' is unavailable or empty."
                ))
                continue

            # 2. Filter rows in DataFrame to match the claim's time window with UTC conversion
            ts_col = None
            for col in ["timestamp_utc", "window_start_utc", "date_utc", "time"]:
                if col in df.columns:
                    ts_col = col
                    break
            
            # Bypass time window filtering for "current" scale since it's a single snapshot
            if ts_col and scale_lower != "current":
                try:
                    df_dates = self._normalize_to_utc(df[ts_col])
                    start_dt = self._normalize_to_utc(start_str)
                    end_dt = self._normalize_to_utc(end_str)
                    mask = (df_dates >= start_dt) & (df_dates <= end_dt)
                    filtered_df = df[mask]
                except Exception as e:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=None,
                        expected_value=threshold_or_delta,
                        reason=f"Date parsing or filtering failed: {e}"
                    ))
                    continue
            else:
                filtered_df = df

            if filtered_df.empty:
                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=False,
                    confidence_score=0.0,
                    observed_value=None,
                    expected_value=threshold_or_delta,
                    reason=f"No observations found in window [{start_str}, {end_str}] for scale '{scale}'."
                ))
                continue

            # 3. Resolve the target column in the DataFrame
            col_name = self._get_column_name(variable, scale_lower)
            if col_name not in filtered_df.columns:
                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=False,
                    confidence_score=0.0,
                    observed_value=None,
                    expected_value=threshold_or_delta,
                    reason=f"Column '{col_name}' representing variable '{variable}' not found in DataFrame."
                ))
                continue

            # Check if column dtype is numeric. If not, run string matching validation.
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                values = filtered_df[col_name].dropna().values
                if len(values) == 0:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=None,
                        expected_value=threshold_or_delta,
                        reason=f"Target column '{col_name}' contains only null values in the filtered window."
                    ))
                    continue
                is_pass, confidence, string_reason = self._validate_string_claim(values, str(threshold_or_delta))
                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=is_pass,
                    confidence_score=confidence,
                    observed_value=", ".join(set(str(v) for v in values)),
                    expected_value=threshold_or_delta,
                    reason=string_reason
                ))
                continue

            # Extract numeric series
            raw_series = filtered_df[col_name].dropna()
            numeric_series = pd.to_numeric(raw_series, errors='coerce').dropna()
            values = numeric_series.values
            if len(values) == 0:
                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=False,
                    confidence_score=0.0,
                    observed_value=None,
                    expected_value=threshold_or_delta,
                    reason=f"Target column '{col_name}' contains no numeric values in the filtered window."
                ))
                continue

            # 4. Perform assertion validation
            assertion_lower = str(assertion_type).lower().strip()
            
            # CASE A: Directional Trends (increasing / decreasing)
            if assertion_lower in ["increasing", "decreasing", "trend"]:
                if len(values) < 2:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=f"Single value: {values[0]}",
                        expected_value=assertion_lower,
                        reason=f"Cannot determine trend consistency with fewer than 2 data points (got {len(values)})."
                    ))
                    continue
                
                # Check trend consistency using linear regression slope and monotonic step ratio
                slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
                diffs = np.diff(values)
                
                if assertion_lower == "increasing" or (assertion_lower == "trend" and "inc" in str(threshold_or_delta).lower()):
                    matching_steps = sum(diffs > 0)
                    total_steps = len(diffs)
                    confidence = float(matching_steps / total_steps)
                    # Pass if slope is positive and at least 50% of consecutive steps are increasing
                    is_pass = (slope > 0.001) and (confidence >= 0.5)
                    observed = f"Slope: {slope:.4f}, Monotonic positive steps: {matching_steps}/{total_steps}"
                    expected = "Positive slope (>0) and monotonic step ratio >= 0.5"
                    reason = f"Trend checked: observed slope is {slope:.4f} with {matching_steps}/{total_steps} positive steps ({confidence*100:.1f}%)."
                else: # decreasing
                    matching_steps = sum(diffs < 0)
                    total_steps = len(diffs)
                    confidence = float(matching_steps / total_steps)
                    is_pass = (slope < -0.001) and (confidence >= 0.5)
                    observed = f"Slope: {slope:.4f}, Monotonic negative steps: {matching_steps}/{total_steps}"
                    expected = "Negative slope (<0) and monotonic step ratio >= 0.5"
                    reason = f"Trend checked: observed slope is {slope:.4f} with {matching_steps}/{total_steps} negative steps ({confidence*100:.1f}%)."

                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=is_pass,
                    confidence_score=confidence,
                    observed_value=observed,
                    expected_value=expected,
                    reason=reason
                ))

            # CASE B: Category Matches or Comparison to Normals
            elif assertion_lower in ["category_match", "comparison_to_normal", "comparison"]:
                threshold_cfg = self._lookup_threshold(variable, str(threshold_or_delta), assertion_type)
                
                if not threshold_cfg:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=None,
                        expected_value=threshold_or_delta,
                        reason=f"Could not resolve category '{threshold_or_delta}' for variable '{variable}' in thresholds.yaml."
                    ))
                    continue

                if "min" in threshold_cfg and "max" in threshold_cfg:
                    # Range check
                    min_val = float(threshold_cfg["min"])
                    max_val = float(threshold_cfg["max"])
                    matching_count = sum((values >= min_val) & (values <= max_val))
                    confidence = float(matching_count / len(values))
                    is_pass = confidence >= 0.5
                    
                    mean_val = float(np.mean(values))
                    observed = f"Mean: {mean_val:.2f}, Range: [{np.min(values):.2f}, {np.max(values):.2f}]"
                    expected = f"[{min_val}, {max_val}]"
                    reason = f"Category match checked: {matching_count}/{len(values)} values ({confidence*100:.1f}%) fall in range [{min_val}, {max_val}]."
                elif "val" in threshold_cfg:
                    # Threshold check (e.g. for significant temperature drop/rise)
                    expected_val = float(threshold_cfg["val"])
                    
                    # Robust delta: difference between window mean of last third and first third
                    n = len(values)
                    if n >= 3:
                        k = n // 3
                        first_third_mean = float(np.mean(values[:k]))
                        last_third_mean = float(np.mean(values[-k:]))
                        delta = last_third_mean - first_third_mean
                        observed_str = f"Delta (mean of last third - first third): {delta:.2f} (first third: {first_third_mean:.2f}, last third: {last_third_mean:.2f})"
                    else:
                        delta = float(values[-1] - values[0])
                        observed_str = f"Delta (end-start): {delta:.2f}"
                    
                    if "drop" in str(threshold_or_delta).lower():
                        is_pass = delta <= expected_val
                        confidence = 1.0 if is_pass else 0.0
                        observed = observed_str
                        expected = f"<= {expected_val}"
                        reason = f"Temperature drop checked: observed change is {delta:.2f}°C (required <= {expected_val}°C)."
                    elif "rise" in str(threshold_or_delta).lower():
                        is_pass = delta >= expected_val
                        confidence = 1.0 if is_pass else 0.0
                        observed = observed_str
                        expected = f">= {expected_val}"
                        reason = f"Temperature rise checked: observed change is {delta:.2f}°C (required >= {expected_val}°C)."
                    else:
                        # Direct value check (e.g. absolute thresholds)
                        matching_count = sum(values >= expected_val) if expected_val >= 0 else sum(values <= expected_val)
                        confidence = float(matching_count / len(values))
                        is_pass = confidence >= 0.5
                        observed = f"Mean: {np.mean(values):.2f}"
                        expected = f">= {expected_val}" if expected_val >= 0 else f"<= {expected_val}"
                        reason = f"Absolute value checked: {matching_count}/{len(values)} values ({confidence*100:.1f}%) match condition."
                else:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=None,
                        expected_value=threshold_or_delta,
                        reason=f"Parsed category config '{threshold_cfg}' was empty or malformed."
                    ))
                    continue

                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=is_pass,
                    confidence_score=confidence,
                    observed_value=observed,
                    expected_value=expected,
                    reason=reason
                ))

            # CASE C: Threshold Exceeded / Numeric Condition
            else:  # e.g. "exceeds_threshold", "less_than", or generic
                parsed_cond = self._parse_numeric_condition(str(threshold_or_delta))
                if not parsed_cond:
                    results.append(ValidationResult(
                        claim_id=claim_id,
                        is_pass=False,
                        confidence_score=0.0,
                        observed_value=None,
                        expected_value=threshold_or_delta,
                        reason=f"Could not parse numeric operator and value from '{threshold_or_delta}'."
                    ))
                    continue

                op, expected_val = parsed_cond
                
                # Check condition fraction
                if op == "range":
                    min_val, max_val = expected_val
                    matching_count = sum((values >= min_val) & (values <= max_val))
                    confidence = float(matching_count / len(values))
                    is_pass = confidence >= 0.25
                    observed = f"Range: [{np.min(values):.2f}, {np.max(values):.2f}]"
                    expected = f"[{min_val}, {max_val}]"
                    reason = f"Range checked: {matching_count}/{len(values)} values ({confidence*100:.1f}%) fall in range [{min_val}, {max_val}]."
                elif op == ">":
                    matching_count = sum(values > expected_val)
                elif op == ">=":
                    matching_count = sum(values >= expected_val)
                elif op == "<":
                    matching_count = sum(values < expected_val)
                elif op == "<=":
                    matching_count = sum(values <= expected_val)
                else: # "=="
                    matching_count = sum(np.isclose(values, expected_val))

                if op != "range":
                    confidence = float(matching_count / len(values))
                    # Pass if the threshold was met in at least 25% of the window
                    is_pass = confidence >= 0.25
                    observed = f"Peak: {np.max(values) if '>' in op else np.min(values)}, Count: {matching_count}/{len(values)}"
                    expected = f"{op} {expected_val}"
                    reason = f"Threshold checked: condition '{op} {expected_val}' met in {matching_count}/{len(values)} observations ({confidence*100:.1f}% of window)."

                results.append(ValidationResult(
                    claim_id=claim_id,
                    is_pass=is_pass,
                    confidence_score=confidence,
                    observed_value=observed,
                    expected_value=expected,
                    reason=reason
                ))

        return results
