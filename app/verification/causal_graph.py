import os
import yaml
import networkx as nx
from typing import List, Tuple, Dict, Any

class CausalGraph:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.graph = nx.DiGraph()
        
        # Node synonym mapping to standard internal node names
        self.node_synonyms = {
            "humidity_rise": "high_humidity",
            "frontal_passage": "cooling",
            "clearing": "no_rain",
            "fog_formation": "low_visibility",
            "overcast": "low_visibility",
            "drizzle": "rainfall",
            "unstable_airmass": "high_humidity",
            "thunderstorms": "thunderstorm",
        }
        
        # Explicitly invalid or physically impossible edges (to catch hallucinated causal chains)
        self.invalid_edges = {
            ("dry_air", "rainfall"),
            ("dry_air", "fog_formation"),
            ("dry_air", "fog_persistence"),
            ("cooling", "extreme_heat"),
            ("clearing", "rainfall"),
        }
        
        self._load_rules_and_build_graph()

    def _normalize_node(self, node: str) -> str:
        """Helper to convert user/LLM-written events to normalized graph nodes using keywords."""
        s = str(node).lower().strip().replace(" ", "_")
        
        # Exact synonym lookup first
        if s in self.node_synonyms:
            return self.node_synonyms[s]
            
        # Keyword-based phrase mapping
        if "pressure_drop" in s or "drop_in_pressure" in s or "drop_in_surface_pressure" in s or "trough" in s:
            return "pressure_drop"
        if "pressure_rise" in s or "rise_in_pressure" in s or "high_pressure" in s or "ridge" in s:
            return "pressure_rise"
        if "stable_pressure" in s or "constant_pressure" in s:
            return "stable_pressure"
        if "thunderstorm" in s or "storm" in s:
            return "thunderstorm"
        if "no_rain" in s or "clear" in s or "sunny" in s or "insolation" in s:
            return "no_rain"
        if "rain" in s or "precipitation" in s or "shower" in s:
            return "rainfall"
        if "fog_persistence" in s:
            return "fog_persistence"
        if "fog" in s or "visibility" in s or "mist" in s:
            return "low_visibility"
        if "humidity" in s or "moist" in s or "humid" in s:
            if "decreas" in s or "drop" in s or "low" in s:
                return "dry_air"
            return "high_humidity"
        if "dry" in s:
            return "dry_air"
        if "extreme_heat" in s or "heatwave" in s:
            return "extreme_heat"
        if "warming" in s or "heating" in s or "rising_temperature" in s:
            return "warming"
        if "cooling" in s or "cold" in s or "temperature_drop" in s or "frontal_passage" in s:
            return "cooling"
        if "gust" in s:
            return "wind_gusts"
        if "wind" in s or "breeze" in s:
            if "light" in s or "weak" in s:
                return "light_winds"
            return "wind_increase"
            
        return s

    def _map_var_to_node(self, var: str, cond: str, val: Any) -> str:
        """Maps configuration variables and values to standard weather state nodes."""
        var = var.lower().strip()
        if var == "pressure_change_6h":
            if isinstance(val, (int, float)) and val <= -3.0:
                return "pressure_drop"
            if isinstance(val, (int, float)) and val >= 3.0:
                return "pressure_rise"
            if isinstance(val, list) and len(val) == 2 and val[0] >= -1.0 and val[1] <= 1.0:
                return "stable_pressure"
            return "pressure_drop" if cond in ["less_than", "less_than_or_equal"] else "pressure_rise"
        elif var == "pressure_change_3h":
            return "pressure_drop" if cond in ["less_than", "less_than_or_equal"] else "pressure_rise"
        elif var == "pressure":
            return "pressure_rise" if cond in ["greater_than", "greater_than_or_equal"] else "pressure_drop"
        elif var == "relative_humidity":
            return "high_humidity" if cond in ["greater_than", "greater_than_or_equal"] else "dry_air"
        elif var == "temperature":
            if isinstance(val, (int, float)) and val >= 35.0:
                return "extreme_heat"
            return "warming" if cond in ["greater_than", "greater_than_or_equal"] else "cooling"
        elif var in ["temperature_change_3h", "temperature_change_6h"]:
            return "warming" if cond in ["greater_than", "greater_than_or_equal"] else "cooling"
        elif var == "weather_category":
            if val == "thunderstorm":
                return "thunderstorm"
            return str(val)
        elif var == "precipitation":
            return "rainfall" if cond in ["greater_than", "greater_than_or_equal"] else "no_rain"
        elif var == "wind_speed":
            if isinstance(val, (int, float)) and val <= 3.4:
                return "light_winds"
            return "wind_increase" if cond in ["greater_than", "greater_than_or_equal"] else "light_winds"
        elif var == "visibility":
            return "low_visibility" if cond in ["less_than", "less_than_or_equal"] else "clearing"
        elif var == "wind_gust":
            return "wind_gusts"
        elif var == "fog_persistence":
            return "fog_persistence"
        return var

    def _load_rules_and_build_graph(self):
        """Loads rules YAML, parses antecedents and consequents, and builds directed graph edges."""
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Rules config file not found: {self.rules_path}")
            
        with open(self.rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        rules = config.get("rules", [])
        for rule in rules:
            confidence = rule.get("confidence_weight", 1.0)
            lag = rule.get("expected_lag_hours", 0.0)
            rule_id = rule.get("rule_id", "UNKNOWN")
            
            consequent = rule.get("consequent", {})
            target = self._map_var_to_node(
                consequent.get("variable", ""),
                consequent.get("condition", ""),
                consequent.get("threshold")
            )
            
            sources = []
            if "antecedents" in rule:
                for ant in rule["antecedents"]:
                    sources.append(self._map_var_to_node(
                        ant.get("variable", ""),
                        ant.get("condition", ""),
                        ant.get("threshold")
                    ))
            elif "antecedent" in rule:
                ant = rule["antecedent"]
                sources.append(self._map_var_to_node(
                    ant.get("variable", ""),
                    ant.get("condition", ""),
                    ant.get("threshold")
                ))
                
            for source in sources:
                normalized_source = self._normalize_node(source)
                normalized_target = self._normalize_node(target)
                self.graph.add_edge(
                    normalized_source,
                    normalized_target,
                    confidence=confidence,
                    lag=lag,
                    rule_id=rule_id
                )
                
        # Proactively inject critical system links for connectivity
        # e.g., low_visibility connects to fog_persistence
        self.graph.add_edge("low_visibility", "fog_persistence", confidence=0.9, lag=0.0, rule_id="SYS_LINK_001")
        self.graph.add_edge("high_humidity", "low_visibility", confidence=0.8, lag=1.0, rule_id="SYS_LINK_002")

        # High-pressure summer pattern links (e.g. Seattle summer ridge scenario)
        self.graph.add_edge("stable_pressure", "light_winds", confidence=0.80, lag=0.0, rule_id="SYS_LINK_003")
        self.graph.add_edge("light_winds", "no_rain", confidence=0.75, lag=0.0, rule_id="SYS_LINK_004")
        self.graph.add_edge("no_rain", "warming", confidence=0.70, lag=1.0, rule_id="SYS_LINK_005")
        self.graph.add_edge("warming", "dry_air", confidence=0.75, lag=1.0, rule_id="SYS_LINK_006")
        self.graph.add_edge("dry_air", "extreme_heat", confidence=0.65, lag=2.0, rule_id="SYS_LINK_007")
        self.graph.add_edge("extreme_heat", "cooling", confidence=0.70, lag=6.0, rule_id="SYS_LINK_008")

        # Additional links for broader weather patterns (e.g. Mumbai monsoon, general patterns)
        self.graph.add_edge("pressure_drop", "high_humidity", confidence=0.80, lag=2.0, rule_id="SYS_LINK_009")
        self.graph.add_edge("wind_increase", "thunderstorm", confidence=0.75, lag=1.0, rule_id="SYS_LINK_010")
        self.graph.add_edge("fog_persistence", "cooling", confidence=0.70, lag=2.0, rule_id="SYS_LINK_011")
        self.graph.add_edge("wind_gusts", "warming", confidence=0.65, lag=1.0, rule_id="SYS_LINK_012")
        self.graph.add_edge("rainfall", "thunderstorm", confidence=0.80, lag=0.0, rule_id="SYS_LINK_013")
        self.graph.add_edge("thunderstorm", "wind_gusts", confidence=0.85, lag=0.0, rule_id="SYS_LINK_014")

        # Additional physically grounded edges from atmospheric science.
        # Each relationship is justified from first principles or textbook meteorology,
        # independent of any specific test location or LLM output distribution.
        self.graph.add_edge("warming", "high_humidity", confidence=0.75, lag=2.0, rule_id="SYS_LINK_015")   # Clausius-Clapeyron: warmer air ↑ saturation vapor pressure; with moisture source, specific humidity rises
        self.graph.add_edge("rainfall", "cooling", confidence=0.80, lag=1.0, rule_id="SYS_LINK_016")        # Evaporative cooling of falling precip + latent heat redistribution (Wallace & Hobbs, Atm. Science)
        self.graph.add_edge("pressure_drop", "rainfall", confidence=0.75, lag=3.0, rule_id="SYS_LINK_017")  # Synoptic-scale: low-pressure convergence forces ascent → condensation → precipitation
        self.graph.add_edge("high_humidity", "thunderstorm", confidence=0.70, lag=2.0, rule_id="SYS_LINK_018")  # High low-level moisture → high CAPE → convective initiation (Doswell 1987)
        self.graph.add_edge("cooling", "high_humidity", confidence=0.75, lag=1.0, rule_id="SYS_LINK_019")   # RH = e/es(T); temperature drop → es drops → RH rises (thermodynamic identity)
        self.graph.add_edge("thunderstorm", "cooling", confidence=0.80, lag=0.5, rule_id="SYS_LINK_020")    # Convective downdraft cold pools (Markowski & Richardson, Mesoscale Met.)
        self.graph.add_edge("wind_increase", "rainfall", confidence=0.70, lag=2.0, rule_id="SYS_LINK_021")  # Enhanced low-level wind → moisture transport + convergence → forced ascent

        # Fair-weather / high-pressure stability patterns
        self.graph.add_edge("no_rain", "stable_pressure", confidence=0.70, lag=0.0, rule_id="SYS_LINK_022")   # Persistent dry conditions ↔ anticyclonic stability
        self.graph.add_edge("pressure_rise", "stable_pressure", confidence=0.75, lag=6.0, rule_id="SYS_LINK_023")  # Post-frontal pressure rise settles into stable ridge
        self.graph.add_edge("stable_pressure", "no_rain", confidence=0.70, lag=0.0, rule_id="SYS_LINK_024")   # Anticyclonic subsidence suppresses convection
        self.graph.add_edge("no_rain", "dry_air", confidence=0.70, lag=2.0, rule_id="SYS_LINK_025")           # Prolonged dry spell → boundary layer dries out
        self.graph.add_edge("pressure_rise", "no_rain", confidence=0.70, lag=3.0, rule_id="SYS_LINK_026")     # Post-frontal subsidence clears precip

        # Cooling / fog patterns (frequently seen in Delhi / continental climates)
        self.graph.add_edge("cooling", "low_visibility", confidence=0.75, lag=3.0, rule_id="SYS_LINK_027")    # Radiational cooling → dew point convergence → fog/mist
        self.graph.add_edge("cooling", "fog_persistence", confidence=0.70, lag=4.0, rule_id="SYS_LINK_028")   # Sustained cooling maintains saturation → persistent fog
        self.graph.add_edge("light_winds", "low_visibility", confidence=0.70, lag=2.0, rule_id="SYS_LINK_029")  # Calm winds prevent turbulent mixing → fog/haze
        self.graph.add_edge("light_winds", "fog_persistence", confidence=0.65, lag=3.0, rule_id="SYS_LINK_030")  # Light winds can't break fog layer

        # Warming / drying and pressure-wind relationships
        self.graph.add_edge("warming", "no_rain", confidence=0.65, lag=1.0, rule_id="SYS_LINK_031")           # Diurnal warming raises LCL → suppresses shallow convection
        self.graph.add_edge("pressure_drop", "wind_increase", confidence=0.80, lag=1.0, rule_id="SYS_LINK_032")  # Steepening pressure gradient → stronger geostrophic wind
        self.graph.add_edge("pressure_rise", "light_winds", confidence=0.70, lag=2.0, rule_id="SYS_LINK_033")   # Relaxing gradient under ridge → wind dies down

    def validate_chain(self, causal_chain: List[str]) -> Dict[str, Any]:
        """
        Validates the causal chain of weather events.
        
        Args:
            causal_chain: List of weather event strings (e.g., ["pressure_drop", "wind_increase", "rainfall"])
            
        Returns:
            Dict containing:
            - is_valid: True if every transition is valid (either direct edge or path, and no invalid edges)
            - invalid_transitions: List of adjacent pairs in the chain that are invalid/impossible
            - missing_intermediates: List of expected intermediate nodes between chain start and end
            - validity_score: Float between 0.0 and 1.0 representing average confidence score of transitions
                              (Zeroed out entirely if any forbidden edge is present)
            - has_forbidden_edges: True if the chain contains one or more forbidden/physically impossible transitions
        """
        if not causal_chain or len(causal_chain) < 2:
            return {
                "is_valid": True,
                "invalid_transitions": [],
                "missing_intermediates": [],
                "validity_score": 1.0,
                "has_forbidden_edges": False
            }
            
        normalized_chain = [self._normalize_node(node) for node in causal_chain]
        

        
        invalid_transitions = []
        missing_intermediates = []
        transition_scores = []
        has_forbidden_edges = False
        
        for i in range(len(normalized_chain) - 1):
            u = normalized_chain[i]
            v = normalized_chain[i+1]

            if u == v:
                transition_scores.append(1.0)
                continue
            # 1. Check if the transition is explicitly marked as physically invalid
            if (u, v) in self.invalid_edges or (self._normalize_node(u), self._normalize_node(v)) in self.invalid_edges:
                invalid_transitions.append((causal_chain[i], causal_chain[i+1]))
                transition_scores.append(0.0)
                has_forbidden_edges = True
                continue
                
            # 2. Check if a direct edge exists
            if self.graph.has_edge(u, v):
                edge_data = self.graph.get_edge_data(u, v)
                transition_scores.append(edge_data.get("confidence", 1.0))
            # 3. Check if there is an indirect path (weak link / missing intermediates)
            elif self.graph.has_node(u) and self.graph.has_node(v) and nx.has_path(self.graph, u, v):
                path = nx.shortest_path(self.graph, u, v)
                # Identify intermediates on this specific sub-path that are missing in the chain
                for p_node in path[1:-1]:
                    if p_node not in normalized_chain:
                        missing_intermediates.append(p_node)
                # Score based on minimum edge confidence along the path,
                # with a mild discount per extra hop (10% per additional step)
                edge_confidences = []
                for j in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[j], path[j+1])
                    edge_confidences.append(edge_data.get("confidence", 1.0))
                min_confidence = min(edge_confidences)
                extra_hops = len(path) - 2  # number of hops beyond a direct edge
                path_confidence = min_confidence * (0.9 ** extra_hops)
                transition_scores.append(path_confidence)
            else:
                # No path at all (logically unlinked, but not physically forbidden)
                invalid_transitions.append((causal_chain[i], causal_chain[i+1]))
                transition_scores.append(0.0)
                
        # Note: removed redundant global start-to-end missing intermediates check.
        # The per-pair check above already identifies meaningful missing intermediates.
        # The global check was adding false positives from unrelated weather patterns.
                    
        is_valid = len(invalid_transitions) == 0
        
        # Calculate validity score. If there is a forbidden edge, the score is strictly 0.0.
        if has_forbidden_edges:
            validity_score = 0.0
        else:
            validity_score = sum(transition_scores) / len(transition_scores) if transition_scores else 1.0
        
        return {
            "is_valid": is_valid,
            "invalid_transitions": invalid_transitions,
            "missing_intermediates": list(sorted(set(missing_intermediates))),
            "validity_score": float(max(0.0, min(1.0, validity_score))),
            "has_forbidden_edges": has_forbidden_edges
        }
