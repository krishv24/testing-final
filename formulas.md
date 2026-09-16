# Formulas Used in the Hierarchical AI Meteorologist Project

All mathematical formulas and scoring equations used across the verification engine, expressed in LaTeX.

---

## 1. Data Fact Validator — [`data_validator.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/data_validator.py)

### 1.1 Fact Score (Pass Rate)

The fraction of claims that passed fact-checking:

$$
\text{fact\_score} = \frac{\text{passed\_facts}}{|\text{fact\_results}|}
$$

### 1.2 Trend Confidence (Increasing / Decreasing)

For directional trend assertions, confidence is the monotonic step ratio:

$$
\text{confidence}_{\text{trend}} = \frac{\#\{\Delta_i > 0\}}{N-1} \quad \text{(for increasing)}
$$

$$
\text{confidence}_{\text{trend}} = \frac{\#\{\Delta_i < 0\}}{N-1} \quad \text{(for decreasing)}
$$

where $\Delta_i = x_{i+1} - x_i$ is the finite difference between consecutive observations.

A trend assertion **passes** if and only if:

$$
\text{slope} > 0.001 \;\;\text{AND}\;\; \text{confidence}_{\text{trend}} \geq 0.5 \quad \text{(increasing)}
$$

$$
\text{slope} < -0.001 \;\;\text{AND}\;\; \text{confidence}_{\text{trend}} \geq 0.5 \quad \text{(decreasing)}
$$

where the slope is the linear regression coefficient from a least-squares fit:

$$
\text{slope} = \frac{N \sum_{i=1}^{N} i \cdot x_i - \left(\sum_{i=1}^{N} i\right)\left(\sum_{i=1}^{N} x_i\right)}{N \sum_{i=1}^{N} i^2 - \left(\sum_{i=1}^{N} i\right)^2}
$$

### 1.3 Category Match Confidence (Range Check)

For category-based claims (e.g., "moderate wind", "heavy precipitation"):

$$
\text{confidence}_{\text{cat}} = \frac{\#\{x_i \in [\min_{\text{cat}},\, \max_{\text{cat}}]\}}{N}
$$

Passes if $\text{confidence}_{\text{cat}} \geq 0.5$.

### 1.4 Temperature Delta (Robust Change Estimate)

For temperature drop/rise assertions with $N \geq 3$ observations:

$$
\Delta T = \overline{x_{\text{last third}}} - \overline{x_{\text{first third}}}
$$

$$
\overline{x_{\text{first third}}} = \frac{1}{k}\sum_{i=1}^{k} x_i, \quad \overline{x_{\text{last third}}} = \frac{1}{k}\sum_{i=N-k+1}^{N} x_i, \quad k = \left\lfloor \frac{N}{3} \right\rfloor
$$

For $N < 3$:

$$
\Delta T = x_N - x_1
$$

### 1.5 Threshold Exceedance Confidence

For threshold assertions (e.g., `> 25.0`, `<= 10`):

$$
\text{confidence}_{\text{thresh}} = \frac{\#\{x_i \;\text{satisfies}\; op\; v\}}{N}
$$

Passes if $\text{confidence}_{\text{thresh}} \geq 0.25$.

### 1.6 String / Categorical Match Confidence

$$
\text{confidence}_{\text{string}} = \frac{\text{matches}}{N}
$$

Passes if $\text{matches} \geq 1$.

---

## 2. Meteorological Rule Engine — [`rule_engine.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/rule_engine.py)

### 2.1 Rule Consistency Score

Weighted fraction of fired rules whose consequents are satisfied:

$$
\text{rule\_consistency\_score} = \frac{\sum_{r \in \text{fired}} w_r \cdot \mathbf{1}[\text{satisfied}_r]}{\sum_{r \in \text{fired}} w_r}
$$

where $w_r$ is the `confidence_weight` of rule $r$ from `met_rules.yaml`, and $\mathbf{1}[\text{satisfied}_r] = 1$ if the consequent was satisfied, $0$ otherwise. Returns $1.0$ if no rules fire.

### 2.2 Magnitude Guard — 50% Threshold Limit

For numeric-threshold rules, a claim must satisfy a proportionality check before firing:

- **Upward condition** (`greater_than`, `greater_than_or_equal`, etc.):

$$
\text{claim\_val} \geq 0.5 \times \text{rule\_threshold}
$$

- **Downward condition** (`less_than`, `less_than_or_equal`, etc.):

$$
\text{claim\_val} \leq 1.5 \times \text{rule\_threshold}
$$

### 2.3 Magnitude Matching with Margin

The claim value must exceed the rule threshold by at least the per-variable margin $\epsilon$:

- **Upward condition:**
$$
\text{claim\_val} \geq \text{rule\_threshold} + \epsilon
$$

- **Downward condition:**
$$
\text{claim\_val} \leq \text{rule\_threshold} - \epsilon
$$

- **Equality condition:**
$$
|\text{claim\_val} - \text{rule\_threshold}| \leq \epsilon
$$

Per-variable margins $\epsilon$:

| Variable | $\epsilon$ |
|---|---|
| Temperature / Temperature change | $1.0\,^\circ\text{C}$ |
| Relative Humidity | $5.0\,\%$ |
| Pressure (absolute) | $2.0\,\text{hPa}$ |
| Pressure change (6h / 3h) | $0.5\,\text{hPa}$ |
| Wind speed / gusts | $1.0\,\text{m/s}$ |
| Precipitation | $0.5\,\text{mm}$ |

### 2.4 Temporal Lag Window for Consequent Claims

Given antecedent window $[t_{\text{start}}, t_{\text{end}}]$ and expected lag $L$ hours:

$$
t_{\text{target\_start}} = t_{\text{start}} + (L - 3)\,\text{h}
$$

$$
t_{\text{target\_end}} = t_{\text{end}} + (L + 6)\,\text{h}
$$

The consequent claim must have its start timestamp within $[t_{\text{target\_start}},\, t_{\text{target\_end}}]$.

### 2.5 Multi-Antecedent Overlap Check

For rules with multiple antecedents $A_1, A_2, \ldots, A_k$ (each with window $[\text{start}_{A_i}, \text{end}_{A_i}]$), the rule fires only if there exists a simultaneously valid combination:

$$
t_{\text{overlap\_start}} = \max_i(\text{start}_{A_i}) \leq \min_i(\text{end}_{A_i}) = t_{\text{overlap\_end}}
$$

---

## 3. Causal Graph Validator — [`causal_graph.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/causal_graph.py)

### 3.1 Transition Confidence Score

For each valid direct edge $(u \to v)$ in the causal chain:

$$
\text{score}_{(u,v)} = \text{confidence}_{(u,v)} \in [0, 1]
$$

For an indirect path $u = p_0 \to p_1 \to \cdots \to p_k = v$ of length $k$, with path length penalty:

$$
\text{score}_{(u,v)} = \frac{1}{k} \prod_{j=0}^{k-1} \text{confidence}_{(p_j, p_{j+1})}
$$

For forbidden or unlinked transitions:

$$
\text{score}_{(u,v)} = 0.0
$$

### 3.2 Causal Graph Validity Score

Given a chain of $n$ events with $n-1$ transitions:

$$
\text{validity\_score} = \frac{1}{n-1} \sum_{i=1}^{n-1} \text{score}_{(e_i, e_{i+1})}
$$

Clamped to $[0, 1]$. If any **forbidden edge** is present:

$$
\text{validity\_score} = 0.0
$$

---

## 4. Temporal Consistency Checker — [`temporal_checker.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/temporal_checker.py)

### 4.1 Temporal Overlap Check

Two claims $c_1 = [s_1, e_1]$ and $c_2 = [s_2, e_2]$ overlap if:

$$
\max(s_1, s_2) \leq \min(e_1, e_2)
$$

### 4.2 Temporal Consistency Score

Given $P$ total cross-scale claim pairs checked (daily↔6h, 6h↔hourly, daily↔hourly), each contradiction $k$ incurs a penalty:

$$
\text{penalty}_k = \begin{cases} 1.0 & \text{if severity is high} \\ 0.5 & \text{if severity is medium} \end{cases}
$$

$$
\text{consistency\_score} = \max\!\left(0.0,\; 1 - \frac{\sum_k \text{penalty}_k}{P}\right)
$$

Returns $1.0$ if $P = 0$ (no cross-scale pairs found).

---

## 5. Confidence Scorer — [`confidence_scorer.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/confidence_scorer.py)

### 5.1 Weighted Harmonic Mean (Core Components)

The three principal components — Data Fact ($S_f$), Causal Graph ($S_c$), Temporal Consistency ($S_t$) — are combined using a **weighted harmonic mean**:

$$
S_{\text{base}} = \frac{w_f + w_c + w_t}{\dfrac{w_f}{S_f} + \dfrac{w_c}{S_c} + \dfrac{w_t}{S_t}}
$$

where:
- $w_f = 0.35$ (data fact weight)
- $w_c = 0.20$ (causal graph weight)
- $w_t = 0.20$ (temporal consistency weight)

> If **any** of $S_f$, $S_c$, or $S_t$ equals $0.0$, then $S_{\text{base}} = 0.0$ (critical failure).

### 5.2 Overall Confidence Score

The meteorological rule score $S_r$ is combined additively with the harmonic base:

$$
\text{overall\_score} = \frac{S_{\text{base}} \cdot (w_f + w_c + w_t) + S_r \cdot w_r}{(w_f + w_c + w_t) + w_r}
$$

where $w_r = 0.10$ (met rule weight).

Default weights sum to $1.0$: $w_f + w_r + w_c + w_t = 0.35 + 0.10 + 0.20 + 0.20 = 0.85$. *(The remaining 0.15 is reserved for future components.)*

---

## 6. Evaluation Tracker — [`evaluator.py`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/evaluator.py)

### 6.1 False Claim Rate

$$
\text{FCR} = \frac{\text{failed\_claims}}{|\text{total\_claims}|}
$$

### 6.2 Hallucination Precision

Fraction of failed fact claims that are corroborated by at least one rule violation or causal failure:

$$
\text{precision}_{\text{hall}} = \frac{\text{corroborated\_failures}}{\text{failed\_claims}}
$$

$$
\text{corroborated\_failures} = \begin{cases} \text{failed\_claims} & \text{if } (\text{rule violations} > 0) \lor (\text{causal failures exist}) \\ 0 & \text{otherwise} \end{cases}
$$

### 6.3 Hallucination Recall

Fraction of rule violations that are also captured by a fact-level failure:

$$
\text{recall}_{\text{hall}} = \frac{\text{recalled\_violations}}{|\text{total rule violations}|}
$$

$$
\text{recalled\_violations} = \begin{cases} |\text{total rule violations}| & \text{if failed\_claims} > 0 \\ 0 & \text{otherwise} \end{cases}
$$

### 6.4 Aggregated Summary Statistics (across $R$ runs)

$$
\overline{\text{FCR}} = \frac{1}{R} \sum_{i=1}^{R} \text{FCR}_i
$$

$$
\text{verified\_report\_rate} = \frac{\#\{\text{runs where overall\_score} \geq \theta\}}{R}
$$

$$
\text{cross\_scale\_consistency\_rate} = \frac{\#\{\text{runs with no high-severity temporal contradiction}\}}{R}
$$

$$
\overline{\text{precision}}_{\text{hall}} = \frac{1}{R} \sum_{i=1}^{R} \text{precision}_{\text{hall},i}, \quad \overline{\text{recall}}_{\text{hall}} = \frac{1}{R} \sum_{i=1}^{R} \text{recall}_{\text{hall},i}
$$

$$
\overline{S} = \frac{1}{R} \sum_{i=1}^{R} \text{overall\_score}_i
$$

where $\theta = 0.55$ is the default verification threshold.

---

## 7. Meteorological Rule Thresholds — [`met_rules.yaml`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/met_rules.yaml) & [`thresholds.yaml`](file:///c:/Users/Krish%20Vinod/Hierarchical_ai_meteorologist_/app/verification/thresholds.yaml)

The following physical threshold inequalities encode domain knowledge as if-then rules:

| Rule | Antecedent | Consequent | Lag |
|---|---|---|---|
| RULE_001 Baric Wind | $\Delta P_{6h} \leq -3.0\,\text{hPa}$ | $v_{\text{wind}} \geq 8.0\,\text{m/s}$ | 3 h |
| RULE_002 Saturated Air | $\text{RH} > 90\,\%$ | $P_{\text{rain}} > 0.0\,\text{mm}$ | 0 h |
| RULE_003 Thermal Low | $T > 35\,^\circ\text{C}$ | $\Delta P_{6h} < -1.5\,\text{hPa}$ | 2 h |
| RULE_005 Cold Frontal | $\Delta T_{3h} < -3.0\,^\circ\text{C}$ | $\Delta P_{3h} > 1.5\,\text{hPa}$ | 1 h |
| RULE_006 Sea Breeze | $T > 28\,^\circ\text{C}$ | $v_{\text{wind}} > 4.0\,\text{m/s}$ | 1 h |
| RULE_007 Cyclone Wind | $\Delta P_{6h} \leq -6.0\,\text{hPa}$ | $v_{\text{wind}} \geq 17.2\,\text{m/s}$ | 2 h |
| RULE_008 Dry Air Rain | $\text{RH} < 30\,\%$ | $P_{\text{rain}} = 0.0\,\text{mm}$ | 0 h |
| RULE_009 Warm Frontal | $\Delta T_{6h} > 3.0\,^\circ\text{C}$ | $\Delta P_{6h} < -1.5\,\text{hPa}$ | 2 h |
| RULE_010 Radiational Cooling | $\text{RH} < 40\,\%$ | $\Delta T_{6h} < -4.0\,^\circ\text{C}$ | 6 h |
| RULE_011 Radiation Fog | $\text{RH} \geq 98\,\%$ | $\text{vis} < 1000\,\text{m}$ | 2 h |
| RULE_015 High Pressure Wind | $P > 1025\,\text{hPa}$ | $v_{\text{wind}} \leq 3.4\,\text{m/s}$ | 0 h |
| RULE_016 Fog Persistence | $\text{RH} > 90\,\%$ AND $\Delta P_{6h} \in [-1.0, 1.0]$ AND $v_{\text{wind}} \leq 3.4\,\text{m/s}$ | fog persistent | 4 h |

**Physical threshold categories** from `thresholds.yaml`:

$$
v_{\text{wind}} \in [0, 3.4)\,\text{m/s} \Rightarrow \text{light}, \quad [3.4, 8.0)\,\text{m/s} \Rightarrow \text{moderate}
$$
$$
[8.0, 17.2)\,\text{m/s} \Rightarrow \text{strong}, \quad [17.2, 50.0)\,\text{m/s} \Rightarrow \text{severe}
$$

$$
P_{\text{rain}} \in [0, 2.5)\,\text{mm} \Rightarrow \text{light}, \quad [2.5, 7.6)\,\text{mm} \Rightarrow \text{moderate}
$$
$$
[7.6, 50.0)\,\text{mm} \Rightarrow \text{heavy}, \quad [50.0, 500.0)\,\text{mm} \Rightarrow \text{extreme}
$$

$$
\text{RH} \in [0, 30)\,\% \Rightarrow \text{dry}, \quad [30, 60)\,\% \Rightarrow \text{comfortable}, \quad [60, 85)\,\% \Rightarrow \text{humid}, \quad [85, 100]\,\% \Rightarrow \text{saturated}
$$

$$
\Delta P_{6h} \in [0, 1.5)\,\text{hPa} \Rightarrow \text{slight}, \quad [1.5, 3.0)\,\text{hPa} \Rightarrow \text{moderate}
$$
$$
\Delta P_{6h} \leq -3.0\,\text{hPa} \Rightarrow \text{significant drop}, \quad \Delta P_{6h} \geq 3.0\,\text{hPa} \Rightarrow \text{significant rise}
$$

$$
T \leq 0.0\,^\circ\text{C} \Rightarrow \text{extreme cold}, \quad T \geq 35.0\,^\circ\text{C} \Rightarrow \text{extreme heat}
$$

$$
|\Delta T| \geq 4.0\,^\circ\text{C} \Rightarrow \text{significant change}
$$
