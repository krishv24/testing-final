"""Quick analysis of the Supabase CSV for the report."""
import csv
import sys
csv.field_size_limit(2**30)
import json
from collections import defaultdict
from pathlib import Path

csv_path = Path(r"Supabase Snippet Untitled query(3).csv")

rows = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")
print(f"Columns: {columns}")
print()

# Filter out rows with errors
valid = [r for r in rows if not r.get('error', '').strip() or r.get('error', '').strip().lower() == 'null']
print(f"Valid (no error) rows: {len(valid)}")

# Parse numeric fields
def safe_float(v, default=0.0):
    try:
        return float(v) if v else default
    except:
        return default

# Aggregate scores
scores = {
    'overall_score': [],
    'fact_score': [],
    'rule_score': [],
    'causal_score': [],
    'temporal_score': [],
    'false_claim_rate': [],
    'hallucination_precision': [],
    'hallucination_recall': [],
    'total_pipeline_ms': [],
}

for r in valid:
    for k in scores:
        scores[k].append(safe_float(r.get(k)))

print("=== AGGREGATE SCORES ===")
for k, vals in scores.items():
    if vals:
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"  {k:30s}  avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}")

# By climate zone
by_cz = defaultdict(list)
for r in valid:
    cz = r.get('climate_zone', 'unknown') or 'unknown'
    by_cz[cz].append(safe_float(r.get('overall_score')))

print("\n=== BY CLIMATE ZONE ===")
for cz, vals in sorted(by_cz.items(), key=lambda x: -len(x[1])):
    avg = sum(vals)/len(vals)
    print(f"  {cz:40s}  n={len(vals):4d}  avg={avg:.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

# By tone
by_tone = defaultdict(list)
for r in valid:
    tone = r.get('tone', 'unknown') or 'unknown'
    by_tone[tone].append(safe_float(r.get('overall_score')))

print("\n=== BY TONE ===")
for t, vals in sorted(by_tone.items()):
    avg = sum(vals)/len(vals)
    print(f"  {t:25s}  n={len(vals):4d}  avg={avg:.4f}")

# By domain
by_dom = defaultdict(list)
for r in valid:
    dom = r.get('domain', 'unknown') or 'unknown'
    by_dom[dom].append(safe_float(r.get('overall_score')))

print("\n=== BY DOMAIN ===")
for d, vals in sorted(by_dom.items()):
    avg = sum(vals)/len(vals)
    print(f"  {d:25s}  n={len(vals):4d}  avg={avg:.4f}")

# By model
by_model = defaultdict(list)
for r in valid:
    m = r.get('gemini_model', 'unknown') or 'unknown'
    by_model[m].append(safe_float(r.get('overall_score')))

print("\n=== BY GEMINI MODEL ===")
for m, vals in sorted(by_model.items()):
    avg = sum(vals)/len(vals)
    print(f"  {m:35s}  n={len(vals):4d}  avg={avg:.4f}")

# Verified report rate
verified_count = sum(1 for r in valid if r.get('verified_report', '').lower() == 'true')
print(f"\n=== VERIFIED REPORT RATE ===")
print(f"  Verified: {verified_count}/{len(valid)} = {verified_count/len(valid)*100:.1f}%")

# Cross scale consistency
css_count = sum(1 for r in valid if r.get('cross_scale_consistent', '').lower() == 'true')
print(f"  Cross-scale consistent: {css_count}/{len(valid)} = {css_count/len(valid)*100:.1f}%")

# Correction needed
corr_count = sum(1 for r in valid if r.get('correction_needed', '').lower() == 'true')
print(f"  Correction needed: {corr_count}/{len(valid)} = {corr_count/len(valid)*100:.1f}%")

# By location
by_loc = defaultdict(list)
for r in valid:
    loc = r.get('location_name', 'unknown') or 'unknown'
    by_loc[loc].append(safe_float(r.get('overall_score')))

print(f"\n=== UNIQUE LOCATIONS: {len(by_loc)} ===")
for loc, vals in sorted(by_loc.items(), key=lambda x: -sum(x[1])/len(x[1]))[:15]:
    avg = sum(vals)/len(vals)
    print(f"  {loc:30s}  n={len(vals):4d}  avg={avg:.4f}")

# Total claims stats
total_claims = [safe_float(r.get('total_claims')) for r in valid]
failed_claims = [safe_float(r.get('failed_claims')) for r in valid]
rule_violations = [safe_float(r.get('rule_violations')) for r in valid]

print(f"\n=== CLAIMS STATS ===")
print(f"  Total claims per run (avg): {sum(total_claims)/len(total_claims):.2f}")
print(f"  Failed claims per run (avg): {sum(failed_claims)/len(failed_claims):.2f}")
print(f"  Rule violations per run (avg): {sum(rule_violations)/len(rule_violations):.2f}")

# Pipeline timing
pipeline_ms = [safe_float(r.get('total_pipeline_ms')) for r in valid if safe_float(r.get('total_pipeline_ms')) > 0]
print(f"\n=== PIPELINE TIMING ===")
print(f"  Avg pipeline time: {sum(pipeline_ms)/len(pipeline_ms):.0f} ms ({sum(pipeline_ms)/len(pipeline_ms)/1000:.1f} s)")
print(f"  Min: {min(pipeline_ms):.0f} ms, Max: {max(pipeline_ms):.0f} ms")

# Context style
by_cs = defaultdict(list)
for r in valid:
    cs = r.get('context_style', 'unknown') or 'unknown'
    by_cs[cs].append(safe_float(r.get('overall_score')))

print("\n=== BY CONTEXT STYLE ===")
for cs, vals in sorted(by_cs.items()):
    avg = sum(vals)/len(vals)
    print(f"  {cs:25s}  n={len(vals):4d}  avg={avg:.4f}")

# Per-component score breakdown across all runs
print("\n=== COMPONENT SCORE DISTRIBUTIONS ===")
for comp in ['fact_score', 'rule_score', 'causal_score', 'temporal_score']:
    vals = scores[comp]
    # Percentage with perfect score
    perfect = sum(1 for v in vals if v >= 0.999) / len(vals) * 100
    # Percentage with zero
    zero = sum(1 for v in vals if v < 0.001) / len(vals) * 100
    print(f"  {comp:25s}  perfect={perfect:.1f}%  zero={zero:.1f}%")
