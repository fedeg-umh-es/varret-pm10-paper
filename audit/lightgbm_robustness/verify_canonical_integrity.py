"""
Canonical Evidence Integrity Tests — Paper A / P4
===================================================
Verifies that the canonical empirical table satisfies all contracts:
- 595 rows, 0 lightgbm rows, 5 models, 17 stations, 7 horizons, 0 duplicate keys
- Core results (277/8/269/97.1%/rho=-0.863/101) trace to canonical table only
- No default script path references the contaminated synthetic table

Run: python3 audit/lightgbm_robustness/verify_canonical_integrity.py
"""
import json
import hashlib
import subprocess
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

REPO = Path('/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper')
CANONICAL = REPO / 'outputs' / 'tables' / 'master_diagnostic_table.csv'
SYNTHETIC_MOVED = REPO / 'audit' / 'lightgbm_robustness' / 'synthetic_outputs' / 'master_diagnostic_table_with_lightgbm_SYNTHETIC.csv'
SYNTHETIC_OLD = REPO / 'outputs' / 'analysis' / 'master_diagnostic_table_with_lightgbm.csv'
CANONICAL_SHA = '6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6'

checks = {}

# ── 1. CANONICAL TABLE STRUCTURE ────────────────────────────────────────────
df = pd.read_csv(CANONICAL)

checks['canonical_rows_595'] = {
    'expected': 595, 'computed': len(df),
    'status': 'PASS' if len(df) == 595 else 'FAIL'
}

lgb_in_canonical = (df['model'] == 'lightgbm_direct').sum()
checks['lightgbm_rows_in_canonical_0'] = {
    'expected': 0, 'computed': int(lgb_in_canonical),
    'status': 'PASS' if lgb_in_canonical == 0 else 'FAIL'
}

n_models = df['model'].nunique()
checks['canonical_models_5'] = {
    'expected': 5, 'computed': int(n_models),
    'status': 'PASS' if n_models == 5 else 'FAIL'
}

n_stations = df['station_id'].nunique()
checks['stations_17'] = {
    'expected': 17, 'computed': int(n_stations),
    'status': 'PASS' if n_stations == 17 else 'FAIL'
}

n_horizons = df['horizon'].nunique()
checks['horizons_7'] = {
    'expected': 7, 'computed': int(n_horizons),
    'status': 'PASS' if n_horizons == 7 else 'FAIL'
}

dup_keys = int(df.duplicated(subset=['station_id', 'model', 'horizon']).sum())
checks['duplicate_keys_0'] = {
    'expected': 0, 'computed': dup_keys,
    'status': 'PASS' if dup_keys == 0 else 'FAIL'
}

# ── 2. SHA-256 OF CANONICAL TABLE ───────────────────────────────────────────
actual_sha = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
checks['canonical_sha256'] = {
    'expected': CANONICAL_SHA,
    'computed': actual_sha,
    'status': 'PASS' if actual_sha == CANONICAL_SHA else 'FAIL'
}

# ── 3. CORE RESULTS ─────────────────────────────────────────────────────────
rule_a = (df['skill'] > 0) & (df['dm_significant'] == True)
rule_b = rule_a & (df['alpha'] >= 0.50) & (df['recall_p75'] >= 0.20)
n_rule_a = int(rule_a.sum())
n_rule_b = int(rule_b.sum())
n_change = n_rule_a - n_rule_b
pct = n_change / n_rule_a * 100 if n_rule_a > 0 else 0

rho, _ = spearmanr(df['alpha'], df['skill'])
discordant = int((rule_a & (df['recall_p75'] >= 0.20) & (df['alpha'] < 0.50)).sum())

checks['rule_a_277'] = {'expected': 277, 'computed': n_rule_a, 'status': 'PASS' if n_rule_a == 277 else 'FAIL'}
checks['rule_b_8'] = {'expected': 8, 'computed': n_rule_b, 'status': 'PASS' if n_rule_b == 8 else 'FAIL'}
checks['changes_269'] = {'expected': 269, 'computed': n_change, 'status': 'PASS' if n_change == 269 else 'FAIL'}
checks['pct_971'] = {'expected': '97.1%', 'computed': f'{pct:.1f}%', 'status': 'PASS' if abs(pct - 97.1) < 0.05 else 'FAIL'}
checks['rho_minus_0863'] = {'expected': -0.863, 'computed': round(float(rho), 3), 'status': 'PASS' if abs(rho - (-0.863)) < 0.005 else 'FAIL'}
checks['discordant_101'] = {'expected': 101, 'computed': discordant, 'status': 'PASS' if discordant == 101 else 'FAIL'}

# ── 4. SYNTHETIC TABLE IS QUARANTINED ───────────────────────────────────────
checks['synthetic_moved_to_quarantine'] = {
    'expected': 'EXISTS in audit/lightgbm_robustness/synthetic_outputs/',
    'computed': str(SYNTHETIC_MOVED.exists()),
    'status': 'PASS' if SYNTHETIC_MOVED.exists() else 'FAIL'
}
checks['synthetic_NOT_in_outputs_analysis'] = {
    'expected': 'NOT EXISTS in outputs/analysis/',
    'computed': str(not SYNTHETIC_OLD.exists()),
    'status': 'PASS' if not SYNTHETIC_OLD.exists() else 'FAIL'
}

# ── 5. NO SCRIPT LOADS SYNTHETIC TABLE BY DEFAULT ───────────────────────────
scripts_loading_synthetic = []
for script in (REPO / 'scripts').glob('*.py'):
    content = script.read_text(errors='replace')
    if 'master_diagnostic_table_with_lightgbm' in content:
        scripts_loading_synthetic.append(script.name)

checks['no_canonical_script_loads_synthetic'] = {
    'expected': 0,
    'computed': len(scripts_loading_synthetic),
    'scripts': scripts_loading_synthetic,
    'status': 'PASS' if len(scripts_loading_synthetic) == 0 else 'FAIL'
}

# ── SUMMARY ─────────────────────────────────────────────────────────────────
passed = sum(1 for v in checks.values() if v.get('status') == 'PASS')
failed = [k for k, v in checks.items() if v.get('status') == 'FAIL']
total = len(checks)

print(f'\n=== CANONICAL INTEGRITY CHECKS ({passed}/{total} PASS) ===')
for k, v in checks.items():
    status = v.get('status', '?')
    mark = 'PASS' if status == 'PASS' else 'FAIL'
    print(f'  [{mark}] {k}: {v.get("computed", "?")}')

if failed:
    print(f'\nFAILED: {failed}')
else:
    print('\nAll checks PASS. Canonical evidence is clean.')

# Write machine-readable output
out = REPO / 'audit' / 'lightgbm_robustness' / 'canonical_integrity_checks.json'
with open(out, 'w') as f:
    json.dump({'summary': {'passed': passed, 'total': total, 'failed': failed}, 'checks': checks}, f, indent=2)
print(f'\nResults written to: {out}')
