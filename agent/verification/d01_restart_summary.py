"""Summarize paired start-policy budgets; holdout never selects parameters."""
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    summary = {}
    for policy in ['configured_first', 'all_random']:
        table = []
        for index, count in enumerate([1, 2, 3, 5, 9]):
            current = [row['policies'][policy][index] for row in rows]
            reference = [row['policies'][policy][-1] for row in rows]
            close = [a['objective']-b['objective'] <= 1e-10 + 1e-4*abs(b['objective'])
                     for a, b in zip(current, reference)]
            table.append(dict(total_runs=count, near_nine_run_objective=sum(close), fits=len(rows),
                              median_normalized_rmse=float(np.median([a['normalized_rmse'] for a in current])),
                              median_evaluations=float(np.median([a['evaluations'] for a in current])),
                              median_seconds=float(np.median([a['seconds'] for a in current]))))
        summary[policy] = table
    summary['configured_vs_random'] = []
    for index, count in enumerate([1, 2, 3, 5, 9]):
        wins = [0, 0, 0]
        for row in rows:
            a, b = [row['policies'][policy][index]['normalized_rmse']
                    for policy in ['configured_first', 'all_random']]
            tolerance = 1e-6 + .01*max(abs(a), abs(b))
            wins[0 if a < b-tolerance else 1 if b < a-tolerance else 2] += 1
        summary['configured_vs_random'].append(dict(total_runs=count, configured_wins=wins[0],
                                                   random_wins=wins[1], ties=wins[2]))
    return summary


root = Path(__file__).parent
rows = json.loads((root/'0918-d01-restart-results.json').read_text())['results']
summary = {'all': summarize(rows)}
for profile in ['default', 'scaled_ard']:
    for family in ['GPR', 'KRG']:
        subset = [row for row in rows if row['profile'] == profile and row['family'] == family]
        summary[profile+'_'+family] = summarize(subset)
(root/'0918-d01-restart-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
