"""Summarize paired benchmark outcomes without pooling raw likelihood scales."""

import argparse
import json
from pathlib import Path

import numpy as np


def winner(left, right, relative, absolute):
    tolerance = absolute + relative * max(abs(left), abs(right))
    return 'tie' if abs(left-right) <= tolerance else ('Boxmin' if left < right else 'LBFGSB')


def summarize(rows):
    groups = []
    cases = []
    for profile in ['default', 'scaled_ard']:
        for family in ['GPR', 'KRG']:
            selected = [row for row in rows if row['profile'] == profile and row['family'] == family]
            pairs = {}
            for row in selected:
                pairs.setdefault((row['case'], row['seed']), {})[row['optimizer']] = row
            assert all(set(pair) == {'Boxmin', 'LBFGSB'} for pair in pairs.values())
            group = {'profile': profile, 'family': family, 'pairs': len(pairs),
                     'fit_failures': sum(not row['fit_success'] for row in selected),
                     'rmse_wins': dict(Boxmin=0, LBFGSB=0, tie=0),
                     'objective_wins': dict(Boxmin=0, LBFGSB=0, tie=0)}
            timeRatios, evaluationRatios = [], []
            for pair in pairs.values():
                box, lbfgsb = pair['Boxmin'], pair['LBFGSB']
                if not (box['fit_success'] and lbfgsb['fit_success']):
                    continue
                group['rmse_wins'][winner(box['normalized_rmse'], lbfgsb['normalized_rmse'], .01, 1e-6)] += 1
                group['objective_wins'][winner(box['objective'], lbfgsb['objective'], 1e-6, 1e-12)] += 1
                timeRatios.append(lbfgsb['median_fit_seconds']/box['median_fit_seconds'])
                evaluationRatios.append(lbfgsb['objective_evaluations']/box['objective_evaluations'])
            group['median_time_ratio_lbfgsb_to_boxmin'] = float(np.median(timeRatios))
            group['median_evaluation_ratio_lbfgsb_to_boxmin'] = float(np.median(evaluationRatios))
            runs = [run for row in selected if row['optimizer'] == 'LBFGSB' for run in row['optimizer_runs']]
            group['lbfgsb_starts'] = len(runs)
            group['lbfgsb_unsuccessful_starts'] = sum(not run['success'] for run in runs)
            group['lbfgsb_fits_with_unsuccessful_start'] = sum(
                any(not run['success'] for run in row['optimizer_runs'])
                for row in selected if row['optimizer'] == 'LBFGSB')
            group['out_of_bounds'] = sum(row['out_of_bounds'] for row in selected)
            group['nonfinite_objectives'] = sum(row['nonfinite_objectives'] for row in selected)
            groups.append(group)
            for case in dict.fromkeys(row['case'] for row in selected):
                entry = {'profile': profile, 'family': family, 'case': case}
                for optimizer in ['Boxmin', 'LBFGSB']:
                    points = [row for row in selected if row['case'] == case and row['optimizer'] == optimizer and row['fit_success']]
                    entry[optimizer] = {
                        'median_normalized_rmse': float(np.median([row['normalized_rmse'] for row in points])),
                        'max_normalized_rmse': float(max(row['normalized_rmse'] for row in points)),
                        'median_objective': float(np.median([row['objective'] for row in points])),
                        'median_evaluations': float(np.median([row['objective_evaluations'] for row in points])),
                        'median_seconds': float(np.median([row['median_fit_seconds'] for row in points])),
                    }
                cases.append(entry)
    return {'rmse_tie_tolerance': '1e-6 + 0.01 * max(abs(a), abs(b))',
            'objective_tie_tolerance': '1e-12 + 1e-6 * max(abs(a), abs(b))',
            'groups': groups, 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize(json.loads(args.input.read_text())['results'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['groups'], indent=2))
