"""Run pytest and export collected IDs, explicit markers and observed durations.

Run from repository root with the py312 interpreter. Extra CLI arguments go to
pytest; default selection remains the complete suite, including statistics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pytest


def testDomain(filename):
    stem = Path(filename).stem.removeprefix('test_')
    aliases = {
        'analysis': ('analysis_',),
        'calibration': ('calibration_', 'sufi2_'),
        'doe': ('doe_', 'lhs_'),
        'inference': ('inference_', 'dream_'),
        'optimization': ('optimization_', 'abc_', 'hv_', 'moasmo_'),
        'surrogate': ('surrogate_', 'autotuner_', 'gpr_', 'kriging_', 'lasso_', 'lbfgsb_', 'mars_', 'rbf_', 'svr_'),
        'problem': ('problem', 'model_problem'),
        'runtime': ('runtime_',),
        'viz': ('viz_',),
        'util': ('util_',),
    }
    for domain, prefixes in aliases.items():
        if stem.startswith(prefixes):
            return domain
    return 'integration'


class Inventory:
    def __init__(self):
        self.cases = []
        self.reports = []

    def pytest_collection_modifyitems(self, items):
        for item in items:
            self.cases.append({
                'node_id': item.nodeid,
                'domain': testDomain(item.nodeid.split('::')[0]),
                'markers': sorted({mark.name for mark in item.iter_markers()} - {'parametrize'}),
            })

    def pytest_runtest_logreport(self, report):
        self.reports.append({'node_id': report.nodeid, 'phase': report.when,
                             'outcome': report.outcome, 'duration_seconds': report.duration})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args, pytestArgs = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    plugin = Inventory()
    status = pytest.main(pytestArgs, plugins=[plugin])
    payload = {
        'exit_code': int(status),
        'collected_count': len(plugin.cases),
        'domain_counts': dict(sorted(Counter(case['domain'] for case in plugin.cases).items())),
        'marker_counts': {name: sum(name in case['markers'] for case in plugin.cases)
                          for name in ('numerical', 'statistical')},
        'cases': plugin.cases, 'reports': plugin.reports,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + '\n')
    raise SystemExit(status)
