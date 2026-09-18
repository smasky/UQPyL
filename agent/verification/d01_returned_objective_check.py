"""Re-evaluate recorded optimizer returns without rerunning optimization."""

import json
from pathlib import Path

import numpy as np

from d01_optimizer_benchmark import makeData, makeModel


root = Path(__file__).resolve().parent
rows = json.loads((root / '0918-d01-optimizer-results.json').read_text())['results']
discrepancies, selectionChanges = [], []
for row in rows:
    model = makeModel(row['family'], row['profile'], row['optimizer'], row['seed'])
    trainX, trainY = model.prepareTrainingData(*makeData(row['case'])[:2])
    infos, _, _ = model.setting.getParaInfos(model.getParaList())
    recomputed = []
    for index, run in enumerate(row['optimizer_runs']):
        model.setting.setVals(infos, np.array(run['returned_point']))
        model.fitModel(trainX, trainY)
        actual = float(model.fitState['objective'])
        recomputed.append(actual)
        reported = run['returned_objective']
        tolerance = 1e-10 + 1e-8 * max(abs(actual), abs(reported))
        if abs(actual-reported) > tolerance:
            discrepancies.append({key: row[key] for key in ['family', 'profile', 'case', 'optimizer', 'seed']} | {
                'restart_index': index, 'reported_objective': reported,
                'recomputed_objective': actual, 'success': run.get('success'),
            })
    chosen = int(np.argmin([run['returned_objective'] for run in row['optimizer_runs']]))
    alternative = int(np.argmin(recomputed))
    if chosen != alternative:
        selectionChanges.append({key: row[key] for key in ['family', 'profile', 'case', 'optimizer', 'seed']} | {
            'chosen_restart': chosen, 'recomputed_best_restart': alternative,
            'chosen_objective': recomputed[chosen], 'alternative_objective': recomputed[alternative],
        })
report = {'tolerance': '1e-10 + 1e-8 * max(abs(recomputed), abs(reported))',
          'discrepancies': discrepancies, 'selection_changes': selectionChanges}
(root / '0918-d01-returned-objective-check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
