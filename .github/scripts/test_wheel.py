"""Install one wheel in a fresh environment and test it outside the checkout."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel-dir', default='dist')
    parser.add_argument('--report-dir', default='.cache/wheel-test')
    args = parser.parse_args()
    repoRoot = Path(__file__).resolve().parents[2]
    wheelPaths = list(Path(args.wheel_dir).resolve().glob('*.whl'))
    if len(wheelPaths) != 1:
        parser.error('Expected exactly one wheel in --wheel-dir.')
    reportDir = Path(args.report_dir).resolve()
    reportDir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uqpyl-wheel-test-') as tmpDir:
        testRoot = Path(tmpDir)
        envDir = testRoot / 'venv'
        venv.EnvBuilder(with_pip=True).create(envDir)
        python = envDir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(testRoot / 'matplotlib'))

        def run(*arguments):
            subprocess.run([str(python), *map(str, arguments)], cwd=testRoot, env=env, check=True)

        run('-m', 'pip', 'install', f'{wheelPaths[0]}[viz]', 'pytest', 'pytest-cov')
        run('-m', 'pip', 'check')
        # Native imports must fail rather than disappearing behind importorskip.
        run('-I', '-c', '''
import importlib
import importlib.machinery
from pathlib import Path
import sys
import UQPyL
packagePath = Path(UQPyL.__file__).resolve()
assert packagePath.is_relative_to(Path(sys.prefix).resolve()), packagePath
modules = [
    *('UQPyL.surrogate.mars.core.' + name for name in
      ['_types', '_util', '_forward', '_record', '_basis', '_pruning', '_qr', '_knot_search']),
    'UQPyL.surrogate.regression.lasso.lasso',
    'UQPyL.surrogate.svr.core.libsvm_interface',
]
for name in modules:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(packagePath.parent), path
    assert any(str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES), path
print('Installed package:', packagePath)
print('All 10 native extensions imported successfully.')
''')
        shutil.copytree(repoRoot / 'tests', testRoot / 'tests',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(repoRoot / 'pyproject.toml', testRoot / 'pyproject.toml')
        try:
            run('-m', 'pytest', '-q', '--cov=UQPyL', '--cov-report=term-missing',
                f'--cov-report=xml:{reportDir / "coverage.xml"}',
                f'--junitxml={reportDir / "junit.xml"}')
        finally:
            coveragePath = reportDir / 'coverage.xml'
            if coveragePath.exists():
                tree = ET.parse(coveragePath)
                # Point coverage consumers at the checkout, not the deleted venv.
                for entry in tree.findall('.//class'):
                    filename = entry.get('filename', '').replace('\\', '/')
                    if '/UQPyL/' in filename:
                        filename = 'UQPyL/' + filename.split('/UQPyL/', 1)[1]
                    elif not filename.startswith('UQPyL/'):
                        filename = 'UQPyL/' + filename
                    entry.set('filename', filename)
                for source in tree.findall('./sources/source'):
                    source.text = str(repoRoot)
                tree.write(coveragePath, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    main()
