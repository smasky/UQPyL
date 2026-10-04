"""Run complete tutorial workflows and refresh their captured example output.

Run in py312 from the repository root. --refresh-output updates only existing
Example output blocks immediately following the selected runnable examples.
API fragments requiring user-defined names are not treated as standalone code.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-output", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    # Keep checkout imports available after moving execution into a clean cwd.
    import sys

    sys.path.insert(0, str(root))
    pages = [
        "examples.md",
        "cn/examples.md",
        "quick_start.md",
        "cn/quick_start.md",
        "surrogate.md",
        "cn/surrogate.md",
        "api/problem.md",
    ]
    records = []
    for page in pages:
        path = root / "docs_v2" / page
        source = path.read_text()
        edits = []
        blocks = list(re.finditer(r"```python\s*\n(.*?)```", source, re.S))
        namespace = {"__name__": "documented_workflow"}
        with tempfile.TemporaryDirectory(prefix="uqpyl-workflow-docs-") as workDir:
            os.chdir(workDir)
            try:
                for index, block in enumerate(blocks):
                    code = block[1]
                    if page.endswith("surrogate.md") and "tuner.gridTune(" not in code:
                        continue
                    if page == "api/problem.md" and "class MSEEvaluator" not in code:
                        continue
                    if "quick_start" not in page:
                        namespace = {"__name__": "documented_workflow"}
                    output = io.StringIO()
                    with np.printoptions(), contextlib.redirect_stdout(output):
                        exec(compile(code, f"{page}:block{index + 1}", "exec"), namespace)
                    printed = output.getvalue()
                    records.append({"page": page, "block": index + 1, "status": "passed", "stdout": printed})
                    following = source[block.end() :]
                    match = re.match(r"(?:(?!```).)*?Example output:\s*```text\n(.*?)```", following, re.S)
                    if match and args.refresh_output:
                        edits.append((block.end() + match.start(1), block.end() + match.end(1), printed))
            finally:
                os.chdir(root)
        for start, end, replacement in reversed(edits):
            source = source[:start] + replacement + source[end:]
        if edits:
            path.write_text(source)
    outputPath = root / "agent/verification/0919-workflows-docs-after.json"
    outputPath.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} Python blocks passed across {len(pages)} pages.")


if __name__ == "__main__":
    main()
