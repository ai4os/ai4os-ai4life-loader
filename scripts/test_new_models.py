"""Run the prediction test for each new model and collect the results.

Usage:
    python scripts/test_new_models.py scripts/new_models.txt

Each model is tested in its own pytest run (tests/test_predictions/ with
``-k <nickname>``) with its own time limit, so one slow or crashing model
only fails itself and never loses the results of the others.

Results are written to scripts/test_results.json after every model:
    {
        "passed": ["model1", "model2"],
        "failed": [{"model": "model3", "error": "..."}]
    }

Environment variables:
    MODEL_TEST_TIMEOUT  time limit per model in seconds (default 1200)

Exit code 0 if all passed, 1 if any failed.
"""

import json
import os
import subprocess
import sys
import time

RESULTS_FILE = "scripts/test_results.json"
DEFAULT_TIMEOUT = 1200  # seconds per model

# pytest exit code when no test matched the -k filter
NO_TESTS_COLLECTED = 5


def nickname_of(key):
    """'affable-shark 🦈' -> 'affable-shark'."""
    return key.split(" ", 1)[0]


def extract_error(output):
    """Return the most useful one-line error from a pytest output."""
    lines = output.splitlines()
    # pytest's short test summary: "ERROR tests/...[model ...] - reason"
    for line in lines:
        if line.startswith(("ERROR tests", "FAILED tests")) and " - " in line:
            return line.split(" - ", 1)[1].strip()[:200]
    # otherwise the last "E   ..." line of a traceback
    for line in reversed(lines):
        if line.startswith("E "):
            return line[1:].strip()[:200]
    for line in lines:
        if line.startswith(("ERROR tests", "FAILED tests")):
            return line.strip()[:200]
    return "test failed (see workflow log)"


def test_model(nickname, timeout):
    """Run the prediction test for one model.

    Returns ``(passed, error_message)``.
    """
    command = [
        sys.executable, "-m", "pytest",
        "tests/test_predictions/",
        "-v",
        "-k", nickname,
        "--tb=short",
        "-W", "ignore",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output[-5000:])
        return False, f"timed out after {timeout // 60} min"

    output = result.stdout + (result.stderr or "")
    print(output if result.returncode else output[-2000:])

    if result.returncode == 0:
        return True, None
    if result.returncode == NO_TESTS_COLLECTED:
        return False, "no test collected for this model"
    if result.returncode < 0:
        # killed by a signal, e.g. -9 when it ran out of memory
        return False, f"test process killed by signal {-result.returncode}"
    return False, extract_error(output)


def write_results(passed, failed):
    with open(RESULTS_FILE, "w") as f:
        json.dump({"passed": passed, "failed": failed}, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_new_models.py <new_models.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        sys.exit(1)

    with open(input_file) as f:
        model_keys = [line.strip() for line in f if line.strip()]

    passed, failed = [], []
    write_results(passed, failed)

    if not model_keys:
        print("No new models to test.")
        sys.exit(0)

    timeout = int(os.environ.get("MODEL_TEST_TIMEOUT", DEFAULT_TIMEOUT))
    print(f"Testing {len(model_keys)} model(s), {timeout // 60} min each max")

    for i, key in enumerate(model_keys, 1):
        nickname = nickname_of(key)
        print(f"\n{'=' * 60}\n[{i}/{len(model_keys)}] {nickname}\n{'=' * 60}")
        start = time.time()
        ok, error = test_model(nickname, timeout)
        duration = int(time.time() - start)
        if ok:
            passed.append(nickname)
            print(f"--> {nickname}: PASSED ({duration}s)")
        else:
            failed.append({"model": nickname, "error": error})
            print(f"--> {nickname}: FAILED ({duration}s): {error}")
        # save after every model, so finished results are never lost
        write_results(passed, failed)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    for item in failed:
        print(f"  FAILED: {item['model']}: {item['error']}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
