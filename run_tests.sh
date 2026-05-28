#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAT_DIR="$SCRIPT_DIR/testcases/sat"
UNSAT_DIR="$SCRIPT_DIR/testcases/unsat"

if [[ $# -gt 0 ]]; then
    EXE="$1"
    shift
else
    echo "Building zig-minisat..."
    zig build -Doptimize=ReleaseFast
    EXE="$SCRIPT_DIR/zig-out/bin/zig-minisat"
fi

if [[ ! -x "$EXE" ]]; then
    echo "ERROR: binary not found at $EXE" >&2
    exit 1
fi

sat_pass=0
sat_fail=0
sat_total=0
unsat_pass=0
unsat_fail=0
unsat_total=0
failures=()

run_case() {
    local file=$1
    local expected=$2
    local label=$3

    local output
    output=$("$EXE" --quiet "$file" 2>/dev/null)

    if [[ "$output" == "$expected" ]]; then
        return 0
    else
        echo "  FAIL $label: expected '$expected', got '$output' ($(basename "$file"))"
        return 1
    fi
}

echo ""
echo "=== SAT test cases ==="
for f in "$SAT_DIR"/*.cnf; do
    sat_total=$((sat_total + 1))
    if run_case "$f" "SATISFIABLE" "sat"; then
        sat_pass=$((sat_pass + 1))
    else
        sat_fail=$((sat_fail + 1))
        failures+=("$f: expected SATISFIABLE")
    fi
done

echo ""
echo "=== UNSAT test cases ==="
for f in "$UNSAT_DIR"/*.cnf; do
    unsat_total=$((unsat_total + 1))
    if run_case "$f" "UNSATISFIABLE" "unsat"; then
        unsat_pass=$((unsat_pass + 1))
    else
        unsat_fail=$((unsat_fail + 1))
        failures+=("$f: expected UNSATISFIABLE")
    fi
done

echo ""
echo "=== Results ==="
echo "SAT:   $sat_pass / $sat_total passed ($sat_fail failed)"
echo "UNSAT: $unsat_pass / $unsat_total passed ($unsat_fail failed)"
total_pass=$((sat_pass + unsat_pass))
total=$((sat_total + unsat_total))
total_fail=$((sat_fail + unsat_fail))
echo "TOTAL: $total_pass / $total passed ($total_fail failed)"

if [[ ${#failures[@]} -gt 0 ]]; then
    echo ""
    echo "=== Failed cases ==="
    for f in "${failures[@]}"; do
        echo "  $f"
    done
    exit 1
fi

echo ""
echo "All tests passed!"