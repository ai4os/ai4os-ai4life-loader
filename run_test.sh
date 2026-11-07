#!/bin/bash
cd ~/ai4os-ai4life-loader
source .venv/bin/activate
mkdir -p logs
for m in content-fish knowledgeable-mouse joyful-deer loyal-ant nervous-shark confused-cow impartial-shrimp; do
  start=$(date +%s)
  AI4LIFE_TEST_ALL_MODELS=1 /usr/bin/time -l python -m pytest tests/test_predictions/ -q -W ignore --tb=short -k "$m" > "logs/$m.log" 2>&1
  code=$?
  mem=$(grep "maximum resident set size" "logs/$m.log" | awk '{printf "%.1f GB", $1/1e9}')
  echo "$m exit=$code $(( $(date +%s) - start ))s $mem $(grep -E 'passed|failed|error' "logs/$m.log" | tail -1)" >> results.txt
done
echo "ALL DONE" >> results.txt