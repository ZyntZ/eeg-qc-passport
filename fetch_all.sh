#!/usr/bin/env bash
# Расширенная выборка: все 109 добровольцев PhysioNet EEGMMIDB, baseline R01 (глаза открыты) и R02 (закрыты).
# 218 файлов × 1,27 МБ ≈ 277 МБ. Данные в репозиторий не входят.
set -uo pipefail
mkdir -p data_cache
BASE="https://physionet.org/files/eegmmidb/1.0.0"
got=0; skip=0; fail=0
for n in $(seq 1 109); do
  s=$(printf '%03d' "$n")            # 1 -> 001 ... 109 -> 109
  for r in 01 02; do
    f="data_cache/S${s}R${r}.edf"
    if [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 1000000 ]; then skip=$((skip+1)); continue; fi
    curl -sL --retry 3 --max-time 90 -o "$f" "$BASE/S${s}/S${s}R${r}.edf" || true
    if [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 1000000 ]; then got=$((got+1)); else rm -f "$f"; fail=$((fail+1)); fi
  done
  if [ $((n % 20)) -eq 0 ]; then echo "прогресс: $n/109 субъектов, файлов: $(ls data_cache/*.edf 2>/dev/null | wc -l)"; fi
done
echo "downloaded=$got skipped=$skip failed=$fail total=$(ls data_cache/*.edf 2>/dev/null | wc -l)"
