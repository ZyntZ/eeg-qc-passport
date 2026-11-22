#!/usr/bin/env bash
# Загрузка публичного датасета PhysioNet EEGMMIDB v1.0.0 (Schalk 2009, ODC-By 1.0)
# Только 20 добровольцев × 2 baseline-записи (R01 глаза открыты, R02 закрыты) — 40 файлов, ~50 МБ.
# Данные НЕ входят в репозиторий (лицензия и объём): качаем по официальному URL.
set -euo pipefail
mkdir -p data_cache
BASE="https://physionet.org/files/eegmmidb/1.0.0"
for s in $(seq -w 1 20); do
  for r in 01 02; do
    f="data_cache/S0${s}R${r}.edf"
    if [ ! -s "$f" ]; then
      echo "downloading $f"
      curl -sL --retry 3 -o "$f" "$BASE/S0${s}/S0${s}R${r}.edf"
    fi
  done
done
echo "done: $(ls data_cache/*.edf | wc -l) files"
