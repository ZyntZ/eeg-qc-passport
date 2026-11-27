#!/usr/bin/env bash
# Загрузка обоих открытых датасетов без регистрации и ключей API:
#  1) PhysioNet EEGMMIDB v1.0.0 — 109 человек, 218 EDF (~300 МБ, ODC-By 1.0)
#  2) UCI EEG Eye State — 14 980 эпох, гарнитура Emotiv EPOC (~2 МБ, CC0-1.0)
set -euo pipefail
bash fetch_data.sh
python3 fetch_data_uci.py
