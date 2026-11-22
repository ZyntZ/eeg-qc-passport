"""
23_lead_parse.py - разбор лога дообучения LEAD в таблицу reports/lead_5fold_<tag>.csv.

Структура лога run.py: на каждый фолд сначала идёт обучение (там метрики
печатаются после каждой эпохи, они не нужны), потом блок
">>>>>>>testing : ...seedNN<<<<" с финальной оценкой лучшего чекпойнта.
Внутри этого блока две пары строк: сначала Validation/Test по окнам
(sample-level), потом Validation/Test по людям (subject-level, голосование
окон внутри человека).

В курсовую идёт subject-level: диагноз ставится человеку, а не двухсекундному
кусочку записи. Sample-level оставлен для сравнения.

Запуск:
    python src/23_lead_parse.py reports/lead_adhc.log --tag adhc
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUROC", "AUPRC"]
NUM = r"([0-9]+\.[0-9]+)"


def parse_metric_line(line):
    out = {}
    for m in METRICS:
        hit = re.search(m + r":\s*" + NUM, line)
        if not hit:
            return None
        out[m.lower()] = float(hit.group(1))
    return out


def load_log(path):
    """Лог Kaggle - JSON со списком записей, локальный прогон - обычный текст."""
    raw = Path(path).read_text(errors="replace")
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return "".join(e.get("data", "") for e in json.loads(raw))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tag", default="multi")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = load_log(args.log).splitlines()

    folds, cur = [], None
    for line in lines:
        hit = re.search(r">>>+testing\s*:.*seed(\d+)", line)
        if hit:
            cur = {"seed": int(hit.group(1)), "tests": []}
            folds.append(cur)
            continue
        if cur is None:
            continue
        hit = re.search(r"Number of test subjects:\s*(\d+)", line)
        if hit:
            cur["n_test_subjects"] = int(hit.group(1))
            continue
        if line.strip().startswith("Test results ---"):
            vals = parse_metric_line(line)
            if vals:
                cur["tests"].append(vals)

    folds = [f for f in folds if len(f["tests"]) >= 2]
    if not folds:
        raise SystemExit("не найдено ни одного блока '>>>>>>>testing' с метриками")

    out = Path(args.out or f"reports/lead_5fold_{args.tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["fold", "seed", "level", "n_test_subjects"] + [m.lower() for m in METRICS]
    rows = []
    for i, f in enumerate(folds):
        for level, vals in zip(("sample", "subject"), f["tests"][:2]):
            rows.append({"fold": i, "seed": f["seed"], "level": level,
                         "n_test_subjects": f.get("n_test_subjects", ""), **vals})
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"фолдов: {len(folds)}  (seed {folds[0]['seed']}..{folds[-1]['seed']})")
    for level in ("sample", "subject"):
        sel = [r for r in rows if r["level"] == level]
        if not sel:
            continue
        print(f"\n{level}-level, тест, среднее по {len(sel)} фолдам:")
        for m in METRICS:
            xs = [r[m.lower()] for r in sel]
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            print(f"  {m:<12} {statistics.mean(xs)*100:5.1f} +- {sd*100:4.1f}%")
    print(f"\nзаписано: {out}")


if __name__ == "__main__":
    main()
