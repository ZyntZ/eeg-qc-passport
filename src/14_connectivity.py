"""
14_connectivity.py - взвешенный индекс фазового запаздывания (wPLI).

Раньше здесь была заглушка: скрипт писал NaN и печатал предупреждение,
что wPLI не считался. Теперь считается по тем же derivatives и тем же участкам
записи, что и спектральные признаки в 04, - код нарезки импортируется
оттуда, чтобы условия совпадали.

Что такое wPLI. Для пары каналов берётся взаимный спектр, от него - мнимая
часть. Если разность фаз держится около нуля, мнимая часть болтается вокруг
нуля и индекс низкий; если фаза устойчиво сдвинута, индекс растёт. Веса по
модулю мнимой части гасят вклад почти синфазных окон, поэтому объёмное
проведение (один источник, видимый сразу на нескольких электродах) мешает
меньше, чем обычной когерентности. Формула из Vinck и соавт., 2011:

    wPLI = |среднее Im(S_xy)| / среднее |Im(S_xy)|,  среднее по окнам

Окно то же, что у Welch в 04 (2 с, перекрытие 50%), чтобы длинная ГЗ
и короткая ГО оценивались с одинаковой точностью.

Выход: data/features_conn.csv - по строке на человека и условие.

Запуск: python src/14_connectivity.py [--limit N]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent


def _load_extractor():
    """04 переиспользуем как модуль: имя с цифр не импортируется обычным import."""
    spec = importlib.util.spec_from_file_location(
        "extract04", HERE / "04_extract_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load_extractor()
mne = E.mne

OUT = Path("data/features_conn.csv")
OUT_MERGED = Path("data/features_with_conn.csv")
OUT_STATS = Path("reports/conn_stats.csv")
SPECTRAL = Path("data/features_spectral.csv")
CONN_BANDS = {"alpha": (8, 13), "theta": (4, 8)}
MIN_WINDOWS = 10          # меньше десяти окон - оценка пустая


def windows(raw, spans, n_fft):
    """Окна Ханна по кускам записи, перекрытие половина окна."""
    data = raw.get_data()
    sf = raw.info["sfreq"]
    step = n_fft // 2
    win = np.hanning(n_fft)
    out = []
    for a, b in spans:
        i0, i1 = int(round(max(a, 0.0) * sf)), int(round(min(b, raw.times[-1]) * sf))
        if i1 - i0 < n_fft:
            continue
        for s in range(i0, i1 - n_fft + 1, step):
            seg = data[:, s:s + n_fft]
            out.append(np.fft.rfft(seg * win, axis=-1))
    return out


def wpli_matrix(ffts, freqs, fmin, fmax):
    """Матрица wPLI по каналам, усреднённая по частотам полосы."""
    sel = (freqs >= fmin) & (freqs <= fmax)
    X = np.stack(ffts)[:, :, sel]                  # окна x каналы x частоты
    # взаимный спектр для всех пар сразу
    S = X[:, :, None, :] * np.conj(X[:, None, :, :])
    im = S.imag
    num = np.abs(im.mean(axis=0))
    den = np.abs(im).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(den > 0, num / den, np.nan)
    return np.nanmean(w, axis=-1)                  # каналы x каналы


def pair_mean(mat, names, group_a, group_b):
    """Среднее по парам между двумя группами электродов.

    clean_label приводит метки к нижнему регистру, а списки в 04 записаны
    как O1/Fp1 - сравниваем в одном регистре, иначе пары не находятся.
    """
    names = [n.lower() for n in names]
    ia = [names.index(c.lower()) for c in group_a if c.lower() in names]
    ib = [names.index(c.lower()) for c in group_b if c.lower() in names]
    if not ia or not ib:
        return np.nan
    vals = [mat[i, j] for i in ia for j in ib if i != j]
    return float(np.nanmean(vals)) if vals else np.nan


def global_mean(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return float(np.nanmean(mat[iu]))


def conn_features(raw, spans):
    sf = raw.info["sfreq"]
    n_fft = int(round(E.WELCH_WIN * sf))
    ffts = windows(raw, spans, n_fft)
    if len(ffts) < MIN_WINDOWS:
        return None
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sf)
    names = [E.clean_label(c) for c in raw.ch_names]
    out = {"n_windows_conn": len(ffts)}
    for band, (lo, hi) in CONN_BANDS.items():
        m = wpli_matrix(ffts, freqs, lo, hi)
        out[f"wpli_{band}_OF"] = pair_mean(m, names, E.OCCIPITAL, E.FRONTAL)
        out[f"wpli_{band}_global"] = global_mean(m)
        out[f"wpli_{band}_OO"] = pair_mean(m, names, E.OCCIPITAL, E.OCCIPITAL)
    return out


def stats_table(df):
    """AD против контроля по каждому условию, поправка BH внутри условия.

    Отдельно считается проверка на смещение: wPLI тем выше, чем меньше окон
    усредняется, поэтому полная ГЗ и обрезанная ГЗ сравниваются парно.
    """
    cols = [c for c in df.columns if c.startswith("wpli_")]
    rows = []
    for cond, sub in df.groupby("condition"):
        ps, keep = [], []
        for col in cols:
            a = sub[sub.diagnosis == "A"][col].dropna()
            c = sub[sub.diagnosis == "C"][col].dropna()
            if len(a) < 5 or len(c) < 5:
                continue
            ps.append(mannwhitneyu(a, c).pvalue)
            keep.append((col, len(a), len(c), a.median(), c.median()))
        if not ps:
            continue
        fdr = multipletests(ps, method="fdr_bh")[1]
        for (col, na, nc, ma, mc), p, q in zip(keep, ps, fdr):
            rows.append(dict(condition=cond, feature=col, n_AD=na, n_HC=nc,
                             median_AD=round(ma, 3), median_HC=round(mc, 3),
                             p=round(p, 4), fdr=round(q, 4)))
    out = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    out.to_csv(OUT_STATS, index=False)
    return out


def bias_check(df):
    """Та же запись, разное число окон: полная ГЗ против обрезанной."""
    w = df.pivot_table(index="participant_id", columns="condition",
                       values="wpli_alpha_global")
    n = df.pivot_table(index="participant_id", columns="condition",
                       values="n_windows_conn")
    if not {"EC", "EC_match"}.issubset(w.columns):
        return None
    both = w[["EC", "EC_match"]].dropna()
    if len(both) < 10:
        return None
    p = wilcoxon(both["EC"], both["EC_match"]).pvalue
    return dict(n=len(both), n_win_EC=int(n["EC"].median()),
                n_win_match=int(n["EC_match"].median()),
                median_EC=round(both["EC"].median(), 3),
                median_match=round(both["EC_match"].median(), 3), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    plist = E.RAW_ROOT / "ds004504/participants.tsv"
    if not plist.exists():
        sys.exit(f"нет {plist}, сначала src/00_fetch_data.py")
    subs = sorted(pd.read_csv(plist, sep="\t")["participant_id"].astype(str))
    if args.limit:
        subs = subs[:args.limit]
    print(f"испытуемых: {len(subs)}")

    rows = []
    for sub in subs:
        print(sub, flush=True)
        per_sub, eo_dur = {}, None

        p, _ = E.resolve_path(sub, "EO")
        if p is not None:
            raw = E.reref(mne.io.read_raw_eeglab(p, preload=True))
            ann = E.annotations_table(raw, sub, "EO")
            tmax = raw.times[-1]
            holes = E.excluded_spans(ann, tmax)
            photic = E.photic_spans(ann)
            strict = E.subtract(E.eye_spans(ann, tmax, "strict"), holes)
            spans_by_cond = {
                "EO": strict,
                "EO_photic": E.intersect(strict, photic),
                "EO_nophotic": E.subtract(strict, photic),
                "EO_broad": E.subtract(E.eye_spans(ann, tmax, "broad"), holes),
            }
            for name, spans in spans_by_cond.items():
                f = conn_features(raw, spans)
                if f:
                    per_sub[name] = f
            if "EO" in per_sub:
                eo_dur = sum(b - a for a, b in strict)
            del raw

        p, _ = E.resolve_path(sub, "EC")
        if p is not None:
            raw = E.reref(mne.io.read_raw_eeglab(p, preload=True))
            ann = E.annotations_table(raw, sub, "EC")
            tmax = raw.times[-1]
            spans = E.subtract([(0.0, tmax)], E.excluded_spans(ann, tmax))
            f = conn_features(raw, spans)
            if f:
                per_sub["EC"] = f
            if eo_dur:
                fm = conn_features(raw, E.head_spans(spans, eo_dur))
                if fm:
                    per_sub["EC_match"] = fm
            del raw

        for cond, f in per_sub.items():
            rows.append({**f, "participant_id": sub, "condition": cond})

    df = pd.DataFrame(rows)
    meta = pd.read_csv(E.RAW_ROOT / "ds004504/participants.tsv", sep="\t")
    meta.columns = meta.columns.str.replace("\ufeff", "").str.strip()
    meta = meta[["participant_id", "Group"]].rename(columns={"Group": "diagnosis"})
    df = df.merge(meta, on="participant_id", how="left")
    cols = ["participant_id", "condition", "diagnosis"] + \
           [c for c in df.columns if c.startswith("wpli_")] + ["n_windows_conn"]
    df = df[cols]
    Path("data").mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)

    # полная таблица: спектральные признаки плюс связность
    if SPECTRAL.exists():
        sp = pd.read_csv(SPECTRAL)
        merged = sp.merge(df.drop(columns=["diagnosis"]),
                          on=["participant_id", "condition"], how="left")
        merged.to_csv(OUT_MERGED, index=False)
        print("сохранено:", OUT_MERGED, f"({len(merged)} строк)")

    st = stats_table(df)
    print("\nAD против контроля, значимые после BH внутри условия:")
    sig = st[st.fdr < 0.05]
    print(sig.to_string(index=False) if len(sig) else "  нет")

    b = bias_check(df)
    if b:
        print(f"\nсмещение по числу окон: та же запись, "
              f"ГЗ {b['n_win_EC']} окон -> {b['median_EC']}, "
              f"обрезанная {b['n_win_match']} окон -> {b['median_match']}, "
              f"p={b['p']:.1e} (n={b['n']})")

    print(f"\nстрок: {len(df)}, людей: {df.participant_id.nunique()}")
    for cond in ("EC", "EC_match", "EO", "EO_photic"):
        sub = df[df.condition == cond]
        if sub.empty:
            continue
        print(f"\n{cond} (n={len(sub)}), медианы по группам:")
        print(sub.groupby("diagnosis")[[c for c in df.columns
                                        if c.startswith("wpli_")]].median().round(3))
    print("\nсохранено:", args.out)


if __name__ == "__main__":
    main()
