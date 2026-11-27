"""Калибровка порогов под профиль прибора и расчёт чувствительности / специфичности.

Схема:
 1. Деление 109 испытуемых по хэшу ID: 54 — калибровка, 55 — отложенная проверка (holdout).
 2. На калибровочной половине измеряется собственный профиль прибора (шум 1–40 Гц,
    межоконный перепад медиан, типичный линейный наклон базовой линии) и формулами
    выводятся адаптивные пороги без ручной подгонки.
 3. На отложенной половине (55 человек) рассчитываются:
    - Специфичность (Specificity = доля чистых записей без ложной блокировки);
    - Доля ложной отбраковки (FPR = 1 − Specificity) для строгих дефолтов, мягких фиксированных
      порогов и адаптивных порогов из профиля прибора;
    - Чувствительность (Sensitivity / Recall) детектора по 6 типам контролируемых дефектов:
      плоский канал, насыщение усилителя, наводка 50 Гц, ступенька контакта,
      дрейф базовой линии, общая помеха референта (crosstalk).
"""
import hashlib
import json
import pathlib
import mne
import numpy as np
import pandas as pd
import qc_engine as Q

mne.set_log_level('ERROR')
DATA = pathlib.Path('data_cache')
OUT = pathlib.Path('results')
OUT.mkdir(exist_ok=True)

MAP = [('Fp1.', 'Fp1'), ('Fp2.', 'Fp2'), ('F7..', 'F7'), ('F8..', 'F8'), ('O1..', 'O1'), ('O2..', 'O2')]
NAMES = [a for a, _ in MAP]
SHORT = [b for _, b in MAP]

BASE_RULES = dict(Q.RULESET)
BASE_PROFILE = dict(Q.DEVICE_PROFILE)
SAT_LIMIT_UV = 1000.0


def split_subjects(subjects):
    calib, holdout = [], []
    for s in subjects:
        h = hashlib.md5(s.encode()).hexdigest()
        (calib if int(h[:8], 16) % 2 == 0 else holdout).append(s)
    return sorted(calib), sorted(holdout)


def load_pair(sub):
    f1, f2 = DATA / f'{sub}R01.edf', DATA / f'{sub}R02.edf'
    if not (f1.exists() and f2.exists()):
        return None
    r1 = mne.io.read_raw_edf(f1, preload=True, verbose=False)
    r2 = mne.io.read_raw_edf(f2, preload=True, verbose=False)
    if not all(n in r1.ch_names for n in NAMES):
        return None
    r1.pick(NAMES)
    r2.pick(NAMES)
    return r1.get_data() * 1e6, r2.get_data() * 1e6, float(r1.info['sfreq'])


def measure_profile(subjects):
    noises, amps, jumps, slopes = [], [], [], []
    for sub in subjects:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, x2, fs = pair
        for x in (x1, x2):
            w = int(fs)
            for k in range(x.shape[0]):
                sig = Q._bp(Q._notch(x[k], fs), fs, 1, 40)
                noises.append(float(np.std(sig)))
                amps.append(float(np.percentile(np.abs(x[k]), 99.5)))
                m = np.array([np.median(sig[i:i + w]) for i in range(0, len(sig) - w, w)])
                if len(m) > 3:
                    jumps.append(np.abs(np.diff(m)))
                t = np.arange(len(x[k])) / fs
                slopes.append(abs(float(np.polyfit(t, x[k], 1)[0])))
    j = np.concatenate(jumps) if jumps else np.array([0.0])
    sl = np.array(slopes) if slopes else np.array([1.0])
    return np.array(noises), np.array(amps), j, sl


def derive_thresholds(noise, amp, jumps, slopes):
    med = float(np.median(noise))
    jump_p95 = float(np.percentile(jumps, 95))
    slope_p95 = float(np.percentile(slopes, 95))
    thr = dict(
        flat_brain_uv=round(0.10 * med, 2),
        step_uv=round(4.0 * jump_p95, 1),
        drift_uv_s=round(max(10.0, 3.0 * slope_p95), 1),
        sat_uv=SAT_LIMIT_UV,
    )
    prof = (round(0.20 * med, 2), round(3.20 * med, 1))
    stats = dict(
        median_noise_uv=round(med, 2),
        p05_noise_uv=round(float(np.percentile(noise, 5)), 2),
        p95_noise_uv=round(float(np.percentile(noise, 95)), 2),
        median_amp_p995_uv=round(float(np.median(amp)), 1),
        window_jump_p95_uv=round(jump_p95, 2),
        window_jump_p99_uv=round(float(np.percentile(jumps, 99)), 2),
        drift_slope_p95_uv_s=round(slope_p95, 2),
    )
    return thr, prof, stats


def inject_saturation(x, fs, seed, k=8.0):
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = int(rng.integers(0, y.shape[0]))
    lo, hi = int(rng.integers(0, y.shape[1] // 3)), int(rng.integers(2 * y.shape[1] // 3, y.shape[1]))
    y[ch, lo:hi] = np.clip(y[ch, lo:hi] * k, -SAT_LIMIT_UV, SAT_LIMIT_UV)
    return y, 'насыщение', ch


def inject_step(x, fs, seed, amp=None):
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = int(rng.integers(0, y.shape[0]))
    at = int(rng.integers(int(fs * 10), y.shape[1] - int(fs * 10)))
    amp = float(rng.uniform(300, 800)) if amp is None else float(amp)
    y[ch, at:] += amp
    return y, 'ступенька', ch


def inject_hum(x, fs, seed, a=120.0):
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = int(rng.integers(0, y.shape[0]))
    t = np.arange(y.shape[1]) / fs
    y[ch] += a * np.sin(2 * np.pi * 50.0 * t)
    return y, 'наводка', ch


def inject_flat(x, fs, seed):
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = int(rng.integers(0, y.shape[0]))
    y[ch] = rng.normal(0, 0.05, y.shape[1])
    return y, 'плоский канал', ch


def inject_drift(x, fs, seed, slope_uv_s=22.0):
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = int(rng.integers(0, y.shape[0]))
    t = np.arange(y.shape[1]) / fs
    y[ch] += slope_uv_s * (t - t.mean())
    return y, 'дрейф базовой линии', ch


def inject_ref_crosstalk(x, fs, seed):
    rng = np.random.default_rng(seed)
    y = x.copy()
    t = np.arange(y.shape[1]) / fs
    common_ref = 180.0 * np.sin(2 * np.pi * 7.5 * t) + rng.normal(0, 140.0, y.shape[1])
    y = y * 0.15 + common_ref[None, :]
    return y, 'помеха референта', -1


INJECTORS = [inject_saturation, inject_step, inject_hum, inject_flat, inject_drift, inject_ref_crosstalk]


def apply_rules(rules_override=None, profile_override=None):
    Q.RULESET.clear()
    Q.RULESET.update(BASE_RULES)
    Q.DEVICE_PROFILE.clear()
    Q.DEVICE_PROFILE.update(BASE_PROFILE)
    if rules_override:
        Q.RULESET.update(rules_override)
    if profile_override:
        Q.DEVICE_PROFILE.update(profile_override)


def run_pass(subjects, rules_override, profile_override, label):
    apply_rules(rules_override, profile_override)
    rows = []
    for sub in subjects:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, _, fs = pair
        chq, sess, _ = Q.analyze(x1, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
        rows.append(dict(
            condition=label,
            subject=sub,
            blocked=int(sess['block']),
            usable=len(sess['usable']),
            profile_ok=int(sess['device_profile_ok']),
            noise_uv=round(float(np.median([r.noise_brain_uv for r in chq])), 1),
        ))
    return pd.DataFrame(rows)


def run_detection(subjects, rules_override, profile_override, label, n_per_type=15):
    apply_rules(rules_override, profile_override)
    rows = []
    seed = 7
    for inj in INJECTORS:
        done = 0
        for sub in subjects:
            if done >= n_per_type:
                break
            pair = load_pair(sub)
            if pair is None:
                continue
            x1, _, fs = pair
            y, name, ch_idx = inj(x1, fs, seed)
            seed += 1
            chq, sess, _ = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            if ch_idx == -1:
                caught = int(bool(sess['ref_crosstalk']) or bool(sess['block']))
                failed = int(bool(sess['block']))
            else:
                hit = chq[ch_idx]
                caught = int(hit.verdict in ('FAIL', 'WARN'))
                failed = int(hit.verdict == 'FAIL')
            rows.append(dict(
                defect=name,
                subject=sub,
                caught=caught,
                failed=failed,
                blocked=int(sess['block']),
            ))
            done += 1
    return pd.DataFrame(rows)


def sweep_detection(subjects, rules_override, profile_override, label, n_per=12):
    apply_rules(rules_override, profile_override)
    rows = []
    seed = 101
    steps = [20, 30, 50, 75, 100, 200, 500]
    sats = [2, 3, 5, 8, 15]
    hums = [3, 5, 10, 20, 50, 120]
    for sub in subjects[:max(n_per, 1)]:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, _, fs = pair
        for a in steps:
            y, _, ci = inject_step(x1, fs, seed, amp=a); seed += 1
            chq, _, _ = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='ступенька', level=a, caught=int(chq[ci].verdict in ('FAIL', 'WARN'))))
        for k in sats:
            y, _, ci = inject_saturation(x1, fs, seed, k=k); seed += 1
            chq, _, _ = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='насыщение', level=k, caught=int(chq[ci].verdict in ('FAIL', 'WARN'))))
        for a in hums:
            y, _, ci = inject_hum(x1, fs, seed, a=a); seed += 1
            chq, _, _ = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='наводка', level=a, caught=int(chq[ci].verdict in ('FAIL', 'WARN'))))
    df = pd.DataFrame(rows)
    df['cond'] = label
    return df


def summarize_pass(df):
    spec = round(float((1 - df['blocked'].mean()) * 100), 1)
    fpr = round(float(df['blocked'].mean() * 100), 1)
    return dict(
        n=int(len(df)),
        pass_share=spec,
        specificity_pct=spec,
        block_share=fpr,
        false_rejection_pct=fpr,
        profile_ok_share=round(float(df['profile_ok'].mean() * 100), 1),
        mean_usable=round(float(df['usable'].mean()), 2),
    )


def summarize_detection(df, label):
    sens = round(float(df['caught'].mean() * 100), 1)
    per = (df.groupby('defect')['caught'].mean() * 100).round(1).to_dict()
    return dict(
        condition=label,
        n_tests=int(len(df)),
        caught_share=sens,
        sensitivity_pct=sens,
        blocked_share=round(float(df['blocked'].mean() * 100), 1),
        by_defect_pct=per,
    )


def main():
    subs = sorted({p.name[:4] for p in DATA.glob('S*R01.edf') if (DATA / f'{p.name[:4]}R02.edf').exists()})
    if not subs:
        raise SystemExit('data_cache/ пуст — сначала запустите bash fetch_all.sh')
    calib, holdout = split_subjects(subs)
    noise, amp, jumps, slopes = measure_profile(calib)
    thr, prof, stats = derive_thresholds(noise, amp, jumps, slopes)

    # Три режима для честного сравнения:
    # A_strict: базовые дефолты сухой гарнитуры без калибровки
    # A_fixed_lab: типовой неадаптивный лабораторный порог (sat=350 мкВ, RMS до 55 мкВ)
    # B_calibrated: адаптивные пороги, выведенные из профиля прибора на первой половине
    fixed_lab_rules = dict(sat_uv=350.0, step_uv=45.0, flat_brain_uv=1.0)
    fixed_lab_prof = dict(rms_brain_uv=(1.5, 55.0))

    A = run_pass(holdout, None, None, 'A: дефолты без калибровки')
    A_lab = run_pass(holdout, fixed_lab_rules, fixed_lab_prof, 'A_fix: фиксированный порог')
    B = run_pass(holdout, thr, dict(rms_brain_uv=prof), 'B: адаптивный профиль прибора')

    det_A = run_detection(holdout, None, None, 'A default')
    det_Alab = run_detection(holdout, fixed_lab_rules, fixed_lab_prof, 'A fixed lab')
    det_B = run_detection(holdout, thr, dict(rms_brain_uv=prof), 'B calibrated')

    sw_A = sweep_detection(holdout, None, None, 'A')
    sw_B = sweep_detection(holdout, thr, dict(rms_brain_uv=prof), 'B')

    res = dict(
        engine=BASE_RULES['version'],
        scheme='holdout: калибровка на 54 чел., независимая проверка на 55 чел.',
        calibration_subjects=len(calib),
        holdout_subjects=len(holdout),
        profile=stats,
        thresholds_derived=thr,
        profile_rms_bounds_uv=list(prof),
        A_default=summarize_pass(A),
        A_fixed_lab=summarize_pass(A_lab),
        B_calibrated=summarize_pass(B),
        detection_A=summarize_detection(det_A, 'A default'),
        detection_A_fixed_lab=summarize_detection(det_Alab, 'A fixed lab'),
        detection_B=summarize_detection(det_B, 'B calibrated'),
        sweep={
            lbl: {
                d: {str(l): int(v) for (dd, l), v in sw.groupby(['defect', 'level'])['caught'].mean().mul(100).round(0).items() if dd == d}
                for d in sw['defect'].unique()
            }
            for lbl, sw in [('A', sw_A), ('B', sw_B)]
        },
    )

    apply_rules()
    pd.concat([A, A_lab, B]).to_csv(OUT / 'calibration_holdout.csv', index=False)
    pd.concat([sw_A, sw_B]).to_csv(OUT / 'calibration_sweep.csv', index=False)
    pd.concat([det_A.assign(cond='A'), det_Alab.assign(cond='A_fix'), det_B.assign(cond='B')]).to_csv(
        OUT / 'calibration_detection.csv', index=False
    )
    (OUT / 'calibration.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({
        'A_default': res['A_default'],
        'A_fixed_lab': res['A_fixed_lab'],
        'B_calibrated': res['B_calibrated'],
        'detection_B': res['detection_B'],
    }, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
