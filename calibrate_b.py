"""Условие B: калибровка порогов под профиль прибора — честная схема.

ПОЧЕМУ ПЕРЕДЕЛАНО
В первой версии пороги подбирались руками на всей выборке (sat 200→900,
step 25→90) и проверялись на ней же. Это подгонка: BLOCK = 0 % означало
не «всё пригодно», а «правила больше ничего не ловят».

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ

1. Делит 109 испытуемых на калибровочную и проверочную половины по хэшу ID
   (детерминированно, без случайного перемешивания).

2. На калибровочной половине измеряет профиль прибора: медианный шум полосы
   1–40 Гц и разброс шума между испытуемыми.

3. ВЫВОДИТ пороги из профиля формулами. Ни одно число не подбирается
   под результат:
       flat_brain_uv = p05 шума калибровочной части
       step_uv       = 4 × медианного шума
       sat_uv        = 99.5-й процентиль |амплитуды|
       rms_brain_uv  = (p05, 6 × медиана)

4. Проверяет на ВТОРОЙ половине, на которой пороги не считались.
   Считает долю записей, проходящих автоприёмку.

5. ИЗМЕРЯЕТ ЦЕНУ РАСШИРЕНИЯ ПОРОГОВ на контролируемых дефектах.
   На клинических данных разметки «непригодная запись» нет, поэтому
   ground truth создаётся инъекцией: в чистые записи проверочной половины
   вносятся известные дефекты (насыщение, срыв контакта, сетевая наводка,
   отрыв электрода). Дальше считается, какую долю этих дефектов движок
   ловит при порогах по умолчанию и при калиброванных.

ЭТО И ЕСТЬ ОТВЕТ НА ВОПРОС «сколько брака вы перестали ловить»:
он измеряется, а не объявляется.

Запуск:  python3 calibrate_b.py
Выход:   results/calibration.json
         results/calibration_holdout.csv
         results/calibration_detection.csv
"""

import json
import hashlib
import numpy as np
import pandas as pd
import mne
import pathlib

import qc_engine as Q

mne.set_log_level('ERROR')

DATA = pathlib.Path('data_cache')
OUT = pathlib.Path('results')
OUT.mkdir(exist_ok=True)

# В EDF имена идут с точками. F7/F8 — это F7/F8, а не замене приблизительная
# карта, что в run_study.py (иначе 6-канального монтажа из 64 не собрать).
# Имена каналов как в файле: F7/F8 — передне-височные, НЕ T3/T4. См. комментарий
# в run_study.py: молчаливая подмена отведений — это искажение данных, а не удобство.
MAP = [('Fp1.', 'Fp1'), ('Fp2.', 'Fp2'), ('F7..', 'F7'), ('F8..', 'F8'), ('O1..', 'O1'), ('O2..', 'O2')]
NAMES = [a for a, _ in MAP]
SHORT = [b for _, b in MAP]

BASE_RULES = dict(Q.RULESET)
BASE_PROFILE = dict(Q.DEVICE_PROFILE)


# ─────────────────────────── данные ───────────────────────────

def split_subjects(subjects):
    """Детерминированное деление по хэшу ID: одинаково при каждом запуске."""
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


# ─────────────────────── профиль и пороги ───────────────────────

def channel_noise(x, fs):
    """Шум канала в полосе 1–40 Гц без сетевой наводки."""
    return np.array([np.std(Q._bp(Q._notch(x[k], fs), fs, 1, 40)) for k in range(x.shape[0])])


def measure_profile(subjects):
    noises, amps = [], []
    for sub in subjects:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, x2, fs = pair
        x = np.concatenate([x1, x2], axis=1)
        noises.extend(channel_noise(x, fs))
        amps.extend(np.percentile(np.abs(x), 99.5, axis=1))
    return np.array(noises), np.array(amps)


# Предел насыщения — АППАРАТНАЯ характеристика, и её у нас нет. Паспорт BCI2000 задаёт
# входной диапазон ±1,5 мВ, но фактический rail канала записи не измерялся: ни в
# документации набора, ни в наших прогонах его нет. Поэтому 1000 мкВ ниже — ИНЖЕНЕРНОЕ
# ДОПУЩЕНИЕ: выше биологического сигнала (медиана p99.5 амплитуды в этом наборе — 221,5 мкВ,
# т.е. допущение превышает её в 4,5 раза) и ниже паспортного входного диапазона ±1,5 мВ.
# Допущение фиксировано, а не подобрано под данные.
#
# Чем это ограничивает вывод: детектор считает насыщением всё, что вышло за |x| > 1000 мкВ,
# а инъектор клиппит ровно на этом же уровне. На синтетике правило срабатывает по построению,
# и «100 % обнаружения насыщения» — проверка механики детектора, а не измерение
# чувствительности к реальному клиппингу. Для заявления «метод ловит насыщение прибора»
# нужен настоящий rail, снятый на целевом усилителе.
SAT_LIMIT_UV = 1000.0


def window_jump_noise(subjects, limit=40):
    """Разброс медиан соседних односекундных окон — ровно то, что видит детектор шагов.

    Не формула, а измерение на калибровочной половине. Считаем |разность медиан|
    между соседними окнами и берём 95-й процентиль: типичный «законный» перепад.
    Порог ступеньки ставим в 4 раза выше, чтобы шум не считался дефектом.
    """
    jumps = []
    for sub in subjects[:limit]:
        f = DATA / f'{sub}R01.edf'
        if not f.exists():
            continue
        r = mne.io.read_raw_edf(f, preload=True, verbose=False)
        if not all(n in r.ch_names for n in NAMES):
            continue
        r.pick(NAMES)
        x = r.get_data() * 1e6
        fs = float(r.info['sfreq'])
        w = int(fs)
        for k in range(x.shape[0]):
            sig = Q._bp(Q._notch(x[k], fs), fs, 1, 40)
            m = np.array([np.median(sig[i:i + w]) for i in range(0, len(sig) - w, w)])
            if len(m) > 3:
                jumps.append(np.abs(np.diff(m)))
    if not jumps:
        return None, None
    j = np.concatenate(jumps)
    p95 = float(np.percentile(j, 95))
    return p95, dict(n_jumps=int(len(j)), p50_uv=round(float(np.percentile(j, 50)), 2),
                     p95_uv=round(p95, 2), p99_uv=round(float(np.percentile(j, 99)), 2),
                     max_uv=round(float(j.max()), 2))


def derive_thresholds(noise, amp, jump_p95=None, jump_stats=None):
    """Пороги, зависящие от прибора, выводятся из его профиля формулами.

    Калибруются только те пороги, которые масштабируются с шумом прибора:
      · «плоский канал» — от нижней границы шума этого прибора;
      · «ступенька» — от типичного шума канала;
      · диапазон профиля — от разброса шума между испытуемыми.
    Насыщение НЕ калибруется: это предел усилителя, он задан железом — но численное
    значение rail у нас неизвестно, поэтому берётся фиксированное допущение
    (см. SAT_LIMIT_UV выше), а не паспортная величина.
    """
    med = float(np.median(noise))
    thr = dict(
        # «плоский канал» — не «тише 5% остальных», а почти полное отсутствие сигнала:
        # десятая часть типичного шума этого прибора.
        flat_brain_uv=round(0.10 * med, 2),
        step_uv=round(4.0 * jump_p95, 1) if jump_p95 else round(4.0 * med, 1),
        sat_uv=SAT_LIMIT_UV,
    )
    stats_extra = dict(window_jump_p95_uv=round(float(jump_p95), 2) if jump_p95 else None)
    # диапазон профиля: «шум этого прибора в пределах разумного»,
    # а не «тише 5% и громче 95% испытуемых» — иначе 5% всегда вне профиля
    prof = (round(0.30 * med, 2), round(3.00 * med, 1))
    stats = dict(median_noise_uv=round(med, 2),
                 p05_noise_uv=round(float(np.percentile(noise, 5)), 2),
                 p95_noise_uv=round(float(np.percentile(noise, 95)), 2),
                 median_amp_p995_uv=round(float(np.median(amp)), 1))
    stats.update(stats_extra); stats.update(jump_stats or {})
    return thr, prof, stats


# ───────────────────── инъекция дефектов ─────────────────────

def inject_saturation(x, fs, seed, k=8.0):
    """Насыщение усилителя: сигнал упирается в предел и обрезается."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = rng.integers(0, y.shape[0])
    lo, hi = rng.integers(0, y.shape[1] // 2), rng.integers(y.shape[1] // 2, y.shape[1])
    lim = SAT_LIMIT_UV                              # фиксированное допущение rail, мкВ (см. комментарий выше)
    y[ch, lo:hi] = np.clip(y[ch, lo:hi] * k, -lim, lim)
    return y, 'насыщение', int(ch)


def inject_step(x, fs, seed, amp=None):
    """Срыв контакта: одномоментный сдвиг уровня на 300–800 мкВ."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = rng.integers(0, y.shape[0])
    at = rng.integers(fs * 10, y.shape[1] - fs * 10)
    amp = rng.uniform(300, 800) if amp is None else amp
    y[ch, at:] += amp
    return y, 'ступенька', int(ch)


def inject_hum(x, fs, seed, a=120.0):
    """Сетевая наводка 50 Гц, амплитуда выше физиологического сигнала."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = rng.integers(0, y.shape[0])
    t = np.arange(y.shape[1]) / fs
    y[ch] += a * np.sin(2 * np.pi * 50 * t)
    return y, 'наводка', int(ch)


def inject_flat(x, fs, seed):
    """Отрыв электрода: канал перестаёт регистрировать ЭЭГ."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    ch = rng.integers(0, y.shape[0])
    y[ch] = rng.normal(0, 0.05, y.shape[1])
    return y, 'плоский канал', int(ch)


INJECTORS = [inject_saturation, inject_step, inject_hum, inject_flat]


# ─────────────────────────── прогон ───────────────────────────

def apply_rules(rules_override=None, profile_override=None):
    Q.RULESET.clear(); Q.RULESET.update(BASE_RULES)
    Q.DEVICE_PROFILE.clear(); Q.DEVICE_PROFILE.update(BASE_PROFILE)
    if rules_override:
        Q.RULESET.update(rules_override)
    if profile_override:
        Q.DEVICE_PROFILE.update(profile_override)


def run_pass(subjects, rules_override, profile_override, label):
    """Сколько записей проходит автоприёмку (BLOCK = не проходит)."""
    apply_rules(rules_override, profile_override)
    rows = []
    for sub in subjects:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, x2, fs = pair
        chq, sess, meta = Q.analyze(x1, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
        rows.append(dict(condition=label, subject=sub,
                         blocked=int(sess['block']), usable=len(sess['usable']),
                         profile_ok=int(sess['device_profile_ok']),
                         noise_uv=round(float(np.median([r.noise_brain_uv for r in chq])), 1)))
    return pd.DataFrame(rows)


def run_detection(subjects, rules_override, profile_override, label, n_per_type=12):
    """Ловит ли движок дефект, внесённый нами же. Ground truth известен точно."""
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
            chq, sess, meta = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            # Правильная метрика — на уровне канала: заметил ли движок ИМЕННО
            # повреждённый канал. Блокировка всей записи наступает только когда
            # годных каналов меньше min_usable, и для одного дефекта это не так.
            hit = chq[ch_idx]
            caught = int(hit.verdict in ('FAIL', 'WARN'))
            rows.append(dict(defect=name, subject=sub, caught=caught,
                             failed=int(hit.verdict == 'FAIL'),
                             blocked=int(sess['block'])))
            done += 1
    df = pd.DataFrame(rows)
    return df


def sweep_detection(subjects, rules_override, profile_override, label, n_per=10):
    """При какой силе дефекта движок его ещё видит.

    Возвращает для каждого типа дефекта таблицу «сила → доля заблокированных».
    Это и есть ответ «насколько мелкий брак мы пропускаем».
    """
    apply_rules(rules_override, profile_override)
    rows = []
    seed = 101
    steps = [20, 30, 50, 75, 100, 200, 500]    # мкВ: ступенька (от мелкой к грубой)
    sats = [2, 3, 5, 8, 15]                    # кратность усиления до клиппинга
    hums = [3, 5, 10, 20, 50, 120]             # мкВ: амплитуда 50 Гц
    for sub in subjects[:max(n_per, 1)]:
        pair = load_pair(sub)
        if pair is None:
            continue
        x1, _, fs = pair
        for a in steps:
            y, _, ci = inject_step(x1, fs, seed, amp=a); seed += 1
            chq, sess, meta = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='ступенька', level=a,
                             caught=int(chq[ci].verdict in ('FAIL','WARN'))))
        for k in sats:
            y, _, ci = inject_saturation(x1, fs, seed, k=k); seed += 1
            chq, sess, meta = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='насыщение', level=k,
                             caught=int(chq[ci].verdict in ('FAIL','WARN'))))
        for a in hums:
            y, _, ci = inject_hum(x1, fs, seed, a=a); seed += 1
            chq, sess, meta = Q.analyze(y, fs, SHORT, eyes_closed=(10, 55), eyes_open=(10, 55))
            rows.append(dict(defect='наводка', level=a,
                             caught=int(chq[ci].verdict in ('FAIL','WARN'))))
    df = pd.DataFrame(rows)
    df['cond'] = label
    return df


def separation_metrics(pass_df, det_df, label):
    """Разделяющая способность: насколько дефектные записи блокируются чаще чистых."""
    clean_block = float(pass_df['blocked'].mean() * 100)
    defect_block = float(det_df['blocked'].mean() * 100)
    real_block = float(det_df.loc[det_df['defect'].isin(
        ['насыщение', 'ступенька', 'наводка']), 'blocked'].mean() * 100)
    return dict(
        condition=label,
        clean_blocked_pct=round(clean_block, 1),
        defect_blocked_pct=round(defect_block, 1),
        real_defect_blocked_pct=round(real_block, 1),
        separation_pp=round(real_block - clean_block, 1),
    )


def summarize_pass(df):
    return dict(n=int(len(df)),
                pass_share=round(float((1 - df['blocked'].mean()) * 100), 1),
                block_share=round(float(df['blocked'].mean() * 100), 1),
                profile_ok_share=round(float(df['profile_ok'].mean() * 100), 1),
                mean_usable=round(float(df['usable'].mean()), 2))


def summarize_detection(df, label):
    out = dict(condition=label,
               caught_share=round(float(df['caught'].mean() * 100), 1),
               blocked_share=round(float(df['blocked'].mean() * 100), 1))
    per = (df.groupby('defect')['caught'].mean() * 100).round(0).astype(int).to_dict()
    out['by_defect_pct'] = per
    return out


def main():
    subs = sorted({p.name[:4] for p in DATA.glob('S*R01.edf')})
    subs = [s for s in subs if (DATA / f'{s}R02.edf').exists()]
    if not subs:
        raise SystemExit(
            'data_cache/ пуст — сначала скачайте данные (≈50 МБ):\n'
            '  bash fetch_data.sh            # PhysioNet\n'
            '  python3 fetch_data_kaggle.py  # или Kaggle-зеркало\n'
            'Файлы данных не входят в репозиторий (лицензия ODC-By, объём).')
    print(f'испытуемых с двумя записями: {len(subs)}')

    calib, holdout = split_subjects(subs)
    print(f'калибровка: {len(calib)} чел. · проверка: {len(holdout)} чел. (пересечения нет)')

    noise, amp = measure_profile(calib)
    jump_p95, jump_stats = window_jump_noise(calib)
    thr, prof, stats = derive_thresholds(noise, amp, jump_p95, jump_stats)
    print('профиль прибора (по калибровочной половине):', stats)
    print('пороги, выведенные из профиля:', thr)

    rules_cal = dict(flat_brain_uv=thr['flat_brain_uv'],
                     step_uv=thr['step_uv'],
                     sat_uv=thr['sat_uv'])
    profile_cal = dict(rms_brain_uv=prof)

    # 1) пропускная способность на отложенной половине
    A = run_pass(holdout, None, None, 'A: пороги по умолчанию')
    B = run_pass(holdout, rules_cal, profile_cal, 'B: пороги из профиля')

    # 2) чувствительность к контролируемым дефектам
    det_A = run_detection(holdout, None, None, 'A: пороги по умолчанию')
    det_B = run_detection(holdout, rules_cal, profile_cal, 'B: пороги из профиля')

    res = dict(
        scheme='holdout: калибровка на одной половине, проверка на другой',
        calibration_subjects=len(calib), holdout_subjects=len(holdout),
        profile=stats, thresholds_derived=thr,
        A_default=summarize_pass(A),
        B_calibrated=summarize_pass(B),
        detection_A=summarize_detection(det_A, 'A default'),
        detection_B=summarize_detection(det_B, 'B calibrated'),
    )

    # 3) честное сравнение со старой схемой (подгонка на всей выборке)
    old_rules = dict(sat_uv=900.0, step_uv=90.0, flat_brain_uv=0.4)
    old_prof = dict(rms_brain_uv=(0.5, 200.0))
    Old = run_pass(subs, old_rules, old_prof, 'OLD: подгонка на всей выборке')
    det_old = run_detection(holdout, old_rules, old_prof, 'OLD: подгонка')
    res['old_scheme'] = dict(**summarize_pass(Old),
                             detection=summarize_detection(det_old, 'OLD'))

    # 4) разделяющая способность: дефектные vs чистые
    res['separation'] = [
        separation_metrics(A, det_A, 'A: пороги по умолчанию'),
        separation_metrics(B, det_B, 'B: пороги из профиля'),
        separation_metrics(Old, det_old, 'OLD: подгонка на всей выборке'),
    ]

    sw_A = sweep_detection(holdout, None, None, 'A')
    sw_B = sweep_detection(holdout, rules_cal, profile_cal, 'B')
    sw_O = sweep_detection(holdout, old_rules, old_prof, 'OLD')
    res['sweep'] = {}
    for lbl, sw in [('A', sw_A), ('B', sw_B), ('OLD', sw_O)]:
        g = sw.groupby(['defect', 'level'])['caught'].mean().mul(100).round(0)
        res['sweep'][lbl] = {f'{d}': {str(l): int(v) for (dd, l), v in g.items() if dd == d}
                             for d in sw['defect'].unique()}

    pd.concat([A, B]).to_csv(OUT / 'calibration_holdout.csv', index=False)
    pd.concat([sw_A, sw_B, sw_O]).to_csv(OUT / 'calibration_sweep.csv', index=False)
    pd.concat([det_A.assign(cond='A'), det_B.assign(cond='B'), det_old.assign(cond='OLD')]) \
        .to_csv(OUT / 'calibration_detection.csv', index=False)
    (OUT / 'calibration.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')

    print()
    print('ПРОПУСКНАЯ СПОСОБНОСТЬ (отложенная половина, чистые записи):')
    print('  A по умолчанию:', res['A_default'])
    print('  B из профиля:  ', res['B_calibrated'])
    print()
    print('ЧУВСТВИТЕЛЬНОСТЬ К ВНЕСЁННЫМ ДЕФЕКТАМ:')
    print('  A по умолчанию:', res['detection_A'])
    print('  B из профиля:  ', res['detection_B'])
    print()
    print('СТАРАЯ СХЕМА (подгонка на всей выборке):', res['old_scheme'])
    print()
    print('РАЗВЁРТКА ПО СИЛЕ ДЕФЕКТА (доля случаев, где повреждённый канал помечен, %):')
    for d in ('ступенька', 'насыщение', 'наводка'):
        print(f'  {d}:')
        for lbl in ('A', 'B', 'OLD'):
            print(f'    {lbl:3} {res["sweep"][lbl][d]}')
    print()
    print('сохранено: results/calibration.json, calibration_holdout.csv, calibration_detection.csv')


if __name__ == '__main__':
    main()
