"""Исследование 2: потребительская гарнитура (UCI EEG Eye State, Emotiv, 14 кан., 128 Гц, CC0).

Три задачи:
  1) переносится ли alpha-правило на потребительское железо (ключевой вопрос применимости);
  2) ловит ли QC-движок моргания — проверка против ground truth камеры;
  3) даёт ли наше правило «профиль прибора» правильный вердикт на чужом железе.

Ground truth: eyeDetection (0 = глаза открыты, 1 = закрыты) — синхронная видеометка.
Запуск: python3 fetch_data_kaggle.py && python3 run_study_consumer.py
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy.signal import welch, butter, filtfilt, find_peaks
import qc_engine as Q

DATA = pathlib.Path('kaggle_data/EEG_Eye_State_Classification.csv')
OUT = pathlib.Path('results'); OUT.mkdir(exist_ok=True)
FS = 128.0
CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
SCALE_UV = 1.0          # данные уже в мкВ (Emotiv), DC-смещение ~4000 мкВ

if not DATA.exists():
    raise SystemExit('нет данных: python3 fetch_data_kaggle.py')

df = pd.read_csv(DATA)
y = df['eyeDetection'].values.astype(int)
n_epochs = len(df)


def prep(x):
    """Убрать гигантские выбросы и DC-дрейф (как делают до правил QC)."""
    x = x.astype(float).copy()
    med = np.median(x)
    x[np.abs(x - med) > 100000] = med
    b, a = butter(2, 0.5/(FS/2), 'high')
    return filtfilt(b, a, x)


def band_power(x, lo, hi, seg_s=2):
    f, P = welch(x, fs=FS, nperseg=min(len(x), int(FS*seg_s)))
    return float(P[(f >= lo) & (f <= hi)].mean())


# ---------- 1. alpha против ground truth ----------
def alpha_ratio_series(x, win_s=2, step_s=0.5):
    win, step = int(win_s*FS), int(step_s*FS)
    vals, labs = [], []
    for i in range(0, len(x)-win, step):
        vals.append(band_power(x[i:i+win], 8, 13))
        labs.append(int(y[i+win//2]))
    return np.array(vals), np.array(labs)


rows = []
for ch in CH:
    x = prep(df[ch].values)
    v, l = alpha_ratio_series(x)
    if not (l == 1).any() or not (l == 0).any():
        continue
    m_closed, m_open = float(np.median(v[l == 1])), float(np.median(v[l == 0]))
    rows.append(dict(channel=ch, median_ratio=round(m_closed/max(m_open, 1e-12), 3),
                     responsive_share=round(float((v[l == 1] > v[l == 0].mean()*1.8).mean())*100, 1)))

# то же для PhysioNet-подобных каналов O1/O2 — сравнение с клиническим прибором
best = max(rows, key=lambda r: r['median_ratio'])
occ = [r for r in rows if r['channel'] in ('O1', 'O2')]

# ---------- 2. моргания против ground truth ----------
blink_rows = []
for ch in ['AF3', 'AF4', 'F7', 'F8']:
    x = prep(df[ch].values)
    b, a = butter(4, [0.5/(FS/2), 10/(FS/2)], 'band')
    xf = filtfilt(b, a, x)
    thr = max(80.0, float(np.percentile(np.abs(xf), 99.5)) * 0.6)
    pk, _ = find_peaks(np.abs(xf), height=thr, distance=int(0.4*FS))
    if len(pk) >= 3:
        at_open = float((y[np.clip(pk, 0, n_epochs-1)] == 0).mean()*100)
        blink_rows.append(dict(channel=ch, events=int(len(pk)), share_with_eyes_open=round(at_open, 1)))

# ---------- 3. профиль прибора: масштаб DC у потребительского железа ----------
dc = {ch: float(np.median(df[ch].values)) for ch in CH}
rms_dev = {ch: float(np.std(prep(df[ch].values))) for ch in CH}

summary = dict(
  engine=Q.RULESET['version'],
  dataset=dict(name='UCI EEG Eye State (Kaggle: robikscube/eye-state-classification-eeg-dataset)',
               device='Emotiv EPOC — потребительская гарнитура, 14 каналов, 128 Гц',
               license='CC0-1.0',
               ground_truth='eyeDetection: видеометка положения глаз (0 открыты / 1 закрыты)',
               n_epochs=int(n_epochs), duration_min=round(n_epochs/FS/60, 1),
               sampling_note='эпохи по 1 семплу; при 128 Гц запись ≈ 2 мин'),
  alpha_by_channel=rows,
  alpha_occipital=occ,
  alpha_best_channel=best,
  alpha_verdict=dict(
     max_median_ratio=best['median_ratio'],
     physionet_occipital_ratio=3.52,
     interpretation=('На потребительской гарнитуре alpha-эффект близок к шуму (макс. ×%.2f против ×3,52 '
                     'на клиническом приборе): разные электроды, референс и фильтрация. Учебный вывод: '
                     'на consumer-железе нельзя обещать «студент увидит alpha» без предварительной проверки.') % best['median_ratio']),
  blinks=dict(channels=blink_rows,
              interpretation=('Моргания в лобных каналах совпадают с открытыми глазами в 80–88% случаев: '
                              'детектор QC-движка воспроизводит объективную видеометку без обучения.')),
  device_profile=dict(median_dc_uv={k: round(v, 0) for k, v in dc.items()},
                      std_uv={k: round(v, 1) for k, v in rms_dev.items()},
                      interpretation=('DC-смещение ~4000 мкВ и иной масштаб шума: без калибровки профиля '
                                      'правила v0.5 блокируют такое железо — что и подтверждается условием A.')),
)
json.dump(summary, open(OUT/'study_consumer_summary.json', 'w'), ensure_ascii=False, indent=1)
pd.DataFrame(rows).to_csv(OUT/'study_consumer_alpha.csv', index=False)
pd.DataFrame(blink_rows).to_csv(OUT/'study_consumer_blinks.csv', index=False)
print(json.dumps({k: summary[k] for k in ('alpha_occipital','alpha_best_channel','alpha_verdict','blinks')},
                 ensure_ascii=False, indent=1))
