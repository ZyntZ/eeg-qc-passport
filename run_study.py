"""Исследование на публичных данных: «Паспорт записи» v0.5.
Протокол: PhysioNet EEGMMIDB v1.0.0 (Schalk 2009, ODC-By 1.0), ВСЕ 109 добровольцев × 2 baseline-записи.
Условия A и B — проверка «защиты от ложной сертификации» и «работы после калибровки».
Воспроизведение: bash fetch_data.sh && python3 run_study.py
"""
import json, pathlib, numpy as np, pandas as pd, mne
from scipy.signal import welch, iirnotch, filtfilt
import qc_engine as Q

mne.set_log_level('ERROR')
DATA = pathlib.Path('data_cache'); OUT = pathlib.Path('results'); OUT.mkdir(exist_ok=True)
MAP = [('Fp1.','Fp1'),('Fp2.','Fp2'),('F7..','T3'),('F8..','T4'),('O1..','O1'),('O2..','O2')]
names = [a for a,_ in MAP]; short = [b for _,b in MAP]
BASE_RULES = dict(Q.RULESET); BASE_PROFILE = dict(Q.DEVICE_PROFILE)

def notch(x, fs):
    y = x
    for fr in (50.0, 60.0):
        if fr < fs/2 - 1:
            b, a = iirnotch(fr, 30, fs); y = filtfilt(b, a, y)
    return y

def band_power(x, fs, lo=8, hi=13, nperseg_s=8):
    f, P = welch(notch(x, fs), fs=fs, nperseg=min(len(x), int(fs*nperseg_s)))
    return float(P[(f >= lo) & (f <= hi)].mean())

def _require_data():
    if not list(DATA.glob('S*R01.edf')):
        raise SystemExit(
            'data_cache/ пуст — сначала скачайте данные (≈50 МБ):\n'
            '  bash fetch_data.sh            # PhysioNet\n'
            '  python3 fetch_data_kaggle.py  # или Kaggle-зеркало\n'
            'Файлы данных не входят в репозиторий (лицензия ODC-By, объём).')


def collect(condition, rules_override=None, profile_override=None):
    Q.RULESET.clear(); Q.RULESET.update(BASE_RULES)
    Q.DEVICE_PROFILE.clear(); Q.DEVICE_PROFILE.update(BASE_PROFILE)
    if rules_override: Q.RULESET.update(rules_override)
    if profile_override: Q.DEVICE_PROFILE.update(profile_override)
    rows = []
    for sub in sorted({p.name[:4] for p in DATA.glob('S*R01.edf')}):
        f1, f2 = DATA/f'{sub}R01.edf', DATA/f'{sub}R02.edf'
        if not (f1.exists() and f2.exists()): continue
        r1 = mne.io.read_raw_edf(f1, preload=True, verbose=False)
        r2 = mne.io.read_raw_edf(f2, preload=True, verbose=False)
        if not all(n in r1.ch_names for n in names): continue
        r1.pick(names); r2.pick(names)
        x1, x2 = r1.get_data()*1e6, r2.get_data()*1e6
        fs = float(r1.info['sfreq'])
        chq, sess, meta = Q.analyze(x1, fs, short, eyes_closed=(10,55), eyes_open=(10,55))
        rec = dict(condition=condition, subject=sub, sfreq=fs, dur_s=round(x1.shape[1]/fs,1),
                   verdict=sess['verdict'], blocked=sess['block'], usable=len(sess['usable']),
                   profile_ok=sess['device_profile_ok'], rms=round(sess['rms_median_uv'],1),
                   n_blinks=len(meta['blinks_s']),
                   fails=sum(r.verdict=='FAIL' for r in chq), warns=sum(r.verdict=='WARN' for r in chq),
                   steps_total=sum(r.steps for r in chq),
                   hum_any=int(any(r.hum_db > Q.RULESET['hum_db'] for r in chq)),
                   bridge_any=int(any(bool(r.bridged_with) for r in chq)))
        for i, c in enumerate(short):
            pc = band_power(x2[i][int(10*fs):int(55*fs)], fs)
            po = band_power(x1[i][int(10*fs):int(55*fs)], fs)
            lg = float(np.log10(max(pc,1e-12)/max(po,1e-12)))
            rec[f'alpha_{c}'] = round(lg,3); rec[f'ratio_{c}'] = round(float(pc/max(po,1e-12)),2)
            rec[f'noise_{c}'] = round(next(r.noise_brain_uv for r in chq if r.name==c),1)
        rows.append(rec)
    Q.RULESET.clear(); Q.RULESET.update(BASE_RULES)
    Q.DEVICE_PROFILE.clear(); Q.DEVICE_PROFILE.update(BASE_PROFILE)
    return pd.DataFrame(rows)

_require_data()
A = collect('A: наши пороги на чужом приборе (без калибровки)')
B = collect('B: пороги откалиброваны под профиль прибора',
            rules_override=dict(sat_uv=900.0, step_uv=90.0, flat_brain_uv=0.4),
            profile_override=dict(rms_brain_uv=(0.5, 200.0)))
all_rows = pd.concat([A, B]); all_rows.to_csv(OUT/'study_all_conditions.csv', index=False)
B.to_csv(OUT/'study_results.csv', index=False)

def summarize(d):
    return dict(n=len(d), pass_share=round(float((~d['blocked']).mean())*100,1),
                block_share=round(float(d['blocked'].mean())*100,1),
                profile_ok_share=round(float(d['profile_ok'].mean())*100,1),
                mean_usable=round(float(d['usable'].mean()),2),
                median_blinks=float(d['n_blinks'].median()),
                median_steps=float(d['steps_total'].median()),
                share_records_with_hum=round(float(d['hum_any'].mean())*100,1),
                share_records_with_steps=round(float((d['steps_total']>0).mean())*100,1),
                share_records_with_blinks=round(float((d['n_blinks']>0).mean())*100,1),
                share_records_bridged=round(float(d['bridge_any'].mean())*100,1))
alpha = {}
for c in ('O1','O2','T3','T4','Fp1','Fp2'):
    lg = B[f'alpha_{c}']
    alpha[c] = dict(median_log10=round(float(lg.median()),3), median_ratio=round(float(10**lg.median()),2),
                    share_responsive=round(float((lg>=0.25).mean())*100,1))
S = dict(
  engine=Q.RULESET['version'],
  dataset=dict(name='PhysioNet EEGMMIDB v1.0.0', authors='Schalk G. et al. (Wadsworth Center)',
               doi='10.13026/C28G6P', license='ODC-By 1.0', device='BCI2000, 64 канала, 160 Гц',
               runs='R01 глаза открыты, R02 закрыты, ~60 с каждая'),
  design=dict(subjects=int(B['subject'].nunique()), files=int(2*B['subject'].nunique()),
              windows='для alpha — 10–55 с; моргания/ступеньки исключаются автоматически'),
  conditions={'A_без_калибровки': summarize(A), 'B_после_калибровки': summarize(B)},
  alpha_by_channel=alpha,
  share_subjects_occipital_response=round(float(((B['alpha_O1']>=0.25)|(B['alpha_O2']>=0.25)).mean())*100,1),
  share_subjects_no_alpha=round(float((B[['alpha_O1','alpha_O2','alpha_T3','alpha_T4']].max(axis=1)<0.25).mean())*100,1),
  share_subjects_any_channel_response=round(float((B[[f'alpha_{c}' for c in ('O1','O2','T3','T4','Fp1','Fp2')]].max(axis=1)>=0.25).mean())*100,1),
  no_response_subjects=B[(B['alpha_O1']<0.25)&(B['alpha_O2']<0.25)]['subject'].tolist(),
  median_rms_range=[float(B['rms'].min()), float(B['rms'].max())],
)
json.dump(S, open(OUT/'study_summary.json','w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
