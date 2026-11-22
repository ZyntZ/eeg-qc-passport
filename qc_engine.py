"""«Паспорт записи» — QC-движок v0.5 (референсная реализация для стенда и демо).
Принципы:
 1) Техническая годность записи и физиологический результат разделены.
 2) Движок не оценивает человека. Отсутствие alpha — учебная находка, не брак.
 3) Все правила детерминированы и версионированы; пороги — в RULES ниже.
Стенд: синтетические сессии S1/S2/S3 (см. make_sessions.py). На реальном приборе пороги калибруются
до начала работы и фиксируются в версии правил."""
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.signal import welch, butter, filtfilt, find_peaks, iirnotch
from scipy.stats import linregress

DEVICE_PROFILE = dict(name='NeuroPlay-6C (профиль по паспорту)', rms_brain_uv=(1.5, 40.0),
                       note='абсолютные пороги имеют смысл только для откалиброванного профиля прибора')
RULESET = dict(version='0.5.0', sat_uv=200.0, flat_brain_uv=0.8, hum_db=10*np.log10(0.02),
               bridge_corr=0.995, step_uv=25.0, alpha_log10=0.25, blink_uv=80.0,
               min_usable=4, alpha_channels=['O1','O2','T3','T4'])

def _bp(x, fs, lo, hi, order=2):
    b, a = butter(order, [lo/(fs/2), min(hi, fs/2*0.99)/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def _notch(x, fs):
    y = x
    for fr in (50.0, 60.0):
        if fr < fs/2 - 1:
            b, a = iirnotch(fr, 30.0, fs)
            y = filtfilt(b, a, y)
    return y

def _psd(x, fs, nperseg_s=4):
    return welch(x, fs=fs, nperseg=min(len(x), int(fs*nperseg_s)), window='hann')

@dataclass
class ChannelQC:
    name: str
    verdict: str = 'PASS'
    noise_brain_uv: float = 0.0
    flat: bool = False
    sat_pct: float = 0.0
    hum_freq: float = 0.0
    hum_db: float = 0.0
    drift_uv_per_s: float = 0.0
    steps: int = 0
    bridged_with: list = field(default_factory=list)
    alpha_log10: object = None
    alpha_responsive: object = None
    reasons: list = field(default_factory=list)

def detect_blinks(data, fs, frontal_idx=(0,1)):
    """Моргания: пики 0.5–10 Гц в каждом фронтальном канале >80 мкВ; объединение событий."""
    times = set()
    for i in frontal_idx:
        x = _notch(_bp(data[i], fs, 0.5, 10), fs)
        pk, _ = find_peaks(x, height=RULESET['blink_uv'], distance=int(2*fs))
        for p in pk:
            times.add(round(float(p)/fs, 1))
    return np.array(sorted(times))

def detect_steps(data, fs, win_s=1.0, thr=None):
    """Ступеньки/дрейф контакта: скачок медианы между соседними окнами > 25 мкВ."""
    thr = thr or RULESET['step_uv']
    w = int(win_s*fs); steps = []
    for k in range(data.shape[0]):
        meds = [np.median(data[k][i:i+w]) for i in range(0, len(data[k])-w, w)]
        for i in range(1, len(meds)):
            if abs(meds[i]-meds[i-1]) > thr:
                steps.append((k, i*win_s))
    return steps

def analyze(data, fs, ch, eyes_closed=(36, 74), eyes_open=(76, 102)):
    x = np.asarray(data, float)
    n = x.shape[1]
    blinks = detect_blinks(x, fs)
    step_events = detect_steps(x, fs)
    excl = np.zeros(n, bool)
    for tb in blinks:
        excl[int(max(0, (tb-0.6)*fs)):int(min(n/fs, (tb+0.6)*fs))] = True
    for k, ts in step_events:                      # ±1 с вокруг ступеньки — не проверяем
        excl[int(max(0,(ts-1)*fs)):int(min(n/fs,(ts+2)*fs))] = True
    keep = ~excl
    rows = []
    occ = []
    for k, c in enumerate(ch):
        s = x[k]
        r = ChannelQC(name=c)
        sg = s[keep]
        # «мозговой» шум: 1–40 Гц без 48–52 (наводка не считается биологическим шумом)
        brain = _bp(_notch(sg, fs), fs, 1, 40) if len(sg) > fs*4 else sg
        r.noise_brain_uv = float(np.std(brain))
        r.flat = bool(r.noise_brain_uv < RULESET['flat_brain_uv'])
        r.sat_pct = float(np.mean(np.abs(sg) > RULESET['sat_uv'])*100) if len(sg) else 0.0
        for fr in (50.0, 60.0):
            if fr >= fs/2: continue
            f, P = _psd(sg, fs)
            hum = P[(f > fr-1) & (f < fr+1)].sum(); tot = P[(f >= 1) & (f <= 60)].sum()
            db = 10*np.log10(max(hum, 1e-15)/max(tot, 1e-15))
            if r.hum_freq == 0 or db > r.hum_db:
                r.hum_db, r.hum_freq = float(db), fr
        if len(sg) > fs*2:
            r.drift_uv_per_s = float(linregress(np.arange(len(sg))/fs, sg).slope)
        r.steps = sum(1 for kk, _ in step_events if kk == k)
        # alpha: только в «спокойных» окнах с открытыми/закрытыми глазами
        ec = np.zeros(n, bool); eo = np.zeros(n, bool)
        ec[int(eyes_closed[0]*fs):int(eyes_closed[1]*fs)] = True
        eo[int(eyes_open[0]*fs):int(eyes_open[1]*fs)] = True
        ec &= keep; eo &= keep
        if ec.sum() > fs*8 and eo.sum() > fs*8:
            fc, Pc = _psd(s[ec], fs, 6); fo, Po = _psd(s[eo], fs, 6)
            bc = Pc[(fc >= 8) & (fc <= 13)].mean(); bo = Po[(fo >= 8) & (fo <= 13)].mean()
            r.alpha_log10 = float(np.log10(max(bc, 1e-15)/max(bo, 1e-15)))
        rows.append(r)
    # электрические «склейки» (соляной мостик)
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            a = x[i][keep]-x[i][keep].mean(); b = x[j][keep]-x[j][keep].mean()
            sd = np.std(a)*np.std(b)
            if sd > 0 and float(np.corrcoef(a, b)[0,1]) > RULESET['bridge_corr']:
                rows[i].bridged_with.append(rows[j].name); rows[j].bridged_with.append(rows[i].name)
    for r in rows:
        if r.flat: r.verdict = 'FAIL'; r.reasons.append('нет контакта: канал не регистрирует ЭЭГ')
        if r.sat_pct > 0.1: r.verdict = 'FAIL'; r.reasons.append(f'насыщение {r.sat_pct:.1f}%')
        if r.hum_db > RULESET['hum_db']:
            if r.verdict == 'PASS': r.verdict = 'WARN'
            r.reasons.append(f'наводка {r.hum_freq:.0f} Гц')
        if r.steps: 
            if r.verdict == 'PASS': r.verdict = 'WARN'
            r.reasons.append(f'ступенька контакта ({r.steps})')
        if abs(r.drift_uv_per_s) > 10:
            if r.verdict == 'PASS': r.verdict = 'WARN'
            r.reasons.append(f'дрейф {r.drift_uv_per_s:.0f} мкВ/с')
        if r.bridged_with:
            if r.verdict == 'PASS': r.verdict = 'WARN'
            r.reasons.append('электроды шунтированы: ' + ', '.join(r.bridged_with))
        usable = r.verdict != 'FAIL' and not r.bridged_with
        r.alpha_responsive = (bool(r.alpha_log10 is not None and r.alpha_log10 >= RULESET['alpha_log10']) if usable else None)
        if not usable:
            r.alpha_log10 = None
    # проверка соответствия профилю прибора: чужой масштаб нельзя сертифицировать автоматически
    rms = float(np.median([r.noise_brain_uv for r in rows]))
    lo, hi = DEVICE_PROFILE['rms_brain_uv']
    device_ok = bool(lo <= rms <= hi)
    usable = [r for r in rows if r.verdict != 'FAIL' and not r.bridged_with]
    occ_rows = [r for r in rows if r.name in RULESET['alpha_channels']]
    occ_usable = [r for r in occ_rows if r.verdict != 'FAIL' and not r.bridged_with]
    block = (len(usable) < RULESET['min_usable']) or (not occ_usable)
    reasons = []
    if len(usable) < RULESET['min_usable']:
        reasons.append(f'годных каналов {len(usable)} < {RULESET["min_usable"]}')
    if not occ_usable:
        reasons.append('нет ни одного годного затылочно-височного канала (O1/O2/T3/T4) для урока об alpha')
    resp = [r.name for r in occ_usable if r.alpha_responsive]
    nonresp = [r.name for r in occ_usable if r.alpha_responsive is False]
    if not device_ok:
        block = True
        reasons.append(f'сигнал не соответствует профилю прибора (медианный RMS {rms:.1f} мкВ вне {lo}–{hi}): '
                       'нужна калибровка/иной профиль, автоприёмка запрещена')
    session = dict(verdict='BLOCK' if block else ('PASS_WITH_WARNINGS' if any(r.verdict=='WARN' for r in rows) else 'PASS'),
                   device_profile_ok=device_ok, rms_median_uv=round(rms,2),
                   block=block, reasons=reasons,
                   usable=[r.name for r in usable], failed=[r.name for r in rows if r.verdict=='FAIL'],
                   occ_usable=[r.name for r in occ_usable],
                   alpha=dict(responsive=resp, not_responsive=nonresp,
                              text=('обнаружена реакция на закрытие глаз: ' + ', '.join(resp)) if resp else
                                   ('реакция на закрытие глаз не обнаружена в годных каналах — это НЕ технический брак')
                                   ))
    meta = dict(blinks_s=[round(float(b),1) for b in blinks], excl_idx=int(excl.sum()),
                step_events=[(ch[k], round(float(ts),1)) for k, ts in step_events],
                excluded_s=round(float(excl.sum()/fs),1), rules=RULESET['version'])
    return rows, session, meta

if __name__ == '__main__':
    import json, csv, subprocess, sys
    if not all(__import__('os').path.exists(f'{k}.npz') for k in ('S1','S2','S3')):
        print('сессии не найдены — генерирую (make_sessions.py)')
        subprocess.run([sys.executable, 'make_sessions.py'], check=True)
    out = {}
    for kind in ('S1','S2','S3'):
        z = np.load(f'{kind}.npz', allow_pickle=True)
        rows, session, meta = analyze(z['data'], float(z['fs']), [str(c) for c in z['ch']])
        out[kind] = dict(meta=meta, channels=[{k:(round(v,3) if isinstance(v,float) else v)
                                               for k,v in asdict(r).items()} for r in rows], session=session)
    json.dump(out, open('qc_result.json','w'), ensure_ascii=False, indent=1)
    for k, v in out.items():
        s = v['session']
        print(f"{k}: {s['verdict']} | годных {len(s['usable'])} {s['usable']} | блок: {s['reasons'] or '—'}")
        for r in v['channels']:
            print(f"    {r['name']:4} {r['verdict']:5} alpha={r['alpha_log10']} resp={r['alpha_responsive']} :: {'; '.join(r['reasons']) or '—'}")
        print('    alpha:', s['alpha']['text'], '| исключено с:', v['meta']['excluded_s'], '| моргания:', v['meta']['blinks_s'])
