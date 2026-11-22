"""Генератор трёх синтетических сессий для демонстрации QC-движка.
ВНИМАНИЕ: данные синтетические (seed=42), не записи людей и не валидация прибора.
S1 «рабочая» — годная запись с ожидаемыми помехами (наводка 50 Гц на части каналов,
   моргания, ступенька контакта на T4) и явным alpha-эффектом O1/O2.
S2 «брак» — плоский Fp1, склейка T3–T4 (соляной мостик), сильная наводка, шум > допустимого:
   технически непригодна, должна быть заблокирована.
S3 «чистая без alpha» — отличная запись без alpha-реакции: должна ПРОЙТИ техпроверку.
   Это прямая проверка требования: отсутствие alpha не является техническим браком."""
import numpy as np, json
from scipy.signal import butter, filtfilt

FS = 125.0
CH = ['Fp1', 'Fp2', 'T3', 'T4', 'O1', 'O2']
DUR = 105.0
T = np.arange(0, DUR, 1/FS)
N = len(T)
BLINKS = [12.0, 28.0, 55.0, 88.0]
EYES_CLOSED = (35.0, 75.0)
STEP = (62.0, 90.0)

def base(rng):
    white = rng.standard_normal((6, N))
    b, a = butter(1, 30/(FS/2), btype='low')
    bg = filtfilt(b, a, white, axis=1)
    return bg/np.std(bg, axis=1, keepdims=True)*8.0

def narrow(rng, lo, hi, amp):
    x = rng.standard_normal(N)
    b, a = butter(2, [lo/(FS/2), hi/(FS/2)], btype='band')
    y = filtfilt(b, a, x); return y/np.std(y)*amp

def blinks():
    s = np.zeros(N)
    for t0 in BLINKS:
        i = (T >= t0) & (T < t0+0.6); tt = T[i]-t0
        s[i] += 120*np.exp(-((tt-0.15)/0.09)**2)
    g = dict(Fp1=1.0, Fp2=.95, T3=.25, T4=.22, O1=.08, O2=.08)
    return np.vstack([s*g[c] for c in CH])

def alpha(rng, amp_scale=1.0):
    a = narrow(rng, 8.5, 11.5, 1.0)
    env = np.zeros(N); env[(T >= EYES_CLOSED[0]) & (T < EYES_CLOSED[1])] = 1.0
    env = np.convolve(env, np.ones(125)/125, mode='same')
    g = dict(Fp1=3, Fp2=3, T3=5, T4=4, O1=20, O2=18)
    return np.vstack([a*env*g[c]*amp_scale for c in CH])

def session(kind):
    rng = np.random.default_rng(42)
    x = base(rng) + blinks()
    if kind in ('S1', 'S3'):
        x += alpha(rng, 1.0 if kind == 'S1' else 0.0)
    if kind == 'S1':
        hum = 12*np.sin(2*np.pi*50*T+0.3)
        for i in (0, 2, 5): x[i] += hum*(1+0.1*rng.standard_normal(N))
        d = np.zeros(N); m = (T >= STEP[0]) & (T < STEP[1])
        d[m] = 120 - (T[m]-STEP[0])/28*40
        x[3] += d
    if kind == 'S2':
        x[0] *= 0.02                       # плоский Fp1 (нет контакта)
        x[2] = x[3]*0.985 + 2               # соляной мостик T3–T4
        hum = 45*np.sin(2*np.pi*50*T)
        x += hum[None, :]
    return x

def save(kind, x, note):
    np.savez_compressed(f'{kind}.npz', data=x, fs=FS, ch=CH, t=T,
                        meta=json.dumps(dict(kind=kind, seed=42, note=note, synthetic=True),
                                        ensure_ascii=False))

save('S1', session('S1'), 'рабочая запись: ожидаемые помехи + alpha O1/O2')
save('S2', session('S2'), 'брак: плоский канал, склейка, наводка')
save('S3', session('S3'), 'чистая запись без alpha-реакции')
print('ok', [f'{k}.npz' for k in ('S1','S2','S3')])
