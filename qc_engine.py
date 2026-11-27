"""EEG QC Passport — движок контроля качества ЭЭГ v0.6.0.

Принципы:
 1) Техническое качество сигнала и физиологический ответ оцениваются раздельно.
 2) Отсутствие альфа-реакции при чистом сигнале не считается браком (учитывается в т.ч.
    десинхронизированный тип ЭЭГ по Е. А. Жирмунской).
 3) Пороги детерминированы, версионированы и калибруются под шумовой профиль прибора.
 4) Два режима: онлайн-окно при постановке электродов (check_live_window) и итоговый
    паспорт записи после пробы (analyze + render_html_passport).
"""
from dataclasses import dataclass, asdict, field
import hashlib
import html
import numpy as np
from scipy.signal import welch, butter, filtfilt, find_peaks, iirnotch
from scipy.stats import linregress

DEVICE_PROFILE = dict(
    name='NeuroPlay-6C (базовый учебный профиль)',
    rms_brain_uv=(1.5, 40.0),
    note='диапазон RMS калибруется под конкретный прибор перед практикумом',
)

RULESET = dict(
    version='0.6.0',
    sat_uv=200.0,
    flat_brain_uv=0.8,
    hum_db=10 * np.log10(0.02),
    bridge_corr=0.995,
    ref_crosstalk_corr=0.92,
    ref_crosstalk_pair_share=0.70,
    step_uv=25.0,
    drift_uv_s=10.0,
    emg_hf_ratio=0.35,
    imp_warn_kohm=20.0,
    imp_fail_kohm=50.0,
    alpha_log10=0.25,
    blink_uv=80.0,
    min_usable=4,
    alpha_channels=['O1', 'O2'],
)


def _bp(x, fs, lo, hi, order=2):
    hi_eff = min(hi, fs / 2 * 0.98)
    if lo >= hi_eff or len(x) < max(18, int(fs * 0.5)):
        return x - np.mean(x)
    b, a = butter(order, [lo / (fs / 2), hi_eff / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


def _notch(x, fs):
    y = x
    if len(y) < 18:
        return y
    for fr in (50.0, 60.0):
        if fr < fs / 2 - 1:
            b, a = iirnotch(fr, 30.0, fs)
            y = filtfilt(b, a, y)
    return y


def _psd(x, fs, nperseg_s=4):
    return welch(x, fs=fs, nperseg=min(len(x), max(16, int(fs * nperseg_s))), window='hann')


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
    emg_hf_share: float = 0.0
    impedance_kohm: object = None
    alpha_log10: object = None
    alpha_responsive: object = None
    reasons: list = field(default_factory=list)
    physio_notes: list = field(default_factory=list)


def detect_blinks(data, fs, frontal_idx=(0, 1)):
    """Моргания (ЭОГ): пики 0.5–10 Гц во фронтальных каналах."""
    times = set()
    for i in frontal_idx:
        if i >= data.shape[0] or data.shape[1] < int(fs * 2):
            continue
        x = _notch(_bp(data[i], fs, 0.5, 10), fs)
        pk, _ = find_peaks(np.abs(x), height=RULESET['blink_uv'], distance=max(1, int(0.8 * fs)))
        for p in pk:
            times.add(round(float(p) / fs, 1))
    return np.array(sorted(times))


def detect_steps(data, fs, win_s=1.0, thr=None):
    """Ступеньки контакта: скачок медианы между соседними окнами."""
    thr = thr if thr is not None else RULESET['step_uv']
    w = max(1, int(win_s * fs))
    steps = []
    for k in range(data.shape[0]):
        if len(data[k]) < 2 * w:
            continue
        meds = [np.median(data[k][i:i + w]) for i in range(0, len(data[k]) - w + 1, w)]
        for i in range(1, len(meds)):
            if abs(meds[i] - meds[i - 1]) > thr:
                steps.append((k, i * win_s))
    return steps


def _detect_crosstalk_and_bridges(x_clean, rows):
    """Разделяет локальный соляной мостик (r > 0.995) и общую помеху референта (crosstalk)."""
    n_ch = len(rows)
    if n_ch < 2 or x_clean.shape[1] < 16:
        return False, 0.0
    pair_corrs = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            a = x_clean[i] - x_clean[i].mean()
            b = x_clean[j] - x_clean[j].mean()
            sd = np.std(a) * np.std(b)
            if sd <= 1e-12:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            pair_corrs.append(abs(corr))
            if corr > RULESET['bridge_corr']:
                rows[i].bridged_with.append(rows[j].name)
                rows[j].bridged_with.append(rows[i].name)
    if not pair_corrs:
        return False, 0.0
    high_share = float(np.mean([c > RULESET['ref_crosstalk_corr'] for c in pair_corrs]))
    ref_issue = bool(n_ch >= 4 and high_share >= RULESET['ref_crosstalk_pair_share'])
    return ref_issue, round(float(np.median(pair_corrs)), 3)


def analyze(data, fs, ch, eyes_closed=(36, 74), eyes_open=(76, 102), impedances_kohm=None):
    x = np.asarray(data, float)
    n = x.shape[1]
    frontal = tuple(i for i, c in enumerate(ch) if c.startswith(('Fp', 'AF', 'F7', 'F8'))) or (0, min(1, len(ch) - 1))
    blinks = detect_blinks(x, fs, frontal_idx=frontal[:2])
    step_events = detect_steps(x, fs)
    excl = np.zeros(n, bool)
    for tb in blinks:
        excl[int(max(0, (tb - 0.6) * fs)):int(min(n, (tb + 0.6) * fs))] = True
    for _, ts in step_events:
        excl[int(max(0, (ts - 1.0) * fs)):int(min(n, (ts + 2.0) * fs))] = True
    keep = ~excl if (~excl).sum() >= max(16, int(fs * 2)) else np.ones(n, bool)

    rows = []
    for k, c in enumerate(ch):
        s = x[k]
        sg = s[keep]
        r = ChannelQC(name=c)
        if impedances_kohm is not None and k < len(impedances_kohm) and impedances_kohm[k] is not None:
            r.impedance_kohm = round(float(impedances_kohm[k]), 1)

        brain = _bp(_notch(sg, fs), fs, 1, 40) if len(sg) > fs * 2 else (sg - np.mean(sg))
        r.noise_brain_uv = float(np.std(brain))
        r.flat = bool(r.noise_brain_uv < RULESET['flat_brain_uv'])
        r.sat_pct = float(np.mean(np.abs(sg) >= RULESET['sat_uv'] * 0.999) * 100) if len(sg) else 0.0

        f, P = _psd(sg, fs)
        tot_1_60 = P[(f >= 1) & (f <= min(60, fs / 2))].sum()
        for fr in (50.0, 60.0):
            if fr >= fs / 2:
                continue
            hum = P[(f > fr - 1) & (f < fr + 1)].sum()
            db = 10 * np.log10(max(hum, 1e-15) / max(tot_1_60, 1e-15))
            if r.hum_freq == 0 or db > r.hum_db:
                r.hum_db, r.hum_freq = float(db), fr

        # Доля высокочастотной мышечной активности ЭМГ (30–45 Гц, без 50 Гц)
        if fs > 64:
            f_clean, P_clean = _psd(_notch(sg, fs), fs)
            p_tot = P_clean[(f_clean >= 1) & (f_clean <= 45)].sum()
            p_hf = P_clean[(f_clean >= 30) & (f_clean <= 45)].sum()
            r.emg_hf_share = round(float(p_hf / max(p_tot, 1e-15)), 3)

        if len(sg) >= fs * 10:
            r.drift_uv_per_s = float(linregress(np.arange(len(sg)) / fs, sg).slope)
        r.steps = sum(1 for kk, _ in step_events if kk == k)

        ec = np.zeros(n, bool)
        eo = np.zeros(n, bool)
        ec[max(0, int(eyes_closed[0] * fs)):min(n, int(eyes_closed[1] * fs))] = True
        eo[max(0, int(eyes_open[0] * fs)):min(n, int(eyes_open[1] * fs))] = True
        ec &= (~excl)
        eo &= (~excl)
        if ec.sum() > fs * 8 and eo.sum() > fs * 8:
            fc, Pc = _psd(_notch(s[ec], fs), fs, 6)
            fo, Po = _psd(_notch(s[eo], fs), fs, 6)
            bc = Pc[(fc >= 8) & (fc <= 13)].mean()
            bo = Po[(fo >= 8) & (fo <= 13)].mean()
            r.alpha_log10 = float(np.log10(max(bc, 1e-15) / max(bo, 1e-15)))
        rows.append(r)

    ref_crosstalk, median_pair_corr = _detect_crosstalk_and_bridges(x[:, keep], rows)

    for r in rows:
        if r.impedance_kohm is not None:
            if r.impedance_kohm > RULESET['imp_fail_kohm']:
                r.verdict = 'FAIL'
                r.reasons.append(f'импеданс {r.impedance_kohm:.0f} кОм > {RULESET["imp_fail_kohm"]:.0f}')
            elif r.impedance_kohm > RULESET['imp_warn_kohm']:
                if r.verdict == 'PASS':
                    r.verdict = 'WARN'
                r.reasons.append(f'повышенный импеданс {r.impedance_kohm:.0f} кОм')
        if r.flat:
            r.verdict = 'FAIL'
            r.reasons.append('нет контакта (плоский канал)')
        if r.sat_pct > 0.1:
            r.verdict = 'FAIL'
            r.reasons.append(f'насыщение усилителя {r.sat_pct:.1f}%')
        if r.hum_db > RULESET['hum_db']:
            if r.verdict == 'PASS':
                r.verdict = 'WARN'
            r.reasons.append(f'сетевая наводка {r.hum_freq:.0f} Гц')
        if r.steps:
            if r.verdict == 'PASS':
                r.verdict = 'WARN'
            r.reasons.append(f'ступенька контакта ({r.steps})')
        if abs(r.drift_uv_per_s) > RULESET['drift_uv_s']:
            if r.verdict == 'PASS':
                r.verdict = 'WARN'
            r.reasons.append(f'дрейф базовой линии {r.drift_uv_per_s:.1f} мкВ/с')
        if r.bridged_with:
            if r.verdict == 'PASS':
                r.verdict = 'WARN'
            r.reasons.append('шунт электродов: ' + ', '.join(r.bridged_with))
        if r.emg_hf_share > RULESET['emg_hf_ratio'] and not r.flat:
            r.physio_notes.append(f'мышечная активность ЭМГ 30–45 Гц ({r.emg_hf_share * 100:.0f}%)')

        usable_ch = r.verdict != 'FAIL' and not r.bridged_with and not ref_crosstalk
        r.alpha_responsive = (
            bool(r.alpha_log10 is not None and r.alpha_log10 >= RULESET['alpha_log10'])
            if usable_ch else None
        )
        if not usable_ch:
            r.alpha_log10 = None

    rms = float(np.median([r.noise_brain_uv for r in rows])) if rows else 0.0
    lo, hi = DEVICE_PROFILE['rms_brain_uv']
    device_ok = bool(lo <= rms <= hi)
    usable = [r for r in rows if r.verdict != 'FAIL' and not r.bridged_with and not ref_crosstalk]
    occ_rows = [r for r in rows if r.name in RULESET['alpha_channels']]
    occ_usable = [r for r in occ_rows if r.verdict != 'FAIL' and not r.bridged_with and not ref_crosstalk]

    block = (len(usable) < RULESET['min_usable']) or (not occ_usable) or ref_crosstalk
    reasons = []
    if ref_crosstalk:
        reasons.append(f'общая помеха референта / crosstalk (медиана |r|={median_pair_corr:.2f})')
    if len(usable) < RULESET['min_usable']:
        reasons.append(f'годных каналов {len(usable)} < {RULESET["min_usable"]}')
    if not occ_usable:
        reasons.append('нет годного затылочного канала (O1/O2)')
    if not device_ok:
        block = True
        reasons.append(
            f'шум {rms:.1f} мкВ вне профиля прибора ({lo}–{hi} мкВ): нужна калибровка профиля'
        )

    resp = [r.name for r in occ_usable if r.alpha_responsive]
    nonresp = [r.name for r in occ_usable if r.alpha_responsive is False]
    if block:
        alpha_text = 'оценка физиологической реакции не проводится: запись технически непригодна'
    elif resp:
        alpha_text = 'реакция на закрытие глаз обнаружена: ' + ', '.join(resp)
    else:
        alpha_text = (
            'реакция не обнаружена (сигнал технически годен; возможен индивидуальный '
            'низкоамплитудный / десинхронизированный тип ЭЭГ по Е. А. Жирмунской)'
        )

    session = dict(
        verdict='BLOCK' if block else ('PASS_WITH_WARNINGS' if any(r.verdict == 'WARN' for r in rows) else 'PASS'),
        device_profile_ok=device_ok,
        rms_median_uv=round(rms, 2),
        ref_crosstalk=ref_crosstalk,
        median_pair_corr=median_pair_corr,
        block=block,
        reasons=reasons,
        usable=[r.name for r in usable],
        failed=[r.name for r in rows if r.verdict == 'FAIL'],
        occ_usable=[r.name for r in occ_usable],
        physio_artifacts=dict(
            blinks_count=int(len(blinks)),
            emg_channels=[r.name for r in rows if r.physio_notes],
        ),
        alpha=dict(responsive=resp, not_responsive=nonresp, text=alpha_text),
    )
    rules_hash = hashlib.sha256(repr(sorted(RULESET.items())).encode()).hexdigest()[:12]
    meta = dict(
        blinks_s=[round(float(b), 1) for b in blinks],
        excl_idx=int(excl.sum()),
        step_events=[(ch[k], round(float(ts), 1)) for k, ts in step_events],
        excluded_s=round(float(excl.sum() / fs), 1),
        rules=RULESET['version'],
        rules_hash=rules_hash,
    )
    return rows, session, meta


def check_live_window(window_uv, fs, ch, impedances_kohm=None):
    """Онлайн-проверка короткого окна (2–4 с) во время постановки электродов студентом.

    Возвращает светофор по каналам и конкретные рекомендации по устранению брака до старта записи.
    """
    w = np.asarray(window_uv, float)
    dur = w.shape[1] / fs
    rows, sess, _ = analyze(w, fs, ch, eyes_closed=(0, 0), eyes_open=(0, 0), impedances_kohm=impedances_kohm)
    hints = []
    if sess['ref_crosstalk']:
        hints.append('Проверьте контакт референтного и заземляющего электродов (мочка уха / мастоид).')
    for r in rows:
        if r.flat or (r.impedance_kohm is not None and r.impedance_kohm > RULESET['imp_fail_kohm']):
            hints.append(f'{r.name}: нет контакта — поправьте электрод на коже / добавьте гель.')
        elif r.sat_pct > 0.1:
            hints.append(f'{r.name}: перегрузка усилителя — дождитесь стабилизации потенциала электрода.')
        elif r.bridged_with:
            hints.append(f'{r.name}–{",".join(r.bridged_with)}: соляной мостик — уберите лишний гель между точками.')
        elif r.hum_db > RULESET['hum_db']:
            hints.append(f'{r.name}: сетевая наводка {r.hum_freq:.0f} Гц — улучшите прижим и отодвиньте блок питания.')
        elif abs(r.drift_uv_per_s) > RULESET['drift_uv_s'] or r.steps:
            hints.append(f'{r.name}: плывёт базовая линия — зафиксируйте кабель и попросите испытуемого не двигать головой.')
        if r.physio_notes:
            hints.append(f'{r.name}: мышечный шум ЭМГ — попросите расслабить челюсть, лоб и плечи.')
    ready = (not sess['ref_crosstalk']) and len(sess['usable']) >= RULESET['min_usable'] and bool(sess['occ_usable'])
    return dict(
        window_s=round(float(dur), 1),
        ready_to_record=bool(ready),
        status='READY' if ready else 'FIX_ELECTRODES',
        channels={r.name: r.verdict for r in rows},
        hints=hints or ['Контакт всех рабочих каналов в норме. Можно начинать запись пробы.'],
    )


def render_html_passport(title, rows, session, meta, file_hashes=None):
    """Самодостаточный HTML-паспорт записи без внешних стилей и сети."""
    badge_color = {'PASS': '#1f7a4c', 'PASS_WITH_WARNINGS': '#b76e00', 'BLOCK': '#b42318'}[session['verdict']]
    badge_label = {
        'PASS': 'ГОДНА (PASS)',
        'PASS_WITH_WARNINGS': 'ГОДНА С ЗАМЕЧАНИЯМИ (WARN)',
        'BLOCK': 'ТЕХНИЧЕСКИЙ БРАК (BLOCK)',
    }[session['verdict']]
    ch_rows = []
    for r in rows:
        vc = {'PASS': '#1f7a4c', 'WARN': '#b76e00', 'FAIL': '#b42318'}[r.verdict]
        ratio_txt = f'×{10 ** r.alpha_log10:.2f} (log10={r.alpha_log10:+.2f})' if r.alpha_log10 is not None else '—'
        imp_txt = f'{r.impedance_kohm:.1f} кОм' if r.impedance_kohm is not None else 'н/д'
        notes = '; '.join(r.reasons + r.physio_notes) or 'без замечаний'
        ch_rows.append(
            f"<tr><td><b>{html.escape(r.name)}</b></td>"
            f"<td style='color:{vc};font-weight:700'>{r.verdict}</td>"
            f"<td>{r.noise_brain_uv:.1f} мкВ</td><td>{imp_txt}</td>"
            f"<td>{r.drift_uv_per_s:+.1f} мкВ/с</td><td>{ratio_txt}</td>"
            f"<td>{html.escape(notes)}</td></tr>"
        )
    hash_html = ''
    if file_hashes:
        items = ''.join(f"<li><code>{html.escape(k)}</code>: <code>{html.escape(v)}</code></li>" for k, v in file_hashes.items())
        hash_html = f"<div class='box'><b>Контрольные суммы исходных файлов (SHA-256, без модификации):</b><ul>{items}</ul></div>"
    reasons_html = ('<ul>' + ''.join(f'<li>{html.escape(x)}</li>' for x in session['reasons']) + '</ul>') if session['reasons'] else '<p>Критических технических нарушений нет.</p>'
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Inter,-apple-system,Segoe UI,Arial,sans-serif;margin:28px auto;max-width:940px;color:#141923;line-height:1.45;background:#f8fafc;padding:0 18px}}
.card{{background:#fff;border:1px solid #dde3ea;border-radius:10px;padding:20px 24px;margin-bottom:16px}}
.hdr{{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}
.badge{{background:{badge_color};color:#fff;padding:6px 14px;border-radius:6px;font-weight:700;font-size:14px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.box{{background:#f1f5f9;border-radius:8px;padding:12px 16px;font-size:14px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:8px}}
th,td{{border-bottom:1px solid #e2e8f0;padding:8px 8px;text-align:left}}
th{{background:#f8fafc;color:#475569;font-weight:600}}
code{{background:#e2e8f0;padding:2px 5px;border-radius:4px;font-size:12.5px}}
.muted{{color:#64748b;font-size:12.5px}}
</style></head><body>
<div class="card">
  <div class="hdr">
    <div><h2 style="margin:0">{html.escape(title)}</h2>
    <div class="muted">EEG QC Passport v{html.escape(meta['rules'])} · профиль: {html.escape(DEVICE_PROFILE['name'])} · hash правил: <code>{meta['rules_hash']}</code></div></div>
    <div class="badge">{badge_label}</div>
  </div>
</div>
<div class="grid">
  <div class="card"><h3 style="margin-top:0">1. Техническое качество сигнала</h3>
    <p><b>Медианный RMS (1–40 Гц):</b> {session['rms_median_uv']:.2f} мкВ · <b>Годных каналов:</b> {len(session['usable'])} из {len(rows)} ({', '.join(session['usable']) or 'нет'})</p>
    <p><b>Исключено артефактных участков:</b> {meta['excluded_s']} с (морганий ЭОГ: {session['physio_artifacts']['blinks_count']})</p>
    {reasons_html}
  </div>
  <div class="card"><h3 style="margin-top:0">2. Физиологический результат пробы (ГЗ / ГО)</h3>
    <p><b>Заключение по пробе Бергера (8–13 Гц):</b><br>{html.escape(session['alpha']['text'])}</p>
    <p class="muted">Учебный отчёт: не является медицинским диагнозом или оценкой испытуемого.</p>
  </div>
</div>
<div class="card"><h3 style="margin-top:0">Детализация по каналам</h3>
  <table><thead><tr><th>Канал</th><th>Статус</th><th>RMS шум</th><th>Импеданс</th><th>Дрейф</th><th>Альфа ГЗ/ГО</th><th>Диагностика канала</th></tr></thead>
  <tbody>{''.join(ch_rows)}</tbody></table>
</div>
{hash_html}
</body></html>"""


if __name__ == '__main__':
    import json, subprocess, sys, os
    if not all(os.path.exists(f'{k}.npz') for k in ('S1', 'S2', 'S3')):
        subprocess.run([sys.executable, 'make_sessions.py'], check=True)
    out = {}
    for kind in ('S1', 'S2', 'S3'):
        z = np.load(f'{kind}.npz', allow_pickle=True)
        rows, session, meta = analyze(z['data'], float(z['fs']), [str(c) for c in z['ch']])
        out[kind] = dict(
            meta=meta,
            channels=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(r).items()} for r in rows],
            session=session,
        )
    with open('qc_result.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, v in out.items():
        s = v['session']
        print(f"{k}: {s['verdict']} | годных {len(s['usable'])} {s['usable']} | блок: {s['reasons'] or '—'}")
        print('   ', s['alpha']['text'])
