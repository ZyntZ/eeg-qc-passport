"""Генерация HTML-паспортов на реальных записях PhysioNet EEGMMIDB (S001 и S005)
и демонстрация онлайн-режима постановки электродов (check_live_window).

Исходные EDF-файлы читаются только на чтение; их SHA-256 фиксируется в паспорте.
"""
import hashlib
import json
import pathlib
import mne
import numpy as np
import qc_engine as Q

mne.set_log_level('ERROR')
DATA = pathlib.Path('data_cache')
OUT = pathlib.Path('results')
OUT.mkdir(exist_ok=True)

MAP = [('Fp1.', 'Fp1'), ('Fp2.', 'Fp2'), ('F7..', 'F7'), ('F8..', 'F8'), ('O1..', 'O1'), ('O2..', 'O2')]
NAMES = [a for a, _ in MAP]
SHORT = [b for _, b in MAP]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_concat_ec_eo(sub: str):
    """Склеивает R02 (глаза закрыты, 0–45 с) и R01 (глаза открыты, 45–90 с) для единого паспорта пробы."""
    f_eo = DATA / f'{sub}R01.edf'
    f_ec = DATA / f'{sub}R02.edf'
    hashes = {f_ec.name: sha256_file(f_ec)[:24] + '…', f_eo.name: sha256_file(f_eo)[:24] + '…'}
    r_eo = mne.io.read_raw_edf(f_eo, preload=True, verbose=False).pick(NAMES)
    r_ec = mne.io.read_raw_edf(f_ec, preload=True, verbose=False).pick(NAMES)
    fs = float(r_ec.info['sfreq'])
    n45 = int(45 * fs)
    x = np.concatenate([r_ec.get_data()[:, :n45] * 1e6, r_eo.get_data()[:, :n45] * 1e6], axis=1)
    return x, fs, hashes


def main():
    cal = json.loads((OUT / 'calibration.json').read_text(encoding='utf-8'))
    Q.RULESET.update(cal['thresholds_derived'])
    Q.DEVICE_PROFILE['name'] = 'BCI2000 64-ch (профиль откалиброван на n=54)'
    Q.DEVICE_PROFILE['rms_brain_uv'] = tuple(cal['profile_rms_bounds_uv'])

    # 1. S001 — выраженная альфа-реакция ГЗ/ГО
    x1, fs, h1 = load_concat_ec_eo('S001')
    rows1, sess1, meta1 = Q.analyze(x1, fs, SHORT, eyes_closed=(2, 43), eyes_open=(47, 88),
                                    impedances_kohm=[6.2, 7.1, 8.5, 9.0, 5.4, 5.8])
    (OUT / 'passport_S001_alpha.html').write_text(
        Q.render_html_passport('Паспорт записи ЭЭГ — испытуемый S001 (PhysioNet EEGMMIDB)', rows1, sess1, meta1, h1),
        encoding='utf-8',
    )

    # 2. S005 — чистый сигнал без альфа-ответа (низкоамплитудный / десинхронизированный вариант нормы)
    x5, fs, h5 = load_concat_ec_eo('S005')
    rows5, sess5, meta5 = Q.analyze(x5, fs, SHORT, eyes_closed=(2, 43), eyes_open=(47, 88),
                                    impedances_kohm=[7.0, 6.8, 9.2, 8.4, 6.1, 6.5])
    (OUT / 'passport_S005_desync.html').write_text(
        Q.render_html_passport('Паспорт записи ЭЭГ — испытуемый S005 (чистый сигнал без альфа-реакции)', rows5, sess5, meta5, h5),
        encoding='utf-8',
    )

    # 3. Демонстрация брака на этапе постановки (обрыв O1 + высокий импеданс O2 + наводка 50 Гц)
    xb = x1.copy()
    t = np.arange(xb.shape[1]) / fs
    xb[4] = 0.05 * np.sin(2 * np.pi * 2 * t)          # O1 оторван
    xb[5] += 180.0 * np.sin(2 * np.pi * 50 * t)       # O2 сильная наводка + плохой контакт
    rows_b, sess_b, meta_b = Q.analyze(xb, fs, SHORT, eyes_closed=(2, 43), eyes_open=(47, 88),
                                       impedances_kohm=[7.0, 7.5, 9.0, 8.8, 68.0, 54.0])
    (OUT / 'passport_S001_broken_setup.html').write_text(
        Q.render_html_passport('Паспорт записи ЭЭГ — пример технического брака постановки (O1/O2)', rows_b, sess_b, meta_b),
        encoding='utf-8',
    )

    # 4. Онлайн-окно 3 с до старта записи (рекомендации студенту в реальном времени)
    live_bad = Q.check_live_window(xb[:, int(52 * fs):int(55 * fs)], fs, SHORT, impedances_kohm=[7.0, 7.5, 9.0, 8.8, 68.0, 54.0])
    live_good = Q.check_live_window(x1[:, int(52 * fs):int(55 * fs)], fs, SHORT, impedances_kohm=[6.2, 7.1, 8.5, 9.0, 5.4, 5.8])
    (OUT / 'live_preflight_demo.json').write_text(
        json.dumps({'before_fix': live_bad, 'after_fix': live_good}, ensure_ascii=False, indent=1),
        encoding='utf-8',
    )
    print('Паспорта сохранены. Онлайн-режим до поправки электродов:', live_bad['status'], live_bad['hints'])
    print('После поправки электродов:', live_good['status'], live_good['hints'])


if __name__ == '__main__':
    main()
