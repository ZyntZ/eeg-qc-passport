"""Автотесты движка EEG QC Passport v0.6.0."""
import copy
import hashlib
import pathlib
import numpy as np
import pytest
import qc_engine as Q
import make_sessions as MS


@pytest.fixture(autouse=True)
def restore_defaults():
    r_bak = copy.deepcopy(Q.RULESET)
    p_bak = copy.deepcopy(Q.DEVICE_PROFILE)
    yield
    Q.RULESET.clear()
    Q.RULESET.update(r_bak)
    Q.DEVICE_PROFILE.clear()
    Q.DEVICE_PROFILE.update(p_bak)


def test_clean_session_with_alpha_passes():
    x = MS.session('S1')
    rows, sess, meta = Q.analyze(x, MS.FS, MS.CH)
    assert not sess['block']
    assert 'O1' in sess['alpha']['responsive'] and 'O2' in sess['alpha']['responsive']
    assert 'обнаружена' in sess['alpha']['text']
    assert len(meta['rules_hash']) == 12


def test_clean_session_without_alpha_is_not_rejected():
    """Ключевое правило (Е. А. Жирмунская): чистый низкоамплитудный/десинхронизированный сигнал без альфа-пика НЕ бракуется."""
    x = MS.session('S3')
    rows, sess, _ = Q.analyze(x, MS.FS, MS.CH)
    assert sess['verdict'] == 'PASS'
    assert not sess['block']
    assert sess['alpha']['responsive'] == []
    assert 'реакция не обнаружена' in sess['alpha']['text']
    assert 'Жирмунской' in sess['alpha']['text']


def test_broken_session_blocks_physiological_conclusion():
    x = MS.session('S2')
    _, sess, _ = Q.analyze(x, MS.FS, MS.CH)
    assert sess['verdict'] == 'BLOCK'
    assert sess['block'] is True
    assert 'не проводится' in sess['alpha']['text']


def test_reference_crosstalk_vs_local_bridge():
    rng = np.random.default_rng(11)
    fs, n = 128.0, 128 * 30
    ch = ['Fp1', 'Fp2', 'T3', 'T4', 'O1', 'O2']
    base = rng.normal(0, 12.0, (6, n))
    # Локальный соляной мостик только между O1 и O2
    bridged = base.copy()
    bridged[4] = bridged[5] * 0.999 + rng.normal(0, 0.05, n)
    rows_b, sess_b, _ = Q.analyze(bridged, fs, ch, eyes_closed=(2, 14), eyes_open=(16, 28))
    assert not sess_b['ref_crosstalk']
    assert 'O2' in rows_b[4].bridged_with

    # Глобальная помеха референта во всех каналах
    common = rng.normal(0, 25.0, n)
    ref_bad = base * 0.1 + common[None, :]
    _, sess_r, _ = Q.analyze(ref_bad, fs, ch, eyes_closed=(2, 14), eyes_open=(16, 28))
    assert sess_r['ref_crosstalk'] is True
    assert sess_r['block'] is True


def test_impedance_and_emg_flags():
    rng = np.random.default_rng(21)
    fs, n = 160.0, 160 * 30
    t = np.arange(n) / fs
    ch = ['Fp1', 'Fp2', 'T3', 'T4', 'O1', 'O2']
    x = rng.normal(0, 10.0, (6, n))
    # Высокочастотный мышечный шум ЭМГ 36 Гц в T3
    x[2] += 22.0 * np.sin(2 * np.pi * 36.0 * t)
    rows, sess, _ = Q.analyze(x, fs, ch, eyes_closed=(2, 14), eyes_open=(16, 28),
                              impedances_kohm=[8.0, 9.0, 12.0, 28.0, 65.0, 7.0])
    assert rows[3].verdict == 'WARN'      # 28 кОм -> WARN
    assert rows[4].verdict == 'FAIL'      # 65 кОм -> FAIL
    assert 'T3' in sess['physio_artifacts']['emg_channels']


def test_live_window_guidance():
    rng = np.random.default_rng(5)
    fs = 128.0
    ch = ['Fp1', 'Fp2', 'T3', 'T4', 'O1', 'O2']
    w = rng.normal(0, 12.0, (6, int(3 * fs)))
    res_ok = Q.check_live_window(w, fs, ch)
    assert res_ok['ready_to_record'] is True
    w_bad = w.copy()
    w_bad[4] = 0.01
    w_bad[5] = 0.01
    res_bad = Q.check_live_window(w_bad, fs, ch)
    assert res_bad['ready_to_record'] is False
    assert any('O1' in h for h in res_bad['hints'])


def test_html_passport_and_immutability():
    edf = pathlib.Path('data_cache/S001R01.edf')
    if edf.exists():
        before = hashlib.sha256(edf.read_bytes()).hexdigest()
    x = MS.session('S1')
    rows, sess, meta = Q.analyze(x, MS.FS, MS.CH)
    doc = Q.render_html_passport('Тест паспорта', rows, sess, meta, {'S001R01.edf': 'abc123'})
    assert '<html' in doc and 'Техническое качество сигнала' in doc and 'abc123' in doc
    if edf.exists():
        after = hashlib.sha256(edf.read_bytes()).hexdigest()
        assert before == after
