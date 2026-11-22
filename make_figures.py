"""Figure generation for the two runs (see README)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.spines.top'] = False; rcParams['axes.spines.right'] = False
ACC='#1B5E8C'; ACC2='#D1495B'; ACC3='#4B8B3B'; GREY='#6C757D'
import os, sys
if not os.path.exists('results/study_summary.json'):
    sys.exit('results/study_summary.json не найден — сначала: bash fetch_data.sh && python3 run_study.py')
S = json.load(open('results/study_summary.json'))
B = pd.read_csv('results/study_results.csv')
A = pd.read_csv('results/study_all_conditions.csv')
A_only = A[A['condition'].str.startswith('A')]

# 1) alpha по каналам: распределение log10(closed/open), N = число строк в study_results.csv
fig, ax = plt.subplots(figsize=(9.4, 4.3), dpi=170)
chs = ['Fp1','Fp2','T3','T4','O1','O2']
data = [B[f'alpha_{c}'].values for c in chs]
bp = ax.boxplot(data, tick_labels=chs, patch_artist=True, widths=0.55,
                medianprops=dict(color='#fff', lw=1.6))
for patch, c in zip(bp['boxes'], ['#C6D8E6','#C6D8E6','#9DBFD6','#9DBFD6', ACC, ACC]):
    patch.set_facecolor(c); patch.set_edgecolor(c)
for i, d in enumerate(data, start=1):
    ax.scatter(np.random.default_rng(1).normal(i, 0.06, len(d)), d, s=14, color='#334', alpha=.65, zorder=3)
ax.axhline(0.25, color=ACC2, ls='--', lw=1)
ax.text(0.42, 0.30, 'порог «есть реакция»', fontsize=8, color=ACC2, ha='left',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.9))
ax.axhline(0, color='#999', lw=0.8)
ax.set_ylabel('log10(мощность 8–13 Гц: закрытые / открытые)')
ax.set_title(f'Alpha-реакция по каналам, N={len(B)}: log10 отношения мощностей закрытые/открытые',
                 fontsize=10.5, loc='left')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('results/fig1_alpha_by_channel.png'); plt.close()

# 2) доля отклика по каналам + 15% без реакции
fig, ax = plt.subplots(figsize=(9.4, 3.9), dpi=170)
shares = [S['alpha_by_channel'][c]['share_responsive'] for c in chs]
bars = ax.bar(chs, shares, color=['#C6D8E6','#C6D8E6','#9DBFD6','#9DBFD6', ACC, ACC], width=0.62)
for b, v in zip(bars, shares):
    ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%', ha='center', fontsize=9.5, fontweight='bold')
ax.axhline(S['share_subjects_occipital_response'], color=ACC3, ls='--', lw=1.2)
ax.text(2.0, S['share_subjects_occipital_response']+4.0,
        f"линия: {S['share_subjects_occipital_response']:.0f}% людей — реакция хотя бы в одном затылочном канале",
        fontsize=8.5, color=ACC3, ha='center',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.9))
ax.set_ylim(0, 100); ax.set_ylabel('доля людей с реакцией, %')
ax.set_title(f"Alpha-реакция при закрытии глаз: у {S['share_subjects_no_alpha']:.0f}% нет ни в одном из 6 каналов", fontsize=10.5, loc='left')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('results/fig2_response_share.png'); plt.close()

# 3) два условия: до/после калибровки
fig, ax = plt.subplots(figsize=(9.4, 3.8), dpi=170)
conds = ['A_без_калибровки','B_после_калибровки']
labels = ['A: наши пороги\nна чужом приборе', 'B: после калибровки\nпод профиль прибора']
passes = [S['conditions'][c]['pass_share'] for c in conds]
blocks = [S['conditions'][c]['block_share'] for c in conds]
x = np.arange(2); w = 0.34
b1 = ax.bar(x-w/2, passes, w, color=ACC3, label='не блокируется')
b2 = ax.bar(x+w/2, blocks, w, color=ACC2, label='BLOCK (заблокировано)')
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{b.get_height():.0f}%', ha='center', fontsize=9.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 108)
ax.set_ylabel('доля записей, %'); ax.legend(frameon=False, fontsize=9)
ax.set_title('Условие A (без калибровки) и условие B (пороги под прибор)', fontsize=10.5, loc='left')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('results/fig3_calibration.png'); plt.close()

# 4) дефекты, найденные автоматически (доля записей с WARN/ступеньками)
fig, ax = plt.subplots(figsize=(9.4, 3.8), dpi=170)
cats = ['сетевая наводка\n50/60 Гц', 'ступеньки контакта', 'моргания']
# флаг сетевой наводки — hum_any (1/0), а не колонки noise_*: там уровень шума, он есть у всех
hum = round(float((B['hum_any'].astype(float) > 0).mean())*100)
steps_share = round(float((B['steps_total'] > 0).mean())*100)
blink_share = round(float((B['n_blinks'] > 0).mean())*100)
vals = [hum, steps_share, blink_share]
bars = ax.bar(cats, vals, color=['#9DBFD6','#E09F3E', ACC], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 115); ax.set_ylabel('доля записей, %')
ax.set_title('Найденные дефекты по записям: моргания, ступеньки контакта, наводка 50 Гц', fontsize=10.5, loc='left')
ax.grid(alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig('results/fig4_found_defects.png'); plt.close()
print('figs ok')

# 5) два класса приборов: клинический vs потребительский
import os as _os
if _os.path.exists('results/study_consumer_summary.json'):
    KV = json.load(open('results/study_consumer_summary.json'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), dpi=170,
                                   gridspec_kw={'width_ratios': [1.15, 1]})
    # левая: alpha по каналам, два прибора
    clin = [a['median_ratio'] for a in S['alpha_by_channel'].values()]
    clin_n = list(S['alpha_by_channel'].keys())
    cons = [r['median_ratio'] for r in KV['alpha_by_channel']]
    cons_n = [r['channel'] for r in KV['alpha_by_channel']]
    ax1.bar(range(len(clin_n)), [a['median_ratio'] for a in S['alpha_by_channel'].values()],
            color=[ACC if c in ('O1','O2') else '#9DBFD6' for c in clin_n], width=0.65)
    ax1.set_xticks(range(len(clin_n))); ax1.set_xticklabels(clin_n)
    ax1.axhline(1.8, color=ACC2, ls='--', lw=1.1)
    ax1.text(len(clin_n)-0.4, 1.9, 'порог ×1,8', fontsize=8, color=ACC2, ha='right')
    ax1.set_ylabel('медиана: закрытые / открытые')
    ax1.set_title(f"Клинический ЭЭГ (N={S['design']['subjects']}): затылок ×3,5", fontsize=10, loc='left')
    ax1.set_ylim(0, 4.3); ax1.grid(alpha=0.2, axis='y')
    # правая: consumer
    ax2.bar(range(len(cons_n)), cons, color='#D1495B', width=0.65)
    ax2.set_xticks(range(len(cons_n))); ax2.set_xticklabels(cons_n, fontsize=8, rotation=45)
    ax2.axhline(1.8, color=ACC2, ls='--', lw=1.1)
    ax2.axhline(1.0, color='#999', lw=0.9)
    ax2.set_ylim(0, 4.3); ax2.set_title('Потребительская гарнитура (Emotiv): максимум ×1,17', fontsize=10, loc='left')
    ax2.set_ylabel('')
    ax2.grid(alpha=0.2, axis='y')
    ax2.text(0.5, 0.60, 'AUC 0,48–0,58\n= уровень случайности', transform=ax2.transAxes,
             ha='center', fontsize=10.5, color=ACC2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.45', fc='#F9ECEE', ec=ACC2))
    plt.tight_layout(); plt.savefig('results/fig5_two_device_classes.png'); plt.close()
    print('fig5 ok')

# 6) моргания против видеометки
if _os.path.exists('results/study_consumer_summary.json'):
    fig, ax = plt.subplots(figsize=(9.0, 3.7), dpi=170)
    b = KV['blinks']['channels']
    names = [x['channel'] for x in b]; shares = [x['share_with_eyes_open'] for x in b]
    events = [x['events'] for x in b]
    bars = ax.bar(names, shares, color=ACC3, width=0.55)
    for bar, sh, ev in zip(bars, shares, events):
        # при малом числе событий процент не подписываем как достижение — важна абсолютная пара
        lab = f'{sh:.0f}%' if ev >= 8 else f'{ev} из {ev}'
        ax.text(bar.get_x()+bar.get_width()/2, sh+2, lab, ha='center', fontsize=11, fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, sh/2, f'{ev} соб.', ha='center', fontsize=9, color='white', fontweight='bold')
    ax.axhline(50, color='#999', ls=':', lw=1.2)
    ax.text(0.5, 51.5, 'случайность, 50%', fontsize=8.5, color='#666', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none'))
    ax.set_ylim(0, 118); ax.set_ylabel('доля морганий при открытых глазах, %')
    ax.set_title('Моргания против видеометки по каналам (n = 35 событий); обучение не использовалось',
                 fontsize=10.5, loc='left')
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout(); plt.savefig('results/fig6_blinks_vs_groundtruth.png'); plt.close()
    print('fig6 ok')
