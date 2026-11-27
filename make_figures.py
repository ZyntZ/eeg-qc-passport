"""Генерация графиков в results/ по итогам расчётов PhysioNet и UCI."""
import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = pathlib.Path('results')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10.5,
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'axes.edgecolor': '#94a3b8',
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'grid.linewidth': 0.8,
})

NAVY = '#1e3a5f'
TEAL = '#0f766e'
AMBER = '#d97706'
RED = '#b91c1c'
SLATE = '#64748b'

s1 = json.load(open(OUT / 'study_summary.json', encoding='utf-8'))
cal = json.load(open(OUT / 'calibration.json', encoding='utf-8'))
B = pd.read_csv(OUT / 'study_results.csv')

# ---- Рис. 1: Три режима порогов на отложенной выборке (holdout n=55) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={'width_ratios': [1.3, 1.0]})
labels = ['Дефолты сухой\nгарнитуры', 'Фиксированный\nлаб. порог', 'Адаптивный\nпрофиль прибора']
spec = [cal['A_default']['specificity_pct'], cal['A_fixed_lab']['specificity_pct'], cal['B_calibrated']['specificity_pct']]
fpr = [cal['A_default']['false_rejection_pct'], cal['A_fixed_lab']['false_rejection_pct'], cal['B_calibrated']['false_rejection_pct']]
x = np.arange(len(labels))
w = 0.36
b1 = ax1.bar(x - w / 2, spec, w, color=TEAL, label='Специфичность (допуск чистых записей)')
b2 = ax1.bar(x + w / 2, fpr, w, color=RED, alpha=0.85, label='Ложная отбраковка чистых записей')
for bars in (b1, b2):
    for bar in bars:
        v = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f'{v:.1f}%', ha='center', fontsize=9.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 114)
ax1.set_ylabel('Доля записей, %')
ax1.set_title('Отложенная выборка (holdout n = 55 человек)')
ax1.legend(frameon=False, loc='upper left', fontsize=9)
ax1.grid(axis='x', visible=False)

defects = list(cal['detection_B']['by_defect_pct'].keys())
sens_vals = [cal['detection_B']['by_defect_pct'][d] for d in defects]
y = np.arange(len(defects))
ax2.barh(y, sens_vals, color=NAVY, height=0.58)
for yi, v in enumerate(sens_vals):
    ax2.text(v - 4, yi, f'{v:.0f}%', va='center', ha='right', color='white', fontweight='bold', fontsize=9.5)
ax2.set_yticks(y)
ax2.set_yticklabels(defects)
ax2.set_xlim(0, 110)
ax2.set_xlabel('Чувствительность детекции, %')
ax2.set_title('Ловля 6 типов брака после калибровки (90 тестов)')
ax2.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(OUT / 'fig1_calibration_holdout.png', bbox_inches='tight')
plt.close(fig)

# ---- Рис. 2: Альфа-реакция по каналам (109 человек, PhysioNet) ----
fig, ax = plt.subplots(figsize=(8.8, 4.4))
order = ['O1', 'O2', 'F8', 'F7', 'Fp2', 'Fp1']
data = [B[f'alpha_{c}'].dropna().values for c in order]
parts = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5,
                   medianprops=dict(color='#0f172a', lw=2))
for i, patch in enumerate(parts['boxes']):
    patch.set_facecolor(TEAL if order[i] in ('O1', 'O2') else '#cbd5e1')
    patch.set_alpha(0.88)
ax.axhline(0.25, color=RED, ls='--', lw=1.3, label='Порог выраженной реакции: ×1,78 (log10 = 0,25)')
ax.axhline(0.0, color=SLATE, ls=':', lw=1.0)
for i, c in enumerate(order, 1):
    sh = s1['alpha_by_channel'][c]['share_responsive']
    ratio = s1['alpha_by_channel'][c]['median_ratio']
    ax.text(i, 1.50, f'{sh:.0f}%\n(×{ratio:.2f})', ha='center', fontsize=9, fontweight='bold', color='#1e293b')
ax.set_ylim(-0.65, 1.78)
ax.set_ylabel('log10(мощность 8–13 Гц ГЗ / ГО)')
ax.set_title('Реакция альфа-ритма на закрытие глаз по отведениям (109 испытуемых)')
ax.legend(loc='lower right', frameon=False)
ax.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(OUT / 'fig2_alpha_by_channel.png', bbox_inches='tight')
plt.close(fig)

# ---- Рис. 3: Шумовой профиль и альфа-реакция у 109 человек (разделение техкачества и физиологии) ----
fig, ax = plt.subplots(figsize=(8.8, 4.6))
best_alpha = B[['alpha_O1', 'alpha_O2']].max(axis=1)
has_alpha = best_alpha >= 0.25
ax.scatter(B.loc[has_alpha, 'rms'], best_alpha[has_alpha],
           c=TEAL, alpha=0.82, s=42, label=f'Реакция обнаружена: {int(has_alpha.sum())} чел. (83,5%)')
ax.scatter(B.loc[~has_alpha, 'rms'], best_alpha[~has_alpha],
           c=AMBER, alpha=0.92, s=52, marker='D',
           label=f'Чистый сигнал, реакция не обнаружена: {int((~has_alpha).sum())} чел. (16,5%)')
lo, hi = s1['calibration']['profile']
ax.axvspan(lo, hi, color='#e0f2fe', alpha=0.35, zorder=0, label=f'Калиброванный профиль шума ({lo:.1f}–{hi:.0f} мкВ)')
ax.axhline(0.25, color=RED, ls='--', lw=1.2)
ax.set_xlabel('Медианный RMS сигнала 1–40 Гц по каналам, мкВ')
ax.set_ylabel('Максимум log10(ГЗ / ГО) в O1/O2')
ax.set_title('Техническая годность записи не равна наличию альфа-пика (109 испытуемых)')
ax.legend(frameon=False, loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'fig3_physio_vs_tech.png', bbox_inches='tight')
plt.close(fig)

# ---- Рис. 4: Клинический ЭЭГ против потребительской гарнитуры (O1/O2) ----
if (OUT / 'study_consumer_summary.json').exists():
    s2 = json.load(open(OUT / 'study_consumer_summary.json', encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    o_c = {r['channel']: r['median_ratio'] for r in s2['alpha_occipital']}
    clin = [s1['alpha_by_channel']['O1']['median_ratio'], s1['alpha_by_channel']['O2']['median_ratio']]
    cons = [o_c['O1'], o_c['O2']]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, clin, w, color=NAVY, label='Клинический ЭЭГ (BCI2000, 109 чел.)')
    b2 = ax.bar(x + w / 2, cons, w, color=AMBER, label='Потребительская гарнитура (Emotiv EPOC, 14 кан.)')
    ax.axhline(10 ** 0.25, color=RED, ls='--', lw=1.2, label='Порог реакции ×1,78')
    ax.axhline(1.0, color=SLATE, ls=':', lw=1.0)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f'×{v:.2f}', ha='center', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Отведение O1', 'Отведение O2'])
    ax.set_ylim(0, 4.3)
    ax.set_ylabel('Отношение мощности 8–13 Гц (ГЗ / ГО)')
    ax.set_title('Воспроизводимость пробы Бергера на разном оборудовании')
    ax.legend(frameon=False, loc='upper right', fontsize=9.2)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_clinical_vs_consumer.png', bbox_inches='tight')
    plt.close(fig)

# ---- Рис. 5: Кривые чувствительности к уровню дефекта (sweep) ----
sw = pd.read_csv(OUT / 'calibration_sweep.csv')
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), sharey=True)
meta_sw = [
    ('ступенька', 'Амплитуда скачка, мкВ', 'Ступенька контакта'),
    ('насыщение', 'Кратность усиления к порогу', 'Насыщение усилителя'),
    ('наводка', 'Амплитуда 50 Гц, мкВ', 'Сетевая наводка 50 Гц'),
]
for ax, (d, xl, ttl) in zip(axes, meta_sw):
    for cond, col, lbl, ls in [('A', SLATE, 'До калибровки', '--'), ('B', TEAL, 'После калибровки', '-')]:
        sub = sw[(sw['defect'] == d) & (sw['cond'] == cond)].groupby('level')['caught'].mean() * 100
        ax.plot(sub.index, sub.values, marker='o', lw=2, ls=ls, color=col, label=lbl)
    ax.set_xscale('log')
    ax.set_xlabel(xl)
    ax.set_title(ttl, fontsize=11)
    ax.set_ylim(-5, 108)
axes[0].set_ylabel('Доля обнаружения дефекта, %')
axes[2].legend(frameon=False, loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'fig5_sensitivity_curves.png', bbox_inches='tight')
plt.close(fig)

print('Графики обновлены:', sorted(p.name for p in OUT.glob('fig*.png')))
