"""Второй (независимый) датасет через Kaggle API: UCI EEG Eye State (Emotiv, 14 каналов, 128 Гц).

Зачем: PhysioNet — клинический 64-канальный прибор. Нужен второй класс приборов — потребительская
гарнитура — чтобы проверить перенос метода и границу применимости.

Лицензия: CC0-1.0. Источник: https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset

Требуется токен: ~/.kaggle/access_token (Kaggle → Settings → API → Create New Token)
Запуск: python3 fetch_data_kaggle.py
"""
import pathlib, subprocess, sys, zipfile

DST = pathlib.Path('kaggle_data'); DST.mkdir(exist_ok=True)
SLUG = 'robikscube/eye-state-classification-eeg-dataset'
TARGET = DST / 'EEG_Eye_State_Classification.csv'

if TARGET.exists():
    print(f'уже есть: {TARGET} ({TARGET.stat().st_size} байт)')
    sys.exit(0)

if not (pathlib.Path.home() / '.kaggle' / 'access_token').exists() and not (pathlib.Path.home() / '.kaggle' / 'kaggle.json').exists():
    sys.exit('Нет токена Kaggle. Положите ~/.kaggle/access_token или ~/.kaggle/kaggle.json')

print(f'downloading {SLUG}')
subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'download',
                '-d', SLUG, '-p', str(DST), '--unzip'], check=True)
if not TARGET.exists():
    for z in DST.glob('*.zip'):
        with zipfile.ZipFile(z) as zf: zf.extractall(DST)
print('готово:', sorted(p.name for p in DST.iterdir()))
