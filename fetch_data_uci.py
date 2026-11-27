"""Прямая загрузка открытого датасета UCI EEG Eye State (CC0, без токенов API)."""
import io, os, urllib.request, zipfile
import pandas as pd

os.makedirs('kaggle_data', exist_ok=True)
URL = 'https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip'
COLS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'eyeDetection']

req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=45).read()))
raw = zf.read(zf.namelist()[0]).decode('utf-8', errors='replace')
rows, in_data = [], False
for line in raw.splitlines():
    s = line.strip()
    if not s or s.startswith('%'):
        continue
    if s.lower().startswith('@data'):
        in_data = True
        continue
    if in_data:
        parts = s.split(',')
        if len(parts) == 15:
            rows.append([float(x) if i < 14 else int(float(x)) for i, x in enumerate(parts)])
df = pd.DataFrame(rows, columns=COLS)
df.to_csv('kaggle_data/EEG_Eye_State_Classification.csv', index=False)
print(f'Сохранено {len(df)} эпох в kaggle_data/EEG_Eye_State_Classification.csv')
