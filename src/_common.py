"""Мелкие общие вещи для скриптов оценки."""
import pandas as pd


def load_features(path="data/features_spectral.csv"):
    """Читает таблицу признаков и заполняет пропуски медианой по условию.

    Пропуск бывает только у peak_alpha: если локального максимума в альфе нет,
    04 пишет NaN и ставит peak_alpha_found=False. LogisticRegression
    NaN не принимает, поэтому заполняем, а флаг остаётся в таблице.
    """
    df = pd.read_csv(path)
    num = df.select_dtypes("number").columns
    for c in num:
        if df[c].isna().any():
            df[c] = df.groupby("condition")[c].transform(lambda s: s.fillna(s.median()))
    return df
