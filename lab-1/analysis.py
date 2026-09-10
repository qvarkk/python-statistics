import numpy as np
import pandas as pd
import seaborn as sns


def load_data() -> pd.Series:
    """Загружает датасет iris и возвращает непрерывную выборку petal_length."""
    df = sns.load_dataset("iris")
    print("Первые 5 строк набора данных:")
    print(df.head(), "\n")
    return df["petal_length"]


def describe_sample(sample: pd.Series) -> dict:
    """Считает основные числовые характеристики выборки."""
    x = sample.to_numpy()

    mean = np.mean(x)
    variance = np.var(x, ddof=1)
    std = np.std(x, ddof=1)
    median = np.median(x)

    return {
        "Объём выборки n": len(x),
        "Среднее": mean,
        "Дисперсия": variance,
        "Стандартное отклонение": std,
        "Медиана": median,
    }


def main():
    sample = load_data()
    stats = describe_sample(sample)

    print("Числовые характеристики выборки petal_length:")
    print("-" * 45)
    for name, value in stats.items():
        if isinstance(value, float):
            print(f"{name:<35}: {value:.4f}")
        else:
            print(f"{name:<35}: {value}")

    print("\nПроверка средствами pandas:")
    print(f"mean = {sample.mean():.4f}, "
          f"var = {sample.var(ddof=1):.4f}, "
          f"std = {sample.std(ddof=1):.4f}, "
          f"median = {sample.median():.4f}")


if __name__ == "__main__":
    main()