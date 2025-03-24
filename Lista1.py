import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from plot_likert import plot_likert
import plot_likert
import bisect
import scipy.stats as stats
from scipy.stats import binom
import numpy as np
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from joblib import Parallel, delayed
from scipy.stats import beta



#Zad1
df = pd.read_csv('ankieta.csv', encoding='Latin2', sep=";")


df.loc[df['WIEK'] <= 35, 'WIEK_KAT'] = '< 36'
df.loc[(df['WIEK'] >= 36) & (df['WIEK'] <= 45), 'WIEK_KAT'] = 'od 36 do 45'
df.loc[(df['WIEK'] >= 46) & (df['WIEK'] <= 55), 'WIEK_KAT'] = 'od 46 do 55'
df.loc[df['WIEK'] > 55, 'WIEK_KAT'] = '55 <'

#print(df)

#Zad3
table_dzial = df['DZIAŁ'].value_counts()
print(table_dzial)

table_staz = df['STAŻ'].value_counts()
print(table_staz)

table_czykier = df['CZY_KIER'].value_counts()
print(table_czykier)

table_plec = df['PŁEĆ'].value_counts()
print(table_plec)

table_wiekkat = df['WIEK_KAT'].value_counts()
print(table_wiekkat)

#zad 4

plt.figure(figsize=(8,8))

table_pyt1 = df['PYT_1'].value_counts()
table_pyt2 = df['PYT_2'].value_counts()

plt.subplot(1,2,1)
plt.pie(table_pyt1, labels=table_pyt1.index, autopct='%1.1f%%', startangle=90)
plt.title('Wykres kołowy dla kolumny "PYT_1"')

plt.subplot(1,2,2)
plt.pie(table_pyt2, labels=table_pyt2.index, autopct='%1.1f%%', startangle=90)
plt.title('Wykres kołowy dla kolumny "PYT_2"')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.bar(table_pyt1.index, table_pyt1)
plt.title('Wykres słupkowy dla kolumny "PYT_1"')

plt.subplot(1,2,2)
plt.bar(table_pyt2.index, table_pyt2)
plt.title('Wykres słupkowy dla kolumny "PYT_2"')

plt.tight_layout()
plt.show()

#zad5
print('5')

crosstab1 = pd.crosstab(df['PYT_1'],
                            df['DZIAŁ'],
                               margins = False)
print(crosstab1)

crosstab2 = pd.crosstab(df['PYT_1'],
                            df['STAŻ'],
                               margins = False)
print(crosstab2)

crosstab3 = pd.crosstab(df['PYT_1'],
                            df['CZY_KIER'],
                               margins = False)
print(crosstab3)

crosstab4 = pd.crosstab(df['PYT_1'],
                            df['PŁEĆ'],
                               margins = False)
print(crosstab4)

crosstab5 = pd.crosstab(df['PYT_1'],
                            df['WIEK_KAT'],
                               margins = False)
print(crosstab5)

print("6")
crosstab6 = pd.crosstab(df['PYT_2'],
                            df['PYT_3'],
                               margins = False)
print(crosstab6)

print("7")

df.loc[df['PYT_2'] <= -1, 'CZY_ZADOW'] = 'NIE'
df.loc[df['PYT_2'] >= 1, 'CZY_ZADOW'] = 'TAK'
print(df)


mosaic(df, ['CZY_ZADOW', 'DZIAŁ'])
plt.show()

mosaic(df, ['CZY_ZADOW', 'STAŻ'])
plt.show()

mosaic(df, ['CZY_ZADOW', 'CZY_KIER'])
plt.show()

mosaic(df, ['CZY_ZADOW', 'PŁEĆ'])
plt.show()

mosaic(df, ['CZY_ZADOW', 'WIEK_KAT'])
plt.show()

print("\n")

print("część II")
print("zad 2")
print(pd.DataFrame(table_pyt1).sort_values("PYT_1").transpose())


df_pivot = df.pivot_table(index="CZY_KIER", columns="PYT_1", aggfunc="size", fill_value=0)
print(df_pivot)

scale = ['-2', '-1', '0', '1', '2']
ax = plot_likert.plot_likert(df['PYT_1'].astype(str), scale, plot_percentage=True, figsize=(10,3),
                        bar_labels=True, bar_labels_color="snow", colors=plot_likert.colors.default_with_darker_neutral)
plt.title('Rozkład odpowiedzi na PYT_1 w podziale na CZY_KIER')

ax.xaxis.set_label_text('% odpowiedzi')
plt.show()

df_czy_kier_tak = df[df['CZY_KIER'] == "Nie"]

ax = plot_likert.plot_likert(df_czy_kier_tak['PYT_1'].astype(str), scale, plot_percentage=True, figsize=(10,3),
                        bar_labels=True, bar_labels_color="snow", colors=plot_likert.colors.default_with_darker_neutral)
plt.title('Rozkład odpowiedzi na PYT_1 dla CZY_KIER = Nie')

ax.xaxis.set_label_text('% odpowiedzi')
plt.show()

df_czy_kier_tak = df[df['CZY_KIER'] == "Tak"]

ax = plot_likert.plot_likert(df_czy_kier_tak['PYT_1'].astype(str), scale, plot_percentage=True, figsize=(10,3),
                        bar_labels=True, bar_labels_color="snow", colors=plot_likert.colors.default_with_darker_neutral)
plt.title('Rozkład odpowiedzi na PYT_1 dla CZY_KIER = Tak')

ax.xaxis.set_label_text('% odpowiedzi')
plt.show()

print(" ")
print("zad3")
import numpy as np

sample_size = int(len(df) * 0.1)

indices_without_replacement = np.random.choice(df.index, size=sample_size, replace=False)
sample_without_replacement = df.loc[indices_without_replacement]

indices_with_replacement = np.random.choice(df.index, size=sample_size, replace=True)
sample_with_replacement = df.loc[indices_with_replacement]

print(f"Liczba wierszy w oryginalnym zbiorze: {len(df)}. Unikalnych: {len(df.drop_duplicates())}")
print("")
print(f"Liczba wierszy w próbie bez zwracania: {len(sample_without_replacement)}")
print(f"Liczba unikalnych wierszy w próbie bez zwracania: {len(sample_without_replacement.drop_duplicates())}")
print("")
print(f"Liczba wierszy w próbie ze zwracaniem: {len(sample_with_replacement)}")
print(f"Liczba unikalnych wierszy w próbie ze zwracaniem: {len(sample_with_replacement.drop_duplicates())}")


print(" ")

print("\n zad 4")


def function_1(n, p):
    random_numbers = np.random.uniform(0, 1, n)
    results = ["0" if x < p else "1" for x in random_numbers]
    count_ones = results.count("1")

    return count_ones


n = 100
p = 0.5

print("finkcja1: ",function_1(n, p))


def function_2(n, p, N):
    # Tworzymy wektor długości N wypełniony zerami
    success_counts = np.zeros(N, dtype=int)

    # Wykonujemy function_1 N razy
    for i in range(N):
        success_counts[i] = function_1(n, p)

    return success_counts


n = 1000
p = 0.5
N = 40

print("funkcja2: ",function_2(n, p, N))

W = function_2(n, p, N)
x = sum(W)

E_theoretical = n * p
Var_theoretical = n * p * (1 - p)

E_empirical = np.mean(W)
Var_empirical = np.var(W)

print(f"Teoretyczna wartość oczekiwana (E): {E_theoretical}")
print(f"Empiryczna wartość oczekiwana (E): {E_empirical}")

print(f"Teoretyczna wariancja (Var): {Var_theoretical}")
print(f"Empiryczna wariancja (Var): {Var_empirical}", "\n")

print("zad 5")


def func1(n, ps, x):
    random_numbers = np.random.uniform(0, 1, n)
    cum_prob = np.cumsum(ps)
    result = np.zeros(len(x))
    for num in random_numbers:
        idx = bisect.bisect(cum_prob, num)
        result[idx] += 1

    return result


n = 100
p = [0.3, 0.3, 0.4]
x = [0, 1, 2]

print("funkcja2: ", func1(n, p, x), "\n")


def func2(n, p, N, x):
    success_counts = np.zeros((N, len(x)))

    for i in range(N):
        success_counts[i] = func1(n, p, x)

    return success_counts


n = 100
p = [0.3, 0.3, 0.4]
x = [0, 1, 2]
N = 1000

print("finkcja2: ", func2(n, p, N, x))


sample = func2(n,p,N,x)
E_teo_multinom = [n * p_ for p_ in p]
E_emp_multinom = np.mean(sample, axis=0)

print(f"Teoretyczna wartość oczekiwana (E): {E_teo_multinom}")
print(f"Empiryczna wartość oczekiwana (E): {E_emp_multinom}", "\n")


print("zad 6")
def clopper_pearson_ci(confidence, successes=None, trials=None, data=None):
    """
    Oblicza przedział ufności Cloppera-Pearsona dla proporcji sukcesów.

    Parametry:
    - confidence: Poziom ufności (np. 0.95 dla 95%)
    - successes: Liczba sukcesów (opcjonalnie, jeśli podano trials)
    - trials: Liczba prób (opcjonalnie, jeśli podano successes)
    - data: Wektor wartości binarnych (opcjonalnie zamiast successes i trials)

    Zwraca:
    - Dolna i górna granica przedziału ufności.
    """
    if data is not None:
        successes = np.sum(data)
        trials = len(data)

    if successes is None or trials is None:
        raise ValueError("Należy podać albo (successes, trials), albo wektor data.")

    alpha = 1 - confidence

    lower_bound = stats.beta.ppf(alpha / 2, successes, trials - successes + 1) if successes > 0 else 0.0
    upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes) if successes < trials else 1.0

    return lower_bound, upper_bound


# Przykłady użycia:
print(clopper_pearson_ci(0.95, successes=10, trials=100))  # (0.048, 0.183)
data = np.random.binomial(1, 0.1, 100)
print(clopper_pearson_ci(0.95, data=data))


print("zad 7 \n")


print("zad 8 \n")

wynik_1 = np.random.binomial(n=10, p=0.5)
print(f"Wynik pojedynczej symulacji: {wynik_1}")

wyniki_1000 = np.random.binomial(n=10, p=0.5, size=1000)

print(f"Średnia liczba sukcesów w 1000 eksperymentach: {np.mean(wyniki_1000)}")