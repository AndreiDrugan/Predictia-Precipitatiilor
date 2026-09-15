# Predicția Precipitațiilor în Australia (Weather Classification)

O soluție de clasificare bazată pe Machine Learning tabular pentru prognoza precipitațiilor în Australia pe baza datelor meteorologice zilnice. Pipeline-ul implementează două modele distincte de gradient boosting cu `CatBoostClassifier`: unul pentru prognozarea condițiilor din ziua curentă (`RainToday`) și unul pentru prognoza zilei următoare (`RainTomorrow`).

## Structura proiectului

- `main.ipynb` — Pipeline-ul de date: curățarea valorilor lipsă, feature engineering temporal, codificarea variabilelor de vânt și precipitații, antrenarea modelelor CatBoost și evaluarea acurateței.
- `.gitignore` — Ignoră setul de date brut (`Weather_Data.csv`).

## Abordare Tehnică

1. **Preprocesare & Feature Engineering:**
   - Extragerea lunii (`Month`) ca variabilă numerică din coloana `Date` pentru a capta sezonalitatea[cite: 9].
   - Eliminarea înregistrărilor incomplete via `dropna()`.
   - Codificarea direcțiilor vântului (`WindGustDir`, `WindDir9am`, `WindDir3pm`) și a etichetei binare `RainToday` folosind `LabelEncoder`[cite: 9].
   - Maparea țintelor binare (`Yes`/`No`) în valori `1` și `0`.
2. **Experimente & Arhitecturi de Modelare:**
   - **Model 1 (`RainToday`):** Antrenat pe un set restrâns de caracteristici (fără măsurătorile de la ora 9:00 și fără acumularea totală de precipitații), obținând o acuratețe de **~81.22%**.
   - **Model 2 (`RainTomorrow`):** Antrenat folosind `CatBoostClassifier` (2000 de iterații) pe setul extins de parametri atmosferici, obținând o acuratețe de **~82.75%** pe setul de test.

## Cum se rulează

1. Clonează repository-ul.
2. Adaugă fișierul `Weather_Data.csv` în directorul rădăcină.
3. Instalează dependențele:
   ```bash
   pip install pandas numpy scikit-learn catboost
   ```
