# Sequential Data Analytics and Modeling (AID01)

**Course companion repository | Undergraduate Artificial Intelligence Program | Universitas Bunda Mulia**

This repository supports **Sequential Data Analytics and Modeling (AID01)**, a underundergraduate course on understanding, preparing, modeling, forecasting, and evaluating data whose observations have a meaningful order in time. It combines statistical methods, Python programming, and deep learning, with practical emphasis on reproducible experiments in **Google Colab**.

Students progress from identifying sequential patterns to building traditional forecasting models and neural networks such as **RNN, LSTM, and GRU**. Examples are intended to be small enough to follow during class and sufficiently rigorous to demonstrate proper training, validation, testing, and error analysis.

> **Syllabus alignment:** This README is based on the uploaded AID01 outcome-based education (OBE) course plan for the 2026 odd semester. Its official course description also references reinforcement learning and stochastic optimization; however, the detailed session-by-session plan concentrates on sequential-data analytics and forecasting. Those additional topics should not be assumed to have dedicated labs in this repository unless new materials are explicitly added.

## 1. Course information

| Item | Details |
|---|---|
| Official course name | **Sequential Data Analytics and Modeling** |
| Course code | **AID01** |
| Institution | Universitas Bunda Mulia (UBM) |
| Faculty | Faculty of Technology and Design |
| Program | Undergraduate Artificial Intelligence Program |
| Semester | 2 |
| Credit allocation | 4 credits (**2 theory / 2 practice**, per syllabus) |
| Syllabus period | 2026 odd semester, regular classes |
| Primary environments | Python and Google Colab |
| Relevant libraries | pandas, NumPy, Matplotlib, scikit-learn, TensorFlow/Keras; statsmodels for classical forecasting exercises |

### Why sequential data matters

A collection of independent property listings is different from a collection of weather measurements sampled every ten minutes. In property data, the order of rows generally has no inherent meaning; in weather data, *when* a measurement occurred affects both what we can predict and how to evaluate it.

Sequential-data problems arise in industrial sensors, smart buildings, demand forecasting, transportation, energy, financial markets, and operational monitoring. They require students to reason about **trend, seasonality, lag, autocorrelation, nonstationarity, forecasting horizons, and temporal data leakage**, rather than treating observations as interchangeable records.

## 2. Learning outcomes

The following are English summaries of the three **course learning outcomes (CPMK)** specified in the AID01 syllabus.

1. **CPMK1 — Understand sequential data.** Explain sequential observations, time series, and the characteristics that distinguish them from non-sequential data.
2. **CPMK2 — Frame and prepare sequential-data problems.** Select appropriate analytical methods; clean, visualize, smooth, difference, and engineer features from ordered observations.
3. **CPMK3 — Develop and evaluate forecasting models.** Implement classical methods—including moving averages, ARIMA, and exponential smoothing—and deep-learning methods such as RNN, LSTM, and GRU; assess their predictive performance.

The course contributes to program outcomes involving the identification and implementation of industry-relevant AI solutions (**CPL05**) and AI model development, optimization, and applications including IoT (**CPL08**).

## 3. Learning pathway

```text
Independent observations                  Sequential observations
(house attributes)                       (time-ordered weather measurements)
       |                                             |
       +-- Classification: category                  +-- Explore time series
       |      (cheap / medium / expensive)            |       (trend, seasonality, lag)
       +-- Regression: continuous value               +-- Prepare temporal features
              (house price)                           |       (windows, scaling, differencing)
                                                     +-- Classical forecasting
                                                     |       (MA / smoothing / ARIMA)
                                                     +-- Deep sequential models
                                                             (RNN / LSTM / GRU)
                                                     +-- Chronological evaluation
                                                             (MAE / RMSE / plots)
```

**Important:** The house-price exercises are **introductory machine-learning bridges**, not sequential-data forecasting tasks. They are included to help students understand neural-network outputs and error metrics before moving to time series and LSTM.

## 4. Practical notebooks and datasets

This repository is an **evolving collection of course materials**. New TensorFlow, statistical forecasting, and sequential-modeling practicals may be added throughout the semester. The examples listed below illustrate the type of work students can expect; they are **not an exhaustive or fixed list** of course practicals. Follow-along notebooks provide explanations before code cells and Indonesian-language teaching comments, while repository documentation is in English.

| Practical | Notebook | Input | Target | Core concept |
|---|---|---|---|---|
| House classification | `Praktikum_Klasifikasi_Harga_Rumah_TensorFlow.ipynb` | Four house attributes | `Murah`, `Sedang`, `Mahal` | Multiclass classification with a DNN |
| House-price regression | `Praktikum_Regresi_Harga_Rumah_TensorFlow.ipynb` | Same four attributes | Price in Indonesian rupiah | Continuous-value regression with a DNN |
| Jakarta temperature forecasting | `Praktikum_LSTM_Prediksi_Suhu_Jakarta_TensorFlow.ipynb` | Recent weather observations | Future air temperature (°C) | Sequence-to-one LSTM forecasting |

### Example: Multiclass house classification

**Goal:** Predict one of three house-price categories from:

- `Luas_Rumah` — house area;
- `Jarak_Ke_Pusat_Kota` — distance from the city center;
- `Jumlah_Kamar` — number of rooms;
- `Usia_Rumah` — age of the house.

**Dataset:** `dataset_kategori_rumah_30000.csv` — **30,000 synthetic records**. The target is `Kategori_Harga`.

**Architecture:** Four input features, dense hidden layers, and three softmax output units. Students inspect learning curves, classification accuracy, a confusion matrix, and individual predictions.

**Discussion:** Why is class accuracy useful here, and what does a confusion matrix reveal that a single accuracy number cannot?

### Example: House-price regression

**Goal:** Predict a numerical property price using the same four features as the classification example.

**Dataset:** `dataset_regresi_harga_rumah_30000.csv` — **30,000 synthetic records**, including the `Harga_Rumah_Rp` target.

**Architecture:** A small feedforward DNN with a **single linear output**, rather than a three-class softmax. The notebook converts the target to billions of rupiah for more convenient numerical training and converts results back for interpretation.

**Metrics:** Mean absolute error (**MAE**), root mean squared error (**RMSE**), and coefficient of determination (**R²**). Students inspect loss curves, actual-versus-predicted prices, and prediction errors.

**Discussion:** Why can the same four inputs support both classification and regression while requiring different output layers, losses, and evaluation metrics?

### Example: Jakarta temperature forecasting with LSTM

**Goal:** Forecast Jakarta's air temperature **one hour ahead** using the preceding **six hours** of measurements sampled every **ten minutes**.

**Dataset:** `Dataset_Cuaca_Jakarta_Sintetis_2024_10Menit.csv` — **52,704 time-stamped synthetic weather observations** covering leap year 2024. The dataset includes weather variables such as temperature (`T (degC)`), humidity, pressure, and wind measurements; the notebook uses a compact subset of predictors.

**Sequence design:** 36 previous observations are converted into a window of input features; the target is the temperature six ten-minute intervals ahead.

**Architecture:** `LSTM(16)` followed by `Dense(1)` in TensorFlow/Keras. Students learn how a 3D sequence tensor differs from an ordinary tabular feature matrix.

**Evaluation:** Training/validation learning curves; test-set MAE in **°C**; a baseline based on the latest observed temperature; actual-versus-forecast temperature plots; and error plots.

**Discussion:** Why must the data be split chronologically? Does a low MAE necessarily mean the model captures sudden temperature changes?

> **Data provenance:** The datasets described above are **synthetic teaching data**, not verified property transactions or official instrument measurements. The Jakarta data are designed to exhibit plausible warm tropical conditions but must **not** be presented as historical BMKG observations, used for meteorological claims, or interpreted as measured Jakarta temperatures.

## 5. Recommended repository layout

The following is an **illustrative, extensible layout**, not a claim that every directory or notebook already exists. Additional notebooks, datasets, and exercises can be added without changing the structure.

```text
AID01-sequential-data-analytics/
├── README.md
├── notebooks/
│   ├── Praktikum_Klasifikasi_Harga_Rumah_TensorFlow.ipynb
│   ├── Praktikum_Regresi_Harga_Rumah_TensorFlow.ipynb
│   └── Praktikum_LSTM_Prediksi_Suhu_Jakarta_TensorFlow.ipynb
├── datasets/
│   ├── dataset_kategori_rumah_30000.csv
│   ├── dataset_regresi_harga_rumah_30000.csv
│   └── Dataset_Cuaca_Jakarta_Sintetis_2024_10Menit.csv
├── slides/                      # Optional course slides
├── exercises/                   # Optional student assignments
└── figures/                     # Optional lecture illustrations
```

If you publish the notebooks under `notebooks/`, update any notebook-relative links accordingly. The example notebooks read CSVs from Colab's `/content/sample_data/` directory rather than automatically resolving the `datasets/` directory in this repository.

## 6. Run the notebooks in Google Colab

**No local installation is required** for the follow-along format. A standard CPU runtime is sufficient for these small teaching models; GPU access is optional.

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook**, and select one `.ipynb` file; alternatively, open a notebook directly from GitHub after publishing the repository.
3. In Colab, open the **Files** sidebar and locate `/content/sample_data/`.
4. Upload that practical's **matching CSV** into `/content/sample_data/`.
5. Run each cell from top to bottom (**Shift + Enter**), reading its explanation before executing the code.
6. Inspect the plots and performance metrics, then change **one hyperparameter at a time** for the experiments.

For example, the LSTM notebook expects:

```python
PATH = "/content/sample_data/Dataset_Cuaca_Jakarta_Sintetis_2024_10Menit.csv"
```

If you store the CSV in a different location, change `PATH` (or `DATA_PATH`, depending on the notebook) before running the data-loading cell.

**Colab note:** Files uploaded to a runtime are generally temporary. After a runtime reset, you may need to upload the CSV again and rerun the cells. Package versions may change in managed Colab runtimes; check the actual installed TensorFlow version if results differ across semesters.

### Optional: Work from a cloned repository

You may also use the terminal in a local environment and install the typical dependencies:

```bash
python -m pip install tensorflow pandas numpy matplotlib scikit-learn
```

Classical forecasting exercises may additionally require:

```bash
python -m pip install statsmodels
```

These are illustrative installation commands, not a validated or pinned environment specification. For reproducible research, record package versions and the Python runtime used for each experiment.

## 7. How the evaluation methods differ

| Aspect | Classification | Regression | Time-series forecasting |
|---|---|---|---|
| Example | House category | House price | Jakarta temperature at a future time |
| Prediction | Discrete label | Continuous value | Future value indexed by time |
| Typical neural-network output | Softmax | Linear | Linear prediction from a sequence model |
| Useful metrics | Accuracy, precision/recall, confusion matrix | MAE, RMSE, R² | MAE, RMSE, plus horizon-based and temporal error analysis |
| Splitting approach | Stratified/random split when appropriate | Random split for independent records | **Chronological** split; avoid future-to-past leakage |
| Essential plot | Confusion matrix | Actual vs. predicted | Actual vs. forecast **over time** |

**MAE is not unique to ordinary regression.** It is also a standard metric for time-series forecasting:

\[
\operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|.
\]

Its interpretation depends on the target: **rupiah** for house-price forecasts and **degrees Celsius** for temperature forecasts. Smaller MAE means smaller average absolute prediction error, but it should be considered alongside a suitable baseline and the task's operational requirements.

### A fair forecasting evaluation

For time-ordered data, a sound evaluation must:

- Preserve **time order** between training, validation, and test intervals.
- Fit preprocessing parameters **only on training data**.
- Ensure input windows contain only information available **before** the forecasted target.
- Report the **forecast horizon** (for example, one hour ahead).
- Compare with a sensible baseline, such as the **last-observed-value forecast**.
- Inspect performance during peaks, abrupt changes, or unusually difficult periods, not just the average error.

In the supplied house exercises, records are designed as independent observations, so random train/validation/test splitting is appropriate. In the weather exercise, random shuffling across time would risk misleading evaluation.

## 8. OBE teaching roadmap

The table below follows the **session-group topics in the uploaded AID01 syllabus**. The sessions described here are curricular topics; practical materials may be added or revised independently as the course progresses.

| Sessions | Syllabus topic | Main competencies |
|---|---|---|
| 1–2 | Foundations of sequential data | Sequential versus cross-sectional data, stochastic processes, time-series components, application domains, Python ecosystem |
| 3–4 | Statistical properties | Stationarity, ACF/PACF, white noise, random walk, unit-root tests, lags and leads |
| 5–6 | Problem identification and taxonomy | State variables, prediction versus decision, sequential problem framing, industrial problem mapping |
| 7–8 | Method selection | Classical statistics versus ML, interpretability, computational trade-offs, data requirements, sparse observations |
| 9–10 | Preprocessing and smoothing | Missing observations, SMA, weighted/exponential moving averages, anomalies, resampling |
| 11–12 | Transformation and feature engineering | Differencing, Box–Cox/log transforms, windowing, statistics-based features, Fourier analysis, scaling |
| 13–14 | Review | Integration and consolidation of preceding concepts |
| Midterm | UTS | Project proposal integrating course concepts |
| 15–16 | Classical forecasting: moving averages and smoothing | Exponential smoothing, parameter tuning, MLE, residual analysis, Python implementation |
| 17–18 | ARIMA family | AR, MA, ARMA, ARIMA, SARIMA, Box–Jenkins approach, AIC/BIC |
| 19–20 | Classical model evaluation | In-sample/out-of-sample forecasting, walk-forward validation, MAE/RMSE/MAPE, uncertainty intervals |
| 21–22 | Deep learning: RNN and sequential foundations | RNN architecture, vanishing/exploding gradients, many-to-one/many-to-many, BPTT, hyperparameters |
| 23–24 | Advanced deep learning: LSTM and GRU | Gating, GRU, bidirectional/stacked models, regularization, IoT-oriented model optimization |
| 25–26 | Implementation and evaluation of deep learning | Scaling, loss functions, internal-state inspection, industrial evaluation, real-time pipelines, end-to-end projects |
| 27–28 | Review | Final integration and synthesis |
| Final | UAS | Final course assessment |

The LSTM weather practical directly supports the neural-network topics in sessions 21–26. The two house examples help introduce supervised-learning outputs and evaluation, but are not substitutes for the syllabus's classical time-series material.

## 9. Assessment scheme

The official syllabus specifies the following **course-level weights**:

| Component | Weight |
|---|---:|
| Practice | 30% |
| Midterm examination (UTS) | 30% |
| Final examination (UAS) | 40% |
| **Total** | **100%** |

The session plan describes the midterm as an integrated **project proposal**. Assessment rubrics, deadlines, submission formats, and final-project requirements should be provided by the instructor; this README does not introduce additional official grading rules.

## 10. Suggested follow-along experiments

Use one change per experiment. Re-run the relevant training and evaluation cells and record a comparable test metric.

**House classification**

1. Increase the number of hidden units (for example, 32 to 64). Does test accuracy improve?
2. Change the learning rate (`0.001` versus `0.01`). What happens to the loss curve?
3. Train with fewer epochs. Does the model underfit?
4. Change the batch size (64 versus 128). What changes in training behavior?
5. Remove a hidden layer. Does the simpler model perform similarly?

**House-price regression**

1. Compare 32 versus 64 hidden units using test MAE.
2. Try two learning rates and inspect convergence.
3. Compare 15 and 40 maximum epochs, considering early stopping.
4. Change the batch size and note differences in training time and test MAE.
5. Remove one hidden layer and compare test MAE, RMSE, and R².

**Jakarta LSTM forecasting**

1. Compare 12 versus 36 past observations (two versus six hours).
2. Compare 8, 16, and 32 LSTM units.
3. Compare learning rates (`0.0001`, `0.001`, `0.01`).
4. Compare batch sizes of 32 and 128.
5. Change the forecast horizon from one hour to three hours, ensuring that window targets and evaluation timestamps are updated consistently.

**Experiment-reporting template:** Record the configuration, random seed, epochs actually completed, training and validation curves, test metric, baseline metric (where applicable), and one short explanation of the observed result. Do not assume that a larger model or a longer training run must perform better.

## 11. Suggested integrative project (non-mandatory example)

A suitable project is **short-horizon temperature forecasting for an urban monitoring system**. Students can extend the LSTM notebook by comparing a persistence baseline, a moving-average baseline, a classical forecasting model, and an LSTM model.

A complete report could cover the problem definition, source and quality of the data, timestamp checks, exploratory plots, preprocessing, feature windows, chronological split, hyperparameter choices, training curves, MAE/RMSE, failure cases, and deployment considerations.

This is **a suggested project example**, not an officially prescribed final-project topic. An authentic applied study should ideally use a properly licensed, real observational dataset rather than the synthetic classroom data.

## 12. Reproducibility and responsible use

- Use fixed random seeds where practical and document remaining sources of nondeterminism.
- Keep training, validation, and testing roles separate; never tune hyperparameters on the final test set.
- Avoid leakage when calculating statistics, scaling, forming windows, or selecting features.
- Treat the included datasets as **synthetic educational examples**; do not describe their generated patterns as scientific findings about actual Jakarta weather or market prices.
- Document any modifications to the original exercises, including new datasets, targets, horizons, and metrics.
- Evaluate forecasts against a transparent baseline before claiming model improvements.
- Use real operational data only with appropriate permissions, data-protection controls, and provenance documentation.

## 13. References and further reading

**References listed in the official AID01 syllabus:**

1. Warren B. Powell, *Sequential Decision Analytics and Modeling: Modeling with Python*, 2nd ed., now Publishers, 2024. **Primary reference, as listed in the syllabus.**
2. Thomas A. Runkler, *Data Analytics (Models and Algorithms for Intelligent Data Analysis)*, Springer, 2025. **Supporting reference, as listed in the syllabus.**

**Useful software documentation:**

- [TensorFlow / Keras](https://www.tensorflow.org/guide/keras)
- [Keras recurrent layers](https://keras.io/api/layers/recurrent_layers/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [statsmodels time-series analysis](https://www.statsmodels.org/stable/tsa.html)
- [Google Colab](https://colab.research.google.com/)

## 14. Usage, attribution, and contributions

Materials in this repository are intended to support instruction and independent study in the AID01 course. Respect the university's rules concerning redistribution of course slides and teaching materials. **No open-source license is implied by the existence of a public GitHub repository.** Add an explicit `LICENSE` file only after confirming that the material may be released under those terms.

When submitting an improvement, describe the educational goal, verify that the notebook runs from a clean runtime with its stated dataset, and include the results needed to reproduce any performance claim. Avoid committing private information, credentials, or real student records.

---

**Course:** Sequential Data Analytics and Modeling (AID01)  
**Program:** Undergraduate Artificial Intelligence Program — Faculty of Technology and Design  
**Institution:** Universitas Bunda Mulia  
**Scope:** Sequential data, classical forecasting, RNN, LSTM, GRU, and responsible model evaluation
