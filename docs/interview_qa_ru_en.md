# Подготовка к интервью — Вопросы и ответы
# Interview Preparation — Questions & Answers

---

## Структура интервью (90 минут)

| Блок | Время | Тема |
|------|-------|------|
| 1 | 10–15 мин | Знакомство, «расскажи про себя» |
| 2 | 45–60 мин | **Разбор проекта** (ключевой блок) |
| 3 | 15–20 мин | General DS / product, вопросы от тебя |

---

---

# БЛОК 1 — Разбор решения

---

## ❓ 1. Объясни своё решение за 2–3 минуты

### 🇷🇺 Русский

Задача — оптимизировать выдачу увеличений кредитных лимитов 30 000 клиентам при ограничении по капиталу и неопределённости.

Я подошёл к этому как к **decision pipeline из 5 этапов**:

1. **EDA** (Notebook 01) — нашёл 6 структурных аномалий, подтвердил что данные синтетические, историческая аллокация случайна. Это означает: задача оптимизации, а не предсказания.
2. **Марковская модель риска** (Notebook 02) — построил цепь Маркова с 4 состояниями (Prime / Near-Prime / Sub-Prime / Default), калибровал PD на бенчмарках. Годовая PD: Prime 10.6%, Near-Prime 19.5%, Sub-Prime 40.4%.
3. **Модель спроса** (Notebook 03) — оценил вероятность принятия оффера через proxy-таргет. AUC слабый (0.61), поэтому перешёл на сегментную модель. Протестировал 3 макросценария (base / stress / benign).
4. **Оптимизация** (Notebook 04) — LP-задача: максимизировать risk-adjusted NPV при капитальном ограничении 30%. Стратегия `lp_round` — предложить увеличение 18 769 клиентам. Детерминированный NPV = +$322.5k.
5. **Монте-Карло** (Notebook 05) — 5 000 симуляций показали: средний NPV = −$737k, VaR95 = −$781k. Стратегия убыточна при текущей PD-калибровке.

**Главный вывод:** детерминированная оптимизация выглядит прибыльной, но не выдерживает стохастической проверки. Broad rollout не рекомендуется. Нужен пилот на Prime-сегменте и перекалибровка PD на реальных данных.

### 🇬🇧 English

The goal was to optimize loan limit increases for 30,000 customers under capital constraints and uncertainty.

I built it as a **5-stage decision pipeline**:

1. **EDA** (Notebook 01) — found 6 structural anomalies, confirmed synthetic data with random historical allocation. This means it's an optimization problem, not a prediction problem.
2. **Markov risk model** (Notebook 02) — built a 4-state Markov chain (Prime / Near-Prime / Sub-Prime / Default), calibrated PD from benchmarks. Annual PD: Prime 10.6%, Near-Prime 19.5%, Sub-Prime 40.4%.
3. **Demand model** (Notebook 03) — estimated acceptance probability via proxy target. AUC was weak (0.61), so I fell back to a segment-based model. Tested 3 macro scenarios (base / stress / benign).
4. **Optimization** (Notebook 04) — LP problem: maximize risk-adjusted NPV under a 30% capital constraint. The `lp_round` strategy offers increases to 18,769 customers. Deterministic NPV = +$322.5k.
5. **Monte Carlo** (Notebook 05) — 5,000 simulations showed: mean NPV = −$737k, VaR95 = −$781k. The strategy is loss-making under current PD calibration.

**Key insight:** the deterministic optimum looks profitable but doesn't survive stochastic validation. Broad rollout is not recommended. A pilot on Prime-only with PD recalibration on real data is the right path.

---

## ❓ 2. Почему deterministic ≠ simulation?

### 🇷🇺 Русский

Разрыв между детерминированным NPV (+$322.5k) и стохастическим (−$737k) — это **≈ $1.06 млн**.

Причина: детерминированная модель использует **математическое ожидание** — средний дефолт, среднее принятие. Но Монте-Карло моделирует **каждого клиента отдельно**: каждый принимает или нет (Binomial), каждый дефолтит или нет (Binomial). Когда дефолт случается, потери = LGD × EAD, то есть 55% от половины размера кредита — это существенно больше, чем $40 прибыли с увеличения.

Иначе говоря: **асимметрия payoff**. Прибыль с одного увеличения = $40, а потери от одного дефолта = сотни долларов. В детерминированной модели эти потери «размазаны» по среднему. В стохастической — реализуются в полном объёме для конкретных клиентов.

Плюс: когда увеличивается acceptance, растут и дефолты (больше принятий → больше exposure → больше дефолтов). Sensitivity-анализ это подтверждает: +10% p_accept ухудшает NPV на $73k.

### 🇬🇧 English

The gap between deterministic NPV (+$322.5k) and stochastic (−$737k) is **≈ $1.06M**.

The deterministic model uses **expected values** — average default rate, average acceptance. Monte Carlo simulates **each customer individually**: each accepts or not (Binomial), each defaults or not (Binomial). When a default occurs, the loss = LGD × EAD = 55% of half the loan amount — far exceeding the $40 profit per increase.

This is a **payoff asymmetry** problem. Revenue per increase = $40, while loss per default = hundreds of dollars. In the deterministic model, losses are spread across the mean. In simulation, they materialize in full for individual customers.

Additionally: higher acceptance means more defaults (more acceptances → more exposure → more defaults). The sensitivity grid confirms this: +10% p_accept worsens NPV by $73k.

---

## ❓ 3. Где в твоей модели самое слабое место?

### 🇷🇺 Русский

**Слабое место №1 — оценка PD (вероятности дефолта).**

В датасете нет явных меток дефолта. Нет временных рядов платёжной истории. Поэтому PD — это **scenario-based approximation**, калиброванная на внешних бенчмарках (FRED delinquency/charge-off rates), а не оценённая на реальных vintage roll-rate данных.

Это означает, что весь pipeline чувствителен к PD-калибровке. Sensitivity-анализ это показывает: сдвиг PD на +10% ухудшает среднее NPV на ~$114k.

**Слабое место №2 — demand model (AUC = 0.61).**

Из-за аномалии A4 (признаки независимы от таргета) логистическая регрессия не даёт надёжного сигнала. Я использовал сегментный fallback, но это скорее assumption, чем empirical model.

Поэтому я намеренно сделал Notebook 05 (Monte Carlo) — чтобы проверить, выдерживает ли стратегия неопределённость, а не просто доверять точечным оценкам.

### 🇬🇧 English

**Weakness #1 — PD estimation.**

The dataset contains no explicit default labels and no time-series payment history. So PD is a **scenario-based approximation** calibrated from external benchmarks (FRED delinquency/charge-off rates), not estimated from actual vintage roll-rate data.

This makes the entire pipeline sensitive to PD calibration. Sensitivity analysis shows: +10% PD shift worsens mean NPV by ~$114k.

**Weakness #2 — demand model (AUC = 0.61).**

Due to anomaly A4 (features are independent of the target), logistic regression provides no reliable signal. I used a segment-based fallback, which is more of an assumption than an empirical model.

This is precisely why I built Notebook 05 (Monte Carlo) — to stress-test the strategy under uncertainty rather than trusting point estimates.

---

## ❓ 4. Почему ты не использовал total_profit из данных?

### 🇷🇺 Русский

Потому что поле `total_profit` в данных — **синтетический артефакт**, а не бизнес-цель.

Я проверил: `total_profit = max(0, n−2) × $40` для всех 30 000 строк. Это значит: первые два увеличения «бесплатны» — прибыль начинается только с третьего.

Но в задании сказано: прибыль = $40 за **каждое** увеличение. Формула `40 × n`.

Три кандидата сравнивались в Notebook 01:
- `40 × n` (по заданию) — не совпадает с данными
- `40 × max(0, n−1)` (activation hypothesis) — тоже не совпадает
- `40 × max(0, n−2)` — единственное совпадение с данными

Я использовал формулу из задания (`40 × n`) как objective для оптимизации, потому что бизнес-задание — это ground truth для постановки задачи, а поле в данных — артефакт генерации синтетики.

### 🇬🇧 English

Because `total_profit` in the dataset is a **synthetic artifact**, not the business objective.

I verified: `total_profit = max(0, n−2) × $40` for all 30,000 rows. This means the first two increases are "free" — profit starts only from the third.

But the assessment specification says: profit = $40 per **each** increase. Formula: `40 × n`.

Three candidates were compared in Notebook 01:
- `40 × n` (per spec) — doesn't match data
- `40 × max(0, n−1)` (activation hypothesis) — doesn't match either
- `40 × max(0, n−2)` — the only match with data

I used the spec formula (`40 × n`) as the optimization objective, because the business specification is the ground truth for problem formulation, while the data field is a generation artifact.

---

## ❓ 5. Почему Markov без временных данных?

### 🇷🇺 Русский

В идеальном мире мне нужна **панель данных** — несколько временных точек для каждого клиента, чтобы оценить реальные переходы между состояниями (vintage roll-rate matrix).

Но датасет — это **snapshot одного момента**: один ряд на клиента, без истории переходов. Я не могу напрямую оценить матрицу переходов.

Решение: **scenario-based Markov approximation**.

1. Разбил клиентов на 3 риск-тира по `on_time_pct` (Prime ≥ 95%, Near-Prime 85–95%, Sub-Prime < 85%)
2. Калибровал ежемесячные PD на внешних бенчмарках (FRED data для unsecured consumer lending)
3. Задал правдоподобные вероятности миграций (апгрейды, даунгрейды)
4. Проверил через 3 сценария (soft / base / conservative) и sensitivity sweep

Это **структурированное бизнес-допущение** — я чётко это коммуницирую. И именно поэтому финальный шаг — Monte Carlo: он показывает, что даже при центральной калибровке стратегия не выживает.

### 🇬🇧 English

Ideally I need **panel data** — multiple time points per customer to estimate actual state transitions (vintage roll-rate matrix).

But the dataset is a **single-point snapshot**: one row per customer, no transition history. I can't directly estimate a transition matrix.

Solution: **scenario-based Markov approximation**.

1. Split customers into 3 risk tiers by `on_time_pct` (Prime ≥ 95%, Near-Prime 85–95%, Sub-Prime < 85%)
2. Calibrated monthly PDs from external benchmarks (FRED data for unsecured consumer lending)
3. Set plausible migration probabilities (upgrades, downgrades)
4. Validated through 3 scenarios (soft / base / conservative) and sensitivity sweep

This is a **structured business assumption** — I communicate this explicitly. And this is exactly why the final step is Monte Carlo: it shows that even under central calibration, the strategy doesn't survive.

---

## ❓ 6. Как работает твоя оптимизация?

### 🇷🇺 Русский

**Формулировка** (Notebook 04):

**Переменная решения:** $x_i$ — число предложенных увеличений клиенту $i$.

**Целевая функция:**

$$\max \sum_{i} x_i \times p_{\text{accept},i} \times EV_i \times \text{discount\_factor}$$

Где:
- $p_{\text{accept},i}$ — вероятность принятия оффера (из Notebook 03)
- $EV_i = 40 - PD_i \times LGD \times EAD_i$ — risk-adjusted ожидаемая ценность (из Notebook 02)
- discount factor — 19% годовых, пересчитанный в месячный

**Ограничения:**
1. $0 \leq x_i \leq \text{eligible}_i \times \text{max\_increases}_i$ — правомочность + лимит 6 в год
2. $\sum_i x_i \times p_{\text{accept},i} \times EAD_i \leq 0.30 \times \sum_i \text{initial\_loan}_i$ — капитальный лимит 30%

**Три стратегии:**
- `lp_relax` — LP-релаксация (дробные $x_i$, верхняя граница)
- `lp_round` — целочисленное округление LP (основная рекомендация)
- `greedy` — жадная эвристика

**Почему `lp_round`?** Gap между LP-relax и LP-round < 0.001% — практически оптимально, но целочисленно (можно развернуть в продакшене).

**Почему именно LP, а не ML?** Потому что аномалия A4 показала: признаки независимы от таргета. ML-модель не может найти сигнал. LP работает с risk-adjusted EV, который рассчитан аналитически.

### 🇬🇧 English

**Formulation** (Notebook 04):

**Decision variable:** $x_i$ — number of offered increases to customer $i$.

**Objective:**

$$\max \sum_{i} x_i \times p_{\text{accept},i} \times EV_i \times \text{discount\_factor}$$

Where:
- $p_{\text{accept},i}$ — acceptance probability (from Notebook 03)
- $EV_i = 40 - PD_i \times LGD \times EAD_i$ — risk-adjusted expected value (from Notebook 02)
- discount factor — 19% annual, converted to monthly

**Constraints:**
1. $0 \leq x_i \leq \text{eligible}_i \times \text{max\_increases}_i$ — eligibility + cap of 6 per year
2. $\sum_i x_i \times p_{\text{accept},i} \times EAD_i \leq 0.30 \times \sum_i \text{initial\_loan}_i$ — 30% capital limit

**Three strategies:**
- `lp_relax` — LP relaxation (fractional $x_i$, upper bound)
- `lp_round` — integer rounding of LP solution (main recommendation)
- `greedy` — greedy heuristic

**Why `lp_round`?** The gap between LP-relax and LP-round is < 0.001% — practically optimal but integer-valued and deployable.

**Why LP, not ML?** Because anomaly A4 showed: features are independent of the target. An ML model can't find signal. LP works with analytically computed risk-adjusted EV.

---

---

# БЛОК 2 — Risk & Business Thinking

---

## ❓ 7. Что бы ты сделал в продакшене?

### 🇷🇺 Русский

Четыре конкретных шага:

1. **Перекалибровать PD** на реальных longitudinal данных — нужна временная панель платежей и дефолтов, а не snapshot. Это устранит главную неопределённость.

2. **Пилот на Prime-сегменте** — запустить champion/challenger на Prime и верхней части Near-Prime. Sub-Prime исключить из первой волны.

3. **Мониторинг** — отслеживать realized default rate, contribution profit, capital usage, acceptance rate. Сравнивать с модельными ожиданиями.

4. **Итеративная перекалибровка** — после пилота пересмотреть пороги стратегии, расширить на Near-Prime, если реальная PD ниже модельной.

Также: production-grade модель спроса (с A/B тестом для causal estimation), real-time scoring, автоматическое обновление PD при макрошоках.

### 🇬🇧 English

Four concrete steps:

1. **Recalibrate PD** on real longitudinal data — need a temporal panel of payments and defaults, not a snapshot. This removes the main uncertainty.

2. **Pilot on Prime segment** — run champion/challenger on Prime and upper Near-Prime. Exclude Sub-Prime from the first wave.

3. **Monitoring** — track realized default rate, contribution profit, capital usage, acceptance rate. Compare against model expectations.

4. **Iterative recalibration** — after the pilot, revise strategy thresholds, expand to Near-Prime if realized PD is below model PD.

Also: production-grade demand model (with A/B test for causal estimation), real-time scoring, automatic PD updates on macro shocks.

---

## ❓ 8. Ты уверен, что стратегия плохая?

### 🇷🇺 Русский

Нет, не уверен что стратегия плохая. Я уверен, что **при текущей PD-калибровке** она убыточна.

Ключевое слово — «при текущей калибровке». PD калиброваны на внешних бенчмарках без реальных данных о дефолтах. Это консервативная оценка.

Если реальная PD окажется ниже — а это вполне вероятно, особенно для Prime-сегмента — стратегия станет прибыльной. Sensitivity-анализ показывает: при PD × 0.9 средний NPV улучшается на ~$115k.

**Стратегия как decision rule (`lp_round`, EV > 0) — правильная.** Вопрос не в rule, а в калибровке параметров.

Это как раз то, что pipeline показывает: правильный framework + честная оценка неопределённости = возможность принять осознанное бизнес-решение. Не «модель говорит делай» или «модель говорит не делай», а «вот при каких условиях делай».

### 🇬🇧 English

No, I'm not saying the strategy is bad. I'm saying it's **loss-making under current PD calibration**.

The key phrase is "under current calibration." PDs are calibrated from external benchmarks without actual default data. This is a conservative estimate.

If real PD turns out lower — which is plausible, especially for Prime — the strategy becomes profitable. Sensitivity analysis shows: at PD × 0.9, mean NPV improves by ~$115k.

**The strategy as a decision rule (`lp_round`, EV > 0) is correct.** The question is not the rule, but the parameter calibration.

This is exactly what the pipeline shows: correct framework + honest uncertainty assessment = ability to make an informed business decision. Not "the model says do it" or "the model says don't," but "here are the conditions under which to proceed."

---

## ❓ 9. Что если PD уменьшить?

### 🇷🇺 Русский

Стратегия **станет прибыльной**.

Конкретные цифры из sensitivity grid (Notebook 05, base scenario):
- PD × 0.9: средний NPV = −$622k (улучшение на $115k)
- PD × 0.9 и p_accept × 0.9: средний NPV = −$560k

Но даже при PD × 0.9 результат всё ещё отрицательный. Чтобы выйти в плюс, нужно более существенное снижение PD — примерно в 2–3 раза от текущей калибровки.

Это реалистично? Вполне. Текущая годовая PD для Sub-Prime = 40.4% — это довольно агрессивная калибровка. Реальный unsecured consumer portfolio с established клиентами (уже имеющими кредитную историю) может иметь PD существенно ниже.

Вывод: **модель правильно идентифицирует точку безубыточности** по PD. Задача прода — измерить реальную PD и сравнить с этим порогом.

### 🇬🇧 English

The strategy **becomes profitable**.

Specific numbers from the sensitivity grid (Notebook 05, base scenario):
- PD × 0.9: mean NPV = −$622k (improvement of $115k)
- PD × 0.9 and p_accept × 0.9: mean NPV = −$560k

But even at PD × 0.9 the result is still negative. To break even, PD needs to decrease more substantially — roughly 2–3× from current calibration.

Is that realistic? Quite plausible. Current annual PD for Sub-Prime = 40.4% — that's fairly aggressive. A real unsecured consumer portfolio with established customers may have significantly lower PD.

The takeaway: **the model correctly identifies the break-even PD threshold**. The production task is to measure actual PD and compare it against this threshold.

---

---

# БЛОК 3 — DS / Modeling

---

## ❓ 10. Как бы ты улучшил PD-модель?

### 🇷🇺 Русский

1. **Longitudinal data** — временная панель минимум 12–24 месяца: ежемесячный статус платежей, даты просрочек, суммы. Это позволит оценить реальную матрицу переходов, а не калибровать на бенчмарках.

2. **Survival model** (Cox PH или AFT) — зависимость от времени: не просто «дефолтнет/не дефолтнет», а «когда».

3. **Scorecard / градиентный бустинг** — включить дополнительные признаки: bureau score, debt-to-income, payment velocity, vintage.

4. **Out-of-time validation** — тренировка на одном периоде, тестирование на следующем. Проверка стабильности PD-оценок.

5. **Стресс-тестирование** — привязать PD к макрофакторам (unemployment rate, interest rates) через регрессию или conditional model.

### 🇬🇧 English

1. **Longitudinal data** — at least 12–24 months of monthly payment status, delinquency dates, amounts. This allows estimating a real transition matrix instead of benchmark calibration.

2. **Survival model** (Cox PH or AFT) — incorporating time dependency: not just "will default" but "when."

3. **Scorecard / gradient boosting** — include additional features: bureau score, debt-to-income, payment velocity, vintage.

4. **Out-of-time validation** — train on one period, test on the next. Check PD estimate stability.

5. **Stress-testing** — link PD to macro factors (unemployment rate, interest rates) via regression or conditional model.

---

## ❓ 11. Как бы ты улучшил demand model?

### 🇷🇺 Русский

Текущая модель слабая (AUC = 0.61) по объективной причине: аномалия A4 — признаки не несут сигнала.

Для улучшения:

1. **A/B тест** — рандомизированный эксперимент: случайное подмножество клиентов получает оффер, отслеживаем кто принял. Это даёт **causal** acceptance rate, а не proxy.

2. **Response modeling** — после A/B: двухэтапная модель (propensity to respond × conditional acceptance).

3. **Дополнительные признаки** — engagement data (login frequency, app usage), channel (SMS vs push vs call), time of offer, previous offer history.

4. **Uplift modeling** — моделировать не «кто примет», а «для кого оффер изменит поведение» (incremental response).

### 🇬🇧 English

The current model is weak (AUC = 0.61) for an objective reason: anomaly A4 — features carry no signal.

To improve:

1. **A/B test** — randomized experiment: random subset gets offers, track who accepts. This gives **causal** acceptance rate, not a proxy.

2. **Response modeling** — post A/B: two-stage model (propensity to respond × conditional acceptance).

3. **Additional features** — engagement data (login frequency, app usage), channel (SMS vs push vs call), time of offer, previous offer history.

4. **Uplift modeling** — model not "who will accept" but "for whom the offer changes behavior" (incremental response).

---

## ❓ 12. Какие метрики ты использовал бы?

### 🇷🇺 Русский

**Метрики модели:**
- AUC-ROC (для demand model) — дискриминирующая способность
- KS-статистика — sepарация
- Brier score — калибровка вероятностей
- Gini — для PD модели

**Бизнес-метрики (для пилота):**
- **Expected NPV** — основной KPI
- **Realized default rate** — vs модельная PD
- **Acceptance rate** — vs модельная p_accept
- **Capital usage** — % от лимита
- **Contribution profit** — выручка минус потери по портфелю
- **VaR95 / CVaR95** — downside risk

**Операционные:**
- Offer-to-acceptance conversion
- Time-to-default distribution
- Segment mix (Prime/Near/Sub) в реальных раздачах

### 🇬🇧 English

**Model metrics:**
- AUC-ROC (for demand model) — discriminative ability
- KS statistic — separation
- Brier score — probability calibration
- Gini — for PD model

**Business metrics (for pilot):**
- **Expected NPV** — primary KPI
- **Realized default rate** — vs model PD
- **Acceptance rate** — vs model p_accept
- **Capital usage** — % of limit
- **Contribution profit** — revenue minus portfolio losses
- **VaR95 / CVaR95** — downside risk

**Operational:**
- Offer-to-acceptance conversion
- Time-to-default distribution
- Segment mix (Prime/Near/Sub) in actual rollout

---

---

# БЛОК 4 — Hard Thinking

---

## ❓ 13. Если бы у тебя были реальные данные — что бы ты сделал?

### 🇷🇺 Русский

Три ключевых изменения:

1. **PD из данных, а не бенчмарков.** Vintage-анализ: для каждого cohort (месяц выдачи) построить кривую delinquency → default. Это даст реальную transition matrix, а не scenario-based approximation. Ожидаю, что PD будет ниже текущей калибровки, особенно для Prime.

2. **Causal demand model.** С реальными логами offer/accept — обучить uplift model. Не proxy, а direct observation. Это уберёт неопределённость из Notebook 03.

3. **Dynamic re-optimization.** С real-time данными — обновлять PD и p_accept ежемесячно, пересчитывать аллокацию. Текущая модель — static snapshot. Production-версия должна быть streaming.

**Pipeline остаётся тем же самым** — EDA → Risk → Demand → Optimize → Validate. Меняется качество входных данных, не архитектура.

### 🇬🇧 English

Three key changes:

1. **PD from data, not benchmarks.** Vintage analysis: for each cohort (origination month), build a delinquency → default curve. This gives a real transition matrix, not a scenario-based approximation. I expect PD to be lower than current calibration, especially for Prime.

2. **Causal demand model.** With real offer/accept logs — train an uplift model. Not a proxy, but direct observation. This removes uncertainty from Notebook 03.

3. **Dynamic re-optimization.** With real-time data — update PD and p_accept monthly, recompute allocation. The current model is a static snapshot. The production version should be streaming.

**The pipeline remains the same** — EDA → Risk → Demand → Optimize → Validate. What changes is input data quality, not architecture.

---

## ❓ 14. Как учитывать экономику (inflation, rates)?

### 🇷🇺 Русский

В текущем проекте макрофакторы введены как **scenario multipliers** в Notebook 03:

| Фактор | Stress | Base | Benign |
|--------|--------|------|--------|
| Инфляция | +3 п.п. | 0 | −1 п.п. |
| Безработица | +2 п.п. | 0 | −1 п.п. |
| Процентная ставка | +100 б.п. | 0 | −50 б.п. |

Они влияют на p_accept (спрос) и косвенно на PD (через сценарий).

**Для production-версии:**

1. **Макро-conditional PD model**: PD(customer, macro) — регрессия PD на macro-variables (unemployment rate, CPI, base rate). IFRS 9 forward-looking approach.

2. **Scenario-weighted optimization**: не один сценарий, а взвешенная смесь (60% base, 25% stress, 15% benign) → robust optimization.

3. **Regime switching**: Markov-switching model для определения текущего макро-режима, автоматический выбор сценария.

### 🇬🇧 English

In the current project, macro factors are introduced as **scenario multipliers** in Notebook 03:

| Factor | Stress | Base | Benign |
|--------|--------|------|--------|
| Inflation | +3pp | 0 | −1pp |
| Unemployment | +2pp | 0 | −1pp |
| Interest rate | +100bp | 0 | −50bp |

They affect p_accept (demand) and indirectly PD (via scenario).

**For production:**

1. **Macro-conditional PD model**: PD(customer, macro) — regress PD on macro variables (unemployment rate, CPI, base rate). IFRS 9 forward-looking approach.

2. **Scenario-weighted optimization**: not a single scenario, but a weighted mix (60% base, 25% stress, 15% benign) → robust optimization.

3. **Regime switching**: Markov-switching model to detect the current macro regime, automatic scenario selection.

---

## ❓ 15. Как бы ты сделал A/B тест?

### 🇷🇺 Русский

**Дизайн:**

1. **Рандомизация** — случайное разбиение eligible клиентов на treatment (получают оффер) и control (не получают). Стратификация по risk-тиру, чтобы тиры были сбалансированы.

2. **Размер выборки** — power analysis: нужно определить minimal detectable effect (например, разница в acceptance rate 5 п.п.), significance level 5%, power 80%. Для наших 25 000 eligible — более чем достаточно.

3. **Что измеряем:**
   - Primary: acceptance rate (treatment vs control — для demand model)
   - Secondary: default rate at 6/12 months (для PD validation)
   - Business: incremental profit per customer

4. **Длительность** — минимум 6 месяцев, чтобы увидеть дефолты. 12 месяцев — для полной годовой PD.

5. **Guardrails** — если realized default rate в treatment группе превышает порог (например, 2× от ожидаемой) — останавливаем.

**Что это даст:** causal p_accept (не proxy), causal PD impact, и реальные юниты для перекалибровки всего pipeline.

### 🇬🇧 English

**Design:**

1. **Randomization** — random split of eligible customers into treatment (get offer) and control (no offer). Stratified by risk tier for balance.

2. **Sample size** — power analysis: determine minimal detectable effect (e.g., 5pp acceptance rate difference), significance 5%, power 80%. With 25,000 eligible customers — more than enough.

3. **What we measure:**
   - Primary: acceptance rate (treatment vs control — for demand model)
   - Secondary: default rate at 6/12 months (for PD validation)
   - Business: incremental profit per customer

4. **Duration** — minimum 6 months for defaults to materialize. 12 months for full annual PD.

5. **Guardrails** — if realized default rate in treatment exceeds threshold (e.g., 2× expected) — stop the test.

**What this gives:** causal p_accept (not proxy), causal PD impact, and real data for recalibrating the entire pipeline.

---

---

# ОПАСНЫЕ ВОПРОСЫ

---

## ⚠️ «Почему вообще твоя модель полезна, если результат отрицательный?»

### 🇷🇺 Русский

**Именно потому что она показала отрицательный результат — она полезна.**

Представьте альтернативу: кто-то строит только детерминированную модель, видит +$322k, говорит «запускаем» — и теряет $700k+.

Мой pipeline предотвращает это:
- Показывает что выглядит прибыльным
- Но проверяет через симуляцию
- И обнаруживает что risk dominates
- И даёт конкретный path forward: пилот на Prime, перекалибровка PD

Ценность модели — не в том, чтобы сказать «да». Ценность — в **корректной оценке risk-adjusted outcomes**. Это позволяет бизнесу принять осознанное решение, а не слепое.

### 🇬🇧 English

**Precisely because it showed a negative result — it is useful.**

Consider the alternative: someone builds only a deterministic model, sees +$322k, says "let's launch" — and loses $700k+.

My pipeline prevents that:
- Shows what looks profitable
- But validates through simulation
- And discovers that risk dominates
- And provides a concrete path forward: pilot on Prime, recalibrate PD

The model's value is not in saying "yes." The value is in **correctly assessing risk-adjusted outcomes**. This lets the business make an informed decision, not a blind one.

---

## ⚠️ «Ты уверен, что не ошибся?»

### 🇷🇺 Русский

Я валидировал результаты на нескольких уровнях:

1. **Sanity-checks** в каждом ноутбуке — аналитические инварианты (capital ≤ limit, LP-relax ≥ integer, VaR ≤ mean, stress ≤ base ≤ benign).

2. **Sensitivity analysis** — 2D grid (p_accept × PD, 9 комбинаций). Результат устойчив: NPV отрицательный во всём тестируемом диапазоне, но улучшается при снижении PD — это логично.

3. **Три стратегии × три сценария** = 9 точек. Все подтверждают один паттерн.

4. **5 000 Monte Carlo прогонов** — распределение NPV стабильное (std ~$27k при mean −$737k, коэффициент вариации ~4%).

Результат сильно зависит от PD-assumptions — это **ожидаемо** в кредитном моделировании. Я это не скрываю, а наоборот — делаю центральным выводом.

### 🇬🇧 English

I validated results at multiple levels:

1. **Sanity checks** in every notebook — analytical invariants (capital ≤ limit, LP-relax ≥ integer, VaR ≤ mean, stress ≤ base ≤ benign).

2. **Sensitivity analysis** — 2D grid (p_accept × PD, 9 combinations). Results are consistent: NPV is negative throughout the tested range, but improves as PD decreases — which is logical.

3. **Three strategies × three scenarios** = 9 data points. All confirm the same pattern.

4. **5,000 Monte Carlo runs** — NPV distribution is stable (std ~$27k vs mean −$737k, coefficient of variation ~4%).

The result is strongly sensitive to PD assumptions — this is **expected** in credit modeling. I don't hide this; I make it the central finding.

---

## ⚠️ «Что бы ты сделал по-другому, если бы начинал заново?»

### 🇷🇺 Русский

Три вещи:

1. **Больше времени на EDA → PD bridge.** Я бы попробовал вывести PD из самих данных через proxy (например, используя `on_time_pct` как survival signal), а не только из внешних бенчмарков. Это дало бы более органичную калибровку.

2. **Bayesian estimation для PD.** Вместо точечных PD + scenario sweep — posterior distribution. Это позволило бы propagate uncertainty через весь pipeline more naturally.

3. **Profit formula sensitivity.** Я зафиксировал $40 per increase по заданию. В production я бы параметризовал profit как переменную и показал break-even profit threshold.

Но архитектуру pipeline менять бы не стал — она правильная.

### 🇬🇧 English

Three things:

1. **More time on EDA → PD bridge.** I would try deriving PD from the data itself through proxies (e.g., using `on_time_pct` as a survival signal), not just external benchmarks. This would give a more organic calibration.

2. **Bayesian estimation for PD.** Instead of point PD + scenario sweep — posterior distribution. This would propagate uncertainty through the pipeline more naturally.

3. **Profit formula sensitivity.** I fixed $40 per increase per the spec. In production I would parameterize profit as a variable and show the break-even profit threshold.

But I wouldn't change the pipeline architecture — it's correct.

---

---

# СОВЕТЫ ПО ПОДАЧЕ

---

## Три золотых правила

### 1. Говори просто

| ❌ Не так | ✅ А так |
|-----------|---------|
| "stochastic process under uncertainty with absorbing states..." | "we simulate default scenarios to understand downside risk" |
| "LP-relaxation with integer rounding heuristic..." | "we optimize who gets an offer under a capital limit" |
| "scenario-based Markov approximation..." | "we estimate default probability from risk tiers because we don't have default history" |

### 2. Всегда связывай с бизнесом

Каждый технический ответ должен заканчиваться бизнес-выводом:
- «...и поэтому broad rollout не рекомендуется»
- «...что позволяет бизнесу принять осознанное решение»
- «...поэтому нужен пилот на Prime-сегменте»

### 3. Не защищай модель — признавай слабости

| ❌ Защита | ✅ Честность |
|-----------|-------------|
| "Моя модель точная" | "PD — главная неопределённость, и я это показал" |
| "Результат правильный" | "Результат зависит от калибровки, вот sensitivity" |
| "Monte Carlo всё доказывает" | "Monte Carlo показывает risk, но PD нужно перекалибровать" |

---

## Ключевые фразы для интервью

| Ситуация | Фраза (EN) | Фраза (RU) |
|----------|-----------|-------------|
| Объяснение pipeline | "I built a full decision pipeline, not just a model" | "Я построил полный decision pipeline, а не просто модель" |
| Про отрицательный NPV | "The value is in preventing a bad decision" | "Ценность в том, что мы предотвращаем убыточное решение" |
| Про слабые места | "The main uncertainty is PD calibration — and I'm transparent about it" | "Главная неопределённость — калибровка PD, и я это явно показываю" |
| Про next steps | "Pilot on Prime, measure real PD, then expand" | "Пилот на Prime, измерить реальную PD, потом расширять" |
| Про synthetics | "The data is synthetic — I proved it — so I built from first principles" | "Данные синтетические — я это доказал — поэтому строил с нуля" |
