# RIDGE--REGRESSION-ON-CRYPTO-SENTIMENT-PREDICTION

##TABLE OF CONTENTS

"[🔍 Project Overview]"


"[📝 Context & Background]"


"[❗Problem Statement]"


"[🎯 Goal of the Analysis]"


"[📌 Focus Areas]"

"[🌐Source Of Data]"

"[📊 About the Dataset]"

"[🛠️ Tech Stack & Libraries Used]"

"[🔧 Approach & Techniques]"

"[🧹 Preparing the Data]"

"[📈 Uncovering Patterns (EDA)]"

"[Assumptions Of Ridge Regression]"

"[Model Summary And Interpretation]"

"[✅ Recommendations]"

"[🙏 Credits & Mentions]"

"[📬Let's Connect]"


## 🔎Project Overview

This project explores confidence levels in cryptocurrency predictions using Ridge Regression, a regularized linear modeling approach that addresses multicollinearity among features. By analyzing real-world crypto sentiment and market data, the model uncovers the most influential variables that shape prediction confidence.Through assumption testing (independence of errors, normality of residuals, and multicollinearity), we ensured the model's reliability.

## 📰Context And Background

Ridge Regression is applied to cryptocurrency data from June and July 2025 to examine the factors influencing prediction confidence across different digital assets. The focus is on identifying whether variables such as sentiment scores, market activity, and cryptocurrency categories affect the model’s confidence in forecasting trends.

## ❗ Problem Statement

In the rapidly evolving cryptocurrency market, traders often make decisions based on volatile trends, social sentiment, and fragmented information—leading to inconsistent confidence in price predictions. This uncertainty increases the risk of poor trading decisions and financial losses.There is a need to identify which market and sentiment-related factors most strongly influence prediction confidence to support more reliable decision-making. However, multicollinearity among these factors complicates model accuracy and interpretability. This project addresses the challenge by applying Ridge Regression to quantify and rank the influence of key variables on prediction confidence, while effectively handling multicollinearity and ensuring robust, data-driven insights for crypto traders and analysts.

## Goal Of The Analysis

The objective of this project is to apply Ridge Regression to cryptocurrency data from June and July 2025 in order to:

✔ Identify the most influential market and sentiment variables affecting prediction confidence.

✔ Address multicollinearity among predictors using regularization techniques.

✔ Evaluate model performance through metrics such as R² and MSE after refining the feature set.

✔ Provide insights that can guide traders, analysts, and enthusiasts toward more confident and data-driven decisions in the volatile crypto market.

## Focus Area

📌 Descriptive Statistics:

Summarized key metrics to understand the distribution, central tendency, and variability of market and sentiment features.

📌 Exploratory Data Analysis (EDA):

Visualized relationships between variables and identified patterns, trends, and potential outliers affecting prediction confidence.

📌 Multicollinearity Assessment:

Checked for high correlation among independent variables using Variance Inflation Factor (VIF) to ensure reliable model estimates.

📌 Assumption Testing for Ridge Regression:

Validated key assumptions such as independence of errors, normality of residuals, and low multicollinearity.

📌 Feature Selection & Preprocessing:

Removed or transformed highly correlated variables (e.g., cryptocurrency_Bitcoin) and scaled the dataset for model efficiency.

📌 Model Implementation – Ridge Regression:

Applied Ridge Regression to quantify the influence of each variable on prediction confidence and optimized the model using cross-validation.

📌 Model Evaluation:

Assessed performance using metrics like R² score and Mean Squared Error (MSE) before and after feature refinement.

## Source Of Data

The dataset used for this project was downloaded from Kaggle — a leading platform for open data and machine learning projects. The dataset is titled "[crypto_sentiment_prediction_dataset]" and contains cryptocurrency-related features such as sentiment scores, trading volume, and confidence predictions.

It covers crypto market activities and sentiment information from June to July 2025, making it ideal for time-relevant prediction analysis in the fast-moving digital asset space.

🔗 Dataset Link: [https://www.kaggle.com/datasets/pratyushpuri/crypto-market-sentiment-and-price-dataset-2025]


## About The Dataset

The dataset comprises detailed cryptocurrency-related observations collected in June and July 2025, designed to support the analysis of factors influencing prediction confidence in digital asset forecasting.

It includes 14 variables combining real-time market metrics, sentiment analysis, and technical indicators. Below is a brief overview of the key features:

🔹 Feature Descriptions:

📍 timestamp – The date and time each data entry was recorded.

📍  cryptocurrency – The name of the digital asset (e.g., Bitcoin, Ethereum).

📍  current_price_usd – The current market price of the cryptocurrency in U.S. dollars.

📍  price_change_24h_percent – Percentage change in price over the past 24 hours.

📍  trading_volume_24h – Total trading volume over the last 24 hours.

📍 market_cap_usd – The cryptocurrency’s total market capitalization in U.S. dollars.

📍 social_sentiment_score – A score reflecting public opinion from social media platforms.

📍 news_sentiment_score – A score based on the tone of news articles related to the cryptocurrency.

📍 news_impact_score – Measures the potential influence of news articles on the market.

📍 social_mentions_count – The number of times the cryptocurrency is mentioned on social platforms.

📍 fear_greed_index – An indicator showing whether the market is driven by fear or greed.

📍  volatility_index – Reflects the level of price fluctuations.

📍 rsi_technical_indicator – Relative Strength Index, a common technical analysis tool indicating overbought or oversold conditions.

📍 prediction_confidence (Target Variable) – Indicates how confident the forecasting model is in its prediction for the cryptocurrency.

## 🛠️ Libraries Used


To ensure efficient data analysis, model fitting, and evaluation, the following tools and libraries were used throughout the project:

Python – Main programming language for data manipulation and modeling.

Pandas – For data loading, wrangling, and transformation.

NumPy – For numerical operations and array handling.

Matplotlib & Seaborn – For visualizing data patterns, distributions, and relationships.

Scikit-learn (sklearn) – Used for preprocessing, regularization (Ridge Regression), scaling, model evaluation, and cross-validation.

Statsmodels – For checking regression assumptions such as multicollinearity, residual diagnostics, and error independence.

Jupyter Notebook – Interactive development environment used to run and document the analysis process.

🔍 Approach and Techniques

This project adopted a predictive modeling approach to explore how various market, technical, and sentiment-based indicators influence prediction confidence in the cryptocurrency space. The workflow combined statistical rigor with machine learning best practices to ensure reliability and interpretability.

✅ Step-by-Step Approach

Understanding the Problem
The target variable, prediction_confidence, represents the model’s certainty in forecasting cryptocurrency trends. The analysis focuses on identifying how features like price_change_24h_percent, social_sentiment_score, and fear_greed_index impact this prediction confidence.

Feature Engineering

Removed non-informative features such as timestamp.

Transformed the categorical variable cryptocurrency into numeric format using one-hot encoding.

Identified and addressed multicollinearity by examining Variance Inflation Factor (VIF).

Assumption Checks (for Ridge Regression)

✅ Independence of Errors: Verified using Durbin-Watson statistic.

✅ Normality of Residuals: Checked using residual plots and normality tests.

⚠️ Multicollinearity: Notably high VIF in cryptocurrency_Bitcoin (over 320) flagged serious collinearity issues.

Action: Removed the problematic variable to stabilize the model.

Modeling Approach

Selected Ridge Regression to handle potential multicollinearity while retaining all important predictors.

Scaled features using StandardScaler to ensure proper regularization.

Performed hyperparameter tuning via cross-validation to select the optimal penalty term (alpha).

## 📈📊EXPLORATORY DATA ANALYSIS

## 📊 Descriptive Statistics

| Feature                   | Count | Mean                 | Min                   | 25%                   | 50% (Median)          | 75%                   | Max                   | Std Dev            |
|---------------------------|-------|----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|--------------------|
| timestamp                | 2063  | 2025-06-19 23:28:01  | 2025-06-04 20:36:49    | 2025-06-12 08:04:43    | 2025-06-20 00:10:12    | 2025-06-27 03:59:34    | 2025-07-04 19:58:28    | —                  |
| current_price_usd        | 2063  | 4260.36              | 0.30                  | 1.21                  | 13.34                 | 84.97                 | 51610.92              | 12603.77           |
| price_change_24h_percent | 2063  | -0.018               | -25.56                | -5.76                 | 0.02                  | 5.76                  | 27.08                 | 8.00               |
| trading_volume_24h       | 2063  | 5,889,564.19         | 206,066.44            | 1,835,157.12          | 3,633,501.93          | 7,111,601.91          | 140,292,456.26        | 7,451,164.88       |
| market_cap_usd           | 2063  | 45.69T               | 50.22M                | 11.13B                | 125.59B               | 763.50B               | 1,009.56T             | 158.15T            |
| social_sentiment_score   | 2063  | 0.011                | -1.00                 | -0.20                 | 0.012                 | 0.215                 | 1.00                  | 0.303              |
| news_sentiment_score     | 2063  | 0.0024               | -1.00                 | -0.202                | 0.007                 | 0.212                 | 1.00                  | 0.309              |
| news_impact_score        | 2063  | 3.69                 | 0.07                  | 2.39                  | 3.67                  | 4.97                  | 9.53                  | 1.71               |
| social_mentions_count    | 2063  | 1218.32              | 2.0                   | 143.5                 | 409.0                 | 1178.0                | 35,578.0              | 2501.25            |
| fear_greed_index         | 2063  | 50.49                | 0.0                   | 41.4                  | 50.6                  | 59.4                  | 100.0                 | 13.37              |
| volatility_index         | 2063  | 76.44                | 21.4                  | 60.7                  | 79.0                  | 100.0                 | 100.0                 | 21.25              |
| rsi_technical_indicator  | 2063  | 50.50                | 1.4                   | 40.25                 | 50.5                  | 60.55                 | 97.1                  | 15.12              |
| prediction_confidence    | 2063  | 77.12                | 55.9                  | 72.65                 | 76.7                  | 81.1                  | 100.0                 | 6.65               |
| date                     | 2063  | 2025-06-19 23:28:01  | 2025-06-04 20:36:49    | 2025-06-12 08:04:43    | 2025-06-20 00:10:12    | 2025-06-27 03:59:34    | 2025-07-04 19:58:28    | —                  |

📊 Scatterplots of Key Relationship

Scatterplot was generated to examine the relationships between independent variables and the target variable prediction_confidence.

🔹 social_sentiment_score vs prediction_confidence
A moderate positive trend was observed — as social sentiment improves, model confidence in predicting prices tends to increase.

![Screenshot][Screenshot(44).png]


📦 Boxplot of Prediction Confidence by Cryptocurrency

A boxplot was used to visualize how prediction_confidence varies across different cryptocurrencies.

Bitcoin showed the highest prediction confidence on average, but also exhibited a wide range and higher variance, indicating greater uncertainty during certain time windows.

Ethereum, BNB, and Solana had more stable confidence levels, with narrower interquartile ranges.

Lower-cap or emerging coins like Dogecoin and Cardano displayed lower and more variable confidence, suggesting that predictions for these assets are less reliable — likely due to higher volatility or limited historical signals.

This analysis highlights that model confidence varies significantly across cryptocurrencies, reflecting differences in market behavior, sentiment responsiveness, and technical signal strength.

![Screenshot][Screenshot(45).png]

🎯 Distribution of Target Variable: prediction_confidence

The distribution of the target variable prediction_confidence appears approximately normal but slightly left-skewed.
Most values cluster between 70 and 85, with the peak around 76–77, indicating that in general, the model was highly confident in its cryptocurrency price predictions during the period analyzed (June–July 2025).

Min: 55.9

Max: 100.0

Mean: 77.12

Std Dev: 6.65

Distribution shape: Near-normal with slight negative skew

![Screenshot][Screenshot(43).png]

✅ Assumptions of Ridge Regression

Ridge Regression, being a linear model with L2 regularization, shares most assumptions of ordinary linear regression. Here are the key assumptions evaluated:

### ✅ Ridge Regression Assumptions & Results

| Assumption                | Description                                                               | Status        | Result                                                                 |
|---------------------------|---------------------------------------------------------------------------|---------------|------------------------------------------------------------------------|
| **Linearity**             | Relationship between predictors and the target should be linear.          | ✅ Met         | Scatterplots showed reasonably linear trends.                         |
| **Independence of Errors**| Residuals should not be autocorrelated.                                   | ✅ Met         | Durbin-Watson statistic ≈ **2.08** (no autocorrelation).               |
| **Normality of Residuals**| Residuals should be normally distributed.                                  | ⚠️ Partially Met | **p-value < 0.05** after variable removal, indicating mild skew.      |
| **Homoscedasticity**      | Residuals should have constant variance.                                  | ❌ Violated    | Residual plot showed pattern (heteroscedasticity).                    |
| **Low Multicollinearity** | Predictors should not be highly correlated.                                | ✅ Met (after fix) | Removed `cryptocurrency_Bitcoin` (VIF ≈ 320) to reduce collinearity.  |


## 📊 Ridge Regression Model Summary

| Metric                     | Value     |
|---------------------------|-----------|
| Best Alpha (λ)            | 1.0       |
| Mean Squared Error (MSE)  | 0.1334    |
| R² Score                  | 0.9967    |
| Durbin-Watson Statistic   | 2.08      |

## 🔎 Interpretation

High R² (0.9967): The model explains ~99.67% of the variance in prediction_confidence, indicating excellent fit after addressing multicollinearity.

Low MSE: Very small average error in predictions.

Durbin-Watson (2.08): Suggests no autocorrelation in residuals.

Note: Homoscedasticity was not satisfied, but ridge regression is robust to this violation.

## ✅ Recommendations

1. **Monitor Sentiment Trends**
    
   The model highlights the impact of sentiment scores (social and news) on prediction confidence. Stakeholders should incorporate sentiment tracking tools to make informed trading decisions.

3. **Watch for Volatility and RSI**
    
   Technical indicators like the Volatility Index and RSI strongly influence the model. Traders should use these as early warning signals to manage risk and optimize entry/exit strategies.

4. **Avoid Over-Reliance on Bitcoin Alone**
    
   The initial inclusion of `cryptocurrency_Bitcoin` created extreme multicollinearity and skewed the model. Diversifying analysis beyond a single dominant currency is essential for robust predictions.

5. **Use Ridge Regression in Real-Time Monitoring**
   
   Ridge regression provides high prediction accuracy when properly tuned. It is well-suited for real-time confidence scoring and risk assessment in crypto platforms.

6. **Refine Data with Consistent Feature Updates**
   
   Incorporate updated and clean data regularly (e.g., price, volume, sentiment) to keep the model accurate as crypto markets evolve rapidly.

   📞 Contact

   Created by Tijani Rofee'ah

🔗 [LinkedIn][https://www.linkedin.com/in/rofee-ah-tijani-713759253?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app ]
📫 For collaboration : Send a message on LinkedIn







