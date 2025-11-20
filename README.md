Advanced Time Series Forecasting with Deep Learning and Explainability (LSTM)
Project Overview
This project focuses on developing and optimizing a deep learning model for complex multivariate time series forecasting. It utilizes a programmatically generated synthetic dataset exhibiting non-stationarity, seasonality, and long-range dependencies. The core of the project involves implementing a Long Short-Term Memory (LSTM) network from near-scratch using TensorFlow/Keras for multi-step-ahead forecasting. A critical component is the integration of an Explainable AI (XAI) technique, Integrated Gradients, to interpret feature importance across the input sequence window. The project demonstrates robust validation procedures and a thorough analysis of the model's predictive performance versus its interpretability findings.
Table of Contents
1.  [Dataset Generation & Preprocessing](#dataset-generation--preprocessing)
2.  [Deep Learning Model (LSTM) Implementation](#deep-learning-model-lstm-implementation)
3.  [Model Training & Optimization](#model-training--optimization)
4.  [Model Evaluation](#model-evaluation)
5.  [Explainable AI (XAI) Analysis](#explainable-ai-xai-analysis)
6.  [Summary of Findings & Next Steps](#summary-of-findings--next-steps)

1. Dataset Generation & Preprocessing
 Data Characteristics
*   Type: Synthetic Multivariate Time Series
*   Features: 5 primary features (`Feature_1` to `Feature_5`) plus 5 engineered temporal features (`hour`, `day_of_week`, `day_of_year`, `month`, `year`).
*   Data Points: 3500 hourly observations.
*   **Temporal Patterns**: Includes linear, non-linear, and exponential trends, daily, weekly, and monthly seasonality, autoregressive components, noise, and inter-feature correlation.
Preprocessing Steps
1. Missing Value Check: Confirmed no missing values in the data.
2. Feature Engineering: Extracted `hour`, `day_of_week`, `day_of_year`, `month`, `year` from the datetime index.
3. Scaling: All 10 numerical features were scaled using `MinMaxScaler`.
4. Sequence Creation: Data was transformed into input-output sequences for the LSTM model:
    *   `look_back`: 24 timesteps (e.g., past 24 hours).
    *   `forecast_horizon`: 12 timesteps (e.g., next 12 hours).
    *   Input sequences (`X`): (Number of samples, 24, 10)
    *   Target sequences (`y`): (Number of samples, 12, 10)
5. Data Split: Chronological split into Training (70%), Validation (15%), and Test (15%) sets.

2. Deep Learning Model (LSTM) Implementation

An LSTM-based sequence-to-sequence model was implemented using TensorFlow/Keras:

* Architecture:
    *   `LSTM(units=100, return_sequences=False, input_shape=(look_back, num_features))`: Processes the input sequence and outputs a single vector (last hidden state).
    *   `RepeatVector(forecast_horizon)`: Repeats the output vector from the first LSTM layer `forecast_horizon` times to match the desired output sequence length.
    *   `LSTM(units=100, return_sequences=True)`: Processes the repeated vector and outputs a sequence.
    *   `TimeDistributed(Dense(units=num_features))`: Applies a Dense layer to each timestep of the output sequence to predict all features for each forecast step.
*   Total Trainable Parameters: 125,810
*   Optimizer: `adam`
*   Loss Function: `mean_squared_error`

3. Model Training & Optimization

*   Epochs: 50
*   Batch Size: 32
*   Early Stopping: Implemented with `patience=10` to stop training if `val_loss` does not improve, restoring the best model weights.
*   Model Checkpoint: Saved the best model weights to `best_model.keras` based on `val_loss`.
*   Training Duration: The model trained for 31 epochs before Early Stopping was triggered, indicating convergence.

4. Model Evaluation

The trained model was evaluated on the held-out test set, and predictions were inverse-transformed to the original scale. The following metrics were calculated:

* Overall Average RMSE: `18.3755`
* Overall Average MAE: `16.7857`

Performance per Feature:

| Feature        | RMSE      | MAE       |
| :------------- | :-------- | :-------- |
| Feature_1      | 4.2496    | 3.3971    |
| Feature_2      | 7.4542    | 5.8837    |
| Feature_3      | 7.7794    | 6.2111    |
| Feature_4      | 152.5184  | 143.6044  |
| Feature_5      | 5.0658    | 4.0054    |
| hour           | 1.1509    | 0.6586    |
| day_of_week    | 0.9538    | 0.3955    |
| day_of_year    | 4.3623    | 3.5263    |
| month          | 0.2045    | 0.1624    |
| year           | 0.0160    | 0.0126    |

Key Observation: `Feature_4` exhibited significantly higher errors compared to other features, suggesting the model struggled to capture its complex exponential trend and short-period seasonality. Other features, especially the engineered temporal ones, showed good predictive performance.

5. Explainable AI (XAI) Analysis

Technique: Integrated Gradients was applied to understand the contributions of input features and timesteps to the model's prediction for `Feature_1` at the first forecast timestep (H+1).

Key XAI Findings:
*   Recency Bias: Consistent across samples, the most recent input timesteps (last 5-10 hours) showed the highest attribution scores (both positive and negative), indicating the model heavily relies on immediate past observations.
*   Feature-Specific Influence: `Feature_1` itself and `Feature_5` (which was designed to be correlated with `Feature_1`) were identified as highly influential. Temporal features like `hour` and `day_of_year` also showed significant, albeit smaller, contributions, confirming the model's use of seasonality.
*   Least Influential: The 'year' feature consistently had very low attribution, as expected for short-term hourly forecasts from a limited dataset span.

XAI Interpretation Snippets (for `Feature_1` at Forecast Timestep 1):

*   Forecast Window 1 (Sample 1):
    *   Interpretation: Prediction driven mainly by `Feature_1` values from the last 5-7 hours (positive attributions). `Feature_5` also contributed positively. Moderate negative attribution from 'hour' suggested a time-of-day adjustment.
    *   Attribution Magnitude Example (Last Hour, for Feature_1): ~0.002 (positive)

*   Forecast Window 2 (Sample 2):
    *   Interpretation: `Feature_1`'s prediction heavily depended on its values within the last 10 hours, with mixed positive/negative influences. `Feature_2` showed small but consistent positive attribution. 'day_of_year' contributed positively across the input window, indicating a seasonal adjustment.
    *   Attribution Magnitude Example (Last Hour, for Feature_5): ~0.0015 (positive)

*   Forecast Window 3 (Sample 3):
    *   Interpretation: Strongest impact from `Feature_1` values in the last 3-5 hours. A notable negative attribution from `Feature_4` in the last hour pushed the `Feature_1` prediction downwards. `Feature_3` from the mid-look-back window showed minor positive influence.
    *   Attribution Magnitude Example (Last Hour, for Feature_4): ~-0.005 (negative)

6. Summary of Findings & Next Steps

The LSTM model effectively forecasted most multivariate time series features, leveraging engineered temporal features to capture cyclical patterns. However, `Feature_4` proved challenging. The Integrated Gradients analysis validated the model's reliance on recent data and key correlated features, confirming that it learned meaningful temporal relationships.

 Future Work / Next Steps:
*   Improve Feature_4 Prediction: Investigate alternative models or specialized techniques for forecasting `Feature_4`, given its high error rates. This could involve exploring different architectures or advanced feature engineering specific to its complex patterns.
*   Refine Model with XAI: Utilize the XAI insights to further refine the model, potentially by focusing on the most influential features or experimenting with different weighting schemes for temporal importance within the input sequences.
*   Hyperparameter Tuning: Conduct more extensive hyperparameter tuning (e.g., using techniques like Grid Search or Bayesian Optimization) to potentially improve overall model performance.

How to Run the Project
The entire project is contained within a Google Colab notebook. Simply execute the cells sequentially from top to bottom. The notebook includes all necessary code for data generation, preprocessing, model definition, training, evaluation, and XAI analysis, along with visualizations and a detailed report.
