# Gold Price Prediction

This notebook aims to predict monthly gold prices using various time series forecasting models.

## Task
The primary goal is to predict future monthly gold prices based on historical data. This is framed as a regression task.

## Dataset
- **Source**: [Monthly Gold Prices from datahub.io](https://datahub.io/core/gold-prices#readme)
- **Description**: Contains monthly gold prices in USD, originally sourced from the World Gold Council and supplemented by the World Bank. The dataset starts from 1833.
- **File**: `monthly.csv`

## Evaluation Metrics
Given that this is a regression problem, the following metrics are used to evaluate model performance:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Data Preprocessing
1.  **Download Data**: The `monthly.csv` dataset and a custom `helper.py` script are downloaded.
2.  **Load and Initial Analysis**: The dataset is loaded into a pandas DataFrame, with the 'Date' column parsed and set as the index.
3.  **Data Filtering**: Early data points (before September 1949) where prices were consistently low and less dynamic are dropped to focus on more relevant historical trends.
4.  **Train/Test Split**: The data is split into training (85%) and testing (15%) sets, ensuring that newer data is reserved for testing.
5.  **Window and Horizon Creation**: A utility function `make_window_horizon` is defined to transform the time series data into input windows and target horizons, suitable for supervised learning models.

## Models Implemented

### 1. Naive Forecast (Baseline)
-   **Description**: A simple baseline model where the prediction for the next timestep is the value of the current timestep.
-   **Result**: Served as a strong baseline, achieving relatively low error metrics.

### 2. Dense Model
-   **Architecture**: A simple feed-forward neural network with `Dense` layers.
-   **Model 1 (Window=7, Horizon=1)**:
    -   **Input**: `(7, 1)` (7 previous timesteps, 1 feature).
    -   **Layers**: `Dense(64, activation="linear")`, `Dense(1, activation="linear")`.
    -   **Hyperparameters**: Adam optimizer (learning_rate=5e-5), MAE loss, EarlyStopping, ModelCheckpoint.
-   **Model 3 (Window=10, Horizon=1)**:
    -   **Input**: `(10, 1)`.
    -   **Layers**: `Dense(128, activation="linear")`, `Dense(1, activation="linear")`.
    -   **Hyperparameters**: Similar to Model 1.

### 3. RNN Model (GRU)
-   **Architecture**: A Recurrent Neural Network using a Gated Recurrent Unit (GRU) layer.
-   **Model 2 (Window=7, Horizon=1)**:
    -   **Input**: `(7, 1)`.
    -   **Layers**: `GRU(128)`, `Dense(1, activation="linear")`.
    -   **Hyperparameters**: Adam optimizer (learning_rate=5e-5), MAE loss, EarlyStopping, ModelCheckpoint.

### 4. Sktime Models (Exploration)
-   **Description**: Various forecasting models from the `sktime` library were explored, including `TrendForecaster`, `ExponentialSmoothing`, `NaiveForecaster`, `ETS`, and `ARIMA`.
-   **Note**: Initial attempts with `sktime` models did not yield successful predictions, indicating potential issues with data preparation or model configuration within the `sktime` framework for this specific problem.

## Results Summary
-   The **Naive Forecast** provided a strong baseline, proving difficult to beat for the initial deep learning models.
-   The **Dense Models (Model 1 & 3)** showed some improvement over the raw errors but still struggled to outperform the simple baseline, indicating potential limitations in capturing complex time series dependencies with dense layers alone or insufficient hyperparameter tuning.
-   The **RNN Model (Model 2)** performed significantly worse than the baseline, suggesting issues with its training process or architecture for this dataset.
-   The **Sktime Models** were not successfully implemented or evaluated due to configuration challenges.

Further work would involve more extensive hyperparameter tuning, exploring more complex RNN architectures (e.g., LSTMs, Transformers), data scaling, and a deeper dive into `sktime` for robust traditional time series models.
