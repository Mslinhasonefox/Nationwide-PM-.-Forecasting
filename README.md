# Nationwide PM₂.₅ Forecasting System

This project develops a national-scale PM₂.₅ forecasting pipeline using historical station-based air quality data and multi-source meteorological inputs.

## Model Components

- **GTNNWR**: A spatial-temporal neural network that learns dynamic weights across locations and time steps.
- **Multi-scale CNN-LSTM**: Captures local spatial patterns and sequential dependencies using convolutional and recurrent layers.
- **STTransformer**: A Transformer-based model that captures long-range temporal structure and periodicity in environmental data.

All models are benchmarked under identical data splits and compared on RMSE, MAE, and R².

## Features

- Data preprocessing and interpolation for 1,200+ stations (2002–2022)
- Fully modular architecture with plug-and-play models
- SHAP-based interpretability for variable attribution

## Repository Structure
