# CS:GO Player K/D Ratio Predictor

A machine learning model that predicts a CS:GO player's K/D (Kill/Death) ratio based on various features including the player, map, teammates, and team ratings.

## Overview

This project uses a Random Forest Regressor to predict player performance in CS:GO professional matches. The model takes into account:
- Player identity
- Map being played
- Teammates
- Team average rating
- Opponent average rating

## Files

- `csgo_player_predictor.py` - Main model training and prediction code
- `csgo_predictor_interface.py` - Interactive interface for making predictions
- `create_visualizations.py` - Script to generate data visualizations
- `csgo_pro_games_data.csv` - Dataset of professional CS:GO matches
- `*.joblib` - Trained model and encoder files
- `*.png` - Generated visualization plots

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:
```bash
python csgo_player_predictor.py
```

This will:
- Load and preprocess the data
- Train a Random Forest Regressor
- Evaluate model performance
- Save the trained model and encoders

### Making Predictions

To use the interactive interface:
```bash
python csgo_predictor_interface.py
```

Or use the prediction function directly in Python:
```python
from csgo_player_predictor import predict_kd_ratio

predicted_kd = predict_kd_ratio(
    player_name="s1mple",
    map_name="d2",
    teammates=["electronic", "B1t", "Perfecto", "Boombl4"],
    team_avg_rating=1.1,
    opponent_avg_rating=1.0
)
print(f"Predicted K/D: {predicted_kd:.2f}")
```

### Creating Visualizations

To generate data visualizations:
```bash
python create_visualizations.py
```

This creates several plots:
- K/D ratio distribution
- K/D ratio by map (boxplot)
- Team rating vs K/D ratio (scatter plot)
- Correlation heatmap

## Model Performance

The model uses a Random Forest Regressor with the following features:
- Player encoding
- Map encoding
- Team average rating
- Opponent average rating
- Teammate average rating

Performance metrics are displayed during training, including:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Accuracy within ±0.1 K/D ratio

## Data

The dataset contains professional CS:GO match data with player statistics, map information, and team ratings.

## Requirements

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- joblib >= 1.0.0
- matplotlib (for visualizations)
- seaborn >= 0.13.0 (for visualizations)
- tqdm >= 4.65.0

