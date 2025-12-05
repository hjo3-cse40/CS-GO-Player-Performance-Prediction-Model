import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from tqdm import tqdm

def load_and_preprocess_data():
    print("Loading data...")
    # Load the data
    df = pd.read_csv('csgo_pro_games_data.csv')
    
    print("Processing player data...")
    # Create a list to store our processed data
    processed_data = []
    
    # Process each row to create player-map combinations
    for _, row in tqdm(df.iterrows(), total=len(df)):
        map_name = row['map_name_short']
        if pd.isna(map_name):  # Skip if map name is missing
            continue
            
        # Process team 1 players
        for i in range(1, 6):
            player_name = row[f'team1_p{i}_name']
            if pd.isna(player_name):  # Skip if player name is missing
                continue
                
            kills = row[f'team1_p{i}_kills']
            deaths = row[f'team1_p{i}_deaths']
            
            if pd.isna(kills) or pd.isna(deaths) or deaths == 0:
                continue
                
            kd_ratio = kills / deaths
            
            # Get teammate names
            teammates = [row[f'team1_p{j}_name'] for j in range(1, 6) 
                        if j != i and not pd.isna(row[f'team1_p{j}_name'])]
            
            if len(teammates) < 4:  # Skip if we don't have enough teammates
                continue
            
            processed_data.append({
                'player': player_name,
                'map': map_name,
                'teammates': teammates[:4],  # Only take first 4 teammates
                'kd_ratio': kd_ratio,
                'team_avg_rating': row['team1_game_rating'],
                'opponent_avg_rating': row['team2_game_rating']
            })
        
        # Process team 2 players
        for i in range(1, 6):
            player_name = row[f'team2_p{i}_name']
            if pd.isna(player_name):  # Skip if player name is missing
                continue
                
            kills = row[f'team2_p{i}_kills']
            deaths = row[f'team2_p{i}_deaths']
            
            if pd.isna(kills) or pd.isna(deaths) or deaths == 0:
                continue
                
            kd_ratio = kills / deaths
            
            # Get teammate names
            teammates = [row[f'team2_p{j}_name'] for j in range(1, 6) 
                        if j != i and not pd.isna(row[f'team2_p{j}_name'])]
            
            if len(teammates) < 4:  # Skip if we don't have enough teammates
                continue
            
            processed_data.append({
                'player': player_name,
                'map': map_name,
                'teammates': teammates[:4],  # Only take first 4 teammates
                'kd_ratio': kd_ratio,
                'team_avg_rating': row['team2_game_rating'],
                'opponent_avg_rating': row['team1_game_rating']
            })
    
    return pd.DataFrame(processed_data)

def prepare_features(df):
    print("Preparing features...")
    # Create encoders for categorical variables
    player_encoder = LabelEncoder()
    map_encoder = LabelEncoder()
    
    # Fit encoders
    print("Encoding players and maps...")
    # Get unique players from both player and teammate columns
    all_players = set(df['player'].unique())
    for teammates in df['teammates']:
        all_players.update(teammates)
    all_players = list(all_players)
    
    # Get unique maps and ensure they're in the correct format
    unique_maps = df['map'].unique()
    map_encoder.fit(unique_maps)
    
    # Save the valid maps for later use
    valid_maps = set(unique_maps)
    joblib.dump(valid_maps, 'valid_maps.joblib')
    
    player_encoder.fit(all_players)
    
    # Transform features
    print("Transforming features...")
    X = pd.DataFrame({
        'player_encoded': player_encoder.transform(df['player']),
        'map_encoded': map_encoder.transform(df['map']),
        'team_avg_rating': df['team_avg_rating'],
        'opponent_avg_rating': df['opponent_avg_rating']
    })
    
    # Add teammate features (average rating of teammates)
    print("Processing teammate features...")
    # Create a dictionary for faster lookups
    player_encoding_dict = {player: idx for idx, player in enumerate(all_players)}
    
    # Vectorized teammate processing
    teammate_ratings = []
    for teammates in tqdm(df['teammates']):
        teammate_encodings = [player_encoding_dict[t] for t in teammates]
        teammate_ratings.append(np.mean(teammate_encodings))
    
    X['teammate_avg_rating'] = teammate_ratings
    
    return X, df['kd_ratio'], player_encoder, map_encoder

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using multiple metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage of predictions within 0.1 K/D ratio
    within_threshold = np.abs(y_test - y_pred) <= 0.1
    accuracy_within_threshold = np.mean(within_threshold) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Accuracy within ±0.1 K/D': accuracy_within_threshold
    }

def train_model():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features
    X, y, player_encoder, map_encoder = prepare_features(df)
    
    print("Splitting data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Train model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=50,  # Increase minimum samples for split
        min_samples_leaf=25,  # Increase minimum samples per leaf
        max_features='sqrt'  # Use sqrt of features for each split
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    performance_metrics = evaluate_model(model, X_test, y_test)
    
    # Print performance metrics
    print("\nModel Performance Metrics:")
    print("-------------------------")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Perform cross-validation with fewer folds and optimized parameters
    print("\nPerforming 3-fold cross-validation...")
    cv_scores = cross_val_score(
        model, 
        X, 
        y, 
        cv=3,  # Reduced from 5 to 3 folds
        scoring='neg_mean_squared_error',
        n_jobs=-1  # Enable parallel processing
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE scores: {cv_rmse}")
    print(f"Mean CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': ['player_encoded', 'map_encoded', 'team_avg_rating', 
                   'opponent_avg_rating', 'teammate_avg_rating'],
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print("------------------")
    print(feature_importance)
    
    print("\nSaving model and encoders...")
    # Save model and encoders
    joblib.dump(model, 'csgo_kd_predictor.joblib')
    joblib.dump(player_encoder, 'player_encoder.joblib')
    joblib.dump(map_encoder, 'map_encoder.joblib')
    
    return model, player_encoder, map_encoder, performance_metrics, feature_importance

def predict_kd_ratio(player_name, map_name, teammates, team_avg_rating, opponent_avg_rating):
    try:
        # Load model and encoders
        model = joblib.load('csgo_kd_predictor.joblib')
        player_encoder = joblib.load('player_encoder.joblib')
        map_encoder = joblib.load('map_encoder.joblib')
        valid_maps = joblib.load('valid_maps.joblib')
        
        # Validate map name
        if map_name not in valid_maps:
            raise ValueError(f"Invalid map name: {map_name}. Valid maps are: {', '.join(valid_maps)}")
        
        # Prepare input features
        X = pd.DataFrame({
            'player_encoded': [player_encoder.transform([player_name])[0]],
            'map_encoded': [map_encoder.transform([map_name])[0]],
            'team_avg_rating': [team_avg_rating],
            'opponent_avg_rating': [opponent_avg_rating],
            'teammate_avg_rating': [np.mean([player_encoder.transform([t])[0] for t in teammates])]
        })
        
        # Make prediction
        predicted_kd = model.predict(X)[0]
        return predicted_kd
        
    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    # Train the model
    model, player_encoder, map_encoder, performance_metrics, feature_importance = train_model()
    
    # Example usage
    player_name = "s1mple"  # Example player
    map_name = "de_dust2"   # Example map
    teammates = ["electronic", "B1t", "Perfecto", "Boombl4"]  # Example teammates
    team_avg_rating = 1.1    # Example team rating
    opponent_avg_rating = 1.0  # Example opponent rating
    
    predicted_kd = predict_kd_ratio(player_name, map_name, teammates, team_avg_rating, opponent_avg_rating)
    print(f"Predicted K/D ratio for {player_name} on {map_name}: {predicted_kd:.2f}") 