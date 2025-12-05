from csgo_player_predictor import train_model, predict_kd_ratio
import joblib

def get_user_input():
    print("\nCS:GO Player K/D Ratio Predictor")
    print("=================================")
    
    # Get player name
    player_name = input("\nEnter the player name to predict: ")
    
    # Get map name
    print("\nAvailable maps:")
    print("1. Dust2 (d2)")
    print("2. Mirage (mrg)")
    print("3. Inferno (inf)")
    print("4. Nuke (nuke)")
    print("5. Overpass (ovp)")
    print("6. Ancient (anc)")
    print("7. Vertigo (vtg)")
    
    map_choice = input("\nEnter the map number (1-7): ")
    map_names = {
        "1": "d2",
        "2": "mrg",
        "3": "inf",
        "4": "nuke",
        "5": "ovp",
        "6": "anc",
        "7": "vtg"
    }
    map_name = map_names.get(map_choice, "d2")
    
    # Get teammate names
    print("\nEnter the names of the 4 teammates (one per line):")
    teammates = []
    for i in range(4):
        teammate = input(f"Teammate {i+1}: ")
        teammates.append(teammate)
    
    # Get team ratings
    while True:
        try:
            team_avg_rating = float(input("\nEnter team's average rating (0.0-2.0): "))
            if 0 <= team_avg_rating <= 2:
                break
            print("Rating must be between 0.0 and 2.0")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            opponent_avg_rating = float(input("Enter opponent's average rating (0.0-2.0): "))
            if 0 <= opponent_avg_rating <= 2:
                break
            print("Rating must be between 0.0 and 2.0")
        except ValueError:
            print("Please enter a valid number")
    
    return player_name, map_name, teammates, team_avg_rating, opponent_avg_rating

def run_prediction(player_name, map_name, teammates, team_avg_rating, opponent_avg_rating):
    try:
        predicted_kd = predict_kd_ratio(
            player_name, 
            map_name, 
            teammates, 
            team_avg_rating, 
            opponent_avg_rating
        )
        
        print("\nPrediction Result:")
        print(f"Player: {player_name}")
        print(f"Map: {map_name}")
        print(f"Teammates: {', '.join(teammates)}")
        print(f"Team Rating: {team_avg_rating}")
        print(f"Opponent Rating: {opponent_avg_rating}")
        print(f"Predicted K/D Ratio: {predicted_kd:.2f}")
        return predicted_kd
    except Exception as e:
        print(f"\nError predicting for {player_name}: {str(e)}")
        return None

def main():
    print("Training the model... This might take a few minutes...")
    train_model()
    
    # Define our predictions
    predictions = [
        # Rain with FaZe
        ("rain", "mrg", ["karrigan", "broky", "Twistzz", "ropz"], 1.15, 1.1),
        # Rain with different teammates
        ("rain", "mrg", ["karrigan", "broky", "Twistzz", "NiKo"], 1.15, 1.1),
        # Ropz with FaZe
        ("ropz", "inf", ["karrigan", "broky", "Twistzz", "rain"], 1.15, 1.1),
        # Stewie2k with Liquid
        ("Stewie2k", "nuke", ["NAF", "EliGE", "nitr0", "Twistzz"], 1.1, 1.05),
        # Stewie2k with Cloud9
        ("Stewie2k", "nuke", ["Skadoodle", "autimatic", "RUSH", "tarik"], 1.05, 1.0)
    ]
    
    print("\nRunning predictions for multiple players...")
    print("==========================================")
    
    results = []
    for pred in predictions:
        player_name, map_name, teammates, team_rating, opp_rating = pred
        print(f"\nPredicting for {player_name}...")
        kd = run_prediction(player_name, map_name, teammates, team_rating, opp_rating)
        if kd is not None:
            results.append((player_name, map_name, kd))
    
    print("\nSummary of Predictions:")
    print("======================")
    for player, map_name, kd in results:
        print(f"{player} on {map_name}: {kd:.2f} K/D")

if __name__ == "__main__":
    main() 