import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from csgo_player_predictor import load_and_preprocess_data

# Set style for better-looking plots
plt.style.use('default')
sns.set_theme()

def create_kd_distribution_plot(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='kd_ratio', bins=50)
    plt.title('Distribution of K/D Ratios Across All Players')
    plt.xlabel('K/D Ratio')
    plt.ylabel('Frequency')
    plt.savefig('kd_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_map_kd_boxplot(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='map', y='kd_ratio')
    plt.title('K/D Ratio Distribution by Map')
    plt.xlabel('Map')
    plt.ylabel('K/D Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('map_kd_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_team_rating_scatter(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='team_avg_rating', y='kd_ratio', alpha=0.5)
    plt.title('Team Rating vs K/D Ratio')
    plt.xlabel('Team Average Rating')
    plt.ylabel('K/D Ratio')
    plt.savefig('team_rating_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(df):
    # Select numeric columns for correlation
    numeric_cols = ['kd_ratio', 'team_avg_rating', 'opponent_avg_rating']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data()
        
        # Create visualizations
        print("Creating K/D ratio distribution plot...")
        create_kd_distribution_plot(df)
        
        print("Creating map K/D boxplot...")
        create_map_kd_boxplot(df)
        
        print("Creating team rating scatter plot...")
        create_team_rating_scatter(df)
        
        print("Creating correlation heatmap...")
        create_correlation_heatmap(df)
        
        print("All visualizations have been created!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 