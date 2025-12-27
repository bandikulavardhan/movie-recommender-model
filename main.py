import pandas as pd
import numpy as np
from src.preprocessor import load_and_clean_data
from src.vectorizer import ManualTFIDF
from src.similarity import calculate_cosine_similarity
from src.recommender import get_recommendations
from src.evaluate import evaluate_precision
import time

def main():
    print("--- Starting Movie Recommendation System ---")
    
    # 1. Load Data
    print("[1/5] Loading and cleaning data...")
    # Make sure this matches your CSV name in the data folder
    movies_path = 'data/tmdb_5000_movies.csv' 
    try:
        df = load_and_clean_data(movies_path)
        print(f"      Successfully loaded {len(df)} movies.")
    except FileNotFoundError:
        print(f"Error: Could not find file at {movies_path}. Please check Step 1.")
        return

    # 2. Vectorization (TF-IDF)
    print("[2/5] Building TF-IDF Matrix (This might take a moment)...")
    start_time = time.time()
    tfidf = ManualTFIDF()
    # We use the 'clean_overview' column we created in preprocessing
    tfidf_matrix = tfidf.fit_transform(df['clean_overview'].tolist())
    print(f"      Matrix shape: {tfidf_matrix.shape}")
    print(f"      Time taken: {time.time() - start_time:.2f} seconds")

    # 3. Similarity Calculation
    print("[3/5] Calculating Cosine Similarity...")
    similarity_matrix = calculate_cosine_similarity(tfidf_matrix)
    print("      Similarity matrix built.")

    # 4. Evaluation
    print("[4/5] Evaluating model performance...")
    # We pass the original dataframe and our similarity matrix
    precision = evaluate_precision(df, similarity_matrix)
    print(f"      Precision@5: {precision:.2%} (Based on Genre matching)")

    # 5. Interactive Recommendation
    print("\n[5/5] System Ready! 🎬")
    while True:
        movie_name = input("\nEnter a movie name (or 'exit' to quit): ").strip()
        if movie_name.lower() == 'exit':
            break
        
        recommendations = get_recommendations(movie_name, df, similarity_matrix)
        
        if isinstance(recommendations, str):
            print(f"      Error: {recommendations}") # Movie not found
        else:
            print(f"      Top 5 Recommendations for '{movie_name}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec}")

if __name__ == "__main__":
    main()