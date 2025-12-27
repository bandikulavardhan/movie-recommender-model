def evaluate_precision(df, similarity_matrix):
    correct_matches = 0
    total_checks = 0
    
    # Test on a sample of 100 movies
    sample_indices = df.sample(100).index
    
    for idx in sample_indices:
        # Get the input movie's genres
        input_genres = set(df.iloc[idx]['genres'].split())
        
        # Get top 5 recommendations
        scores = similarity_matrix[idx]
        top_indices = scores.argsort()[::-1][1:6]
        
        # Check if recommendations share a genre
        for rec_idx in top_indices:
            rec_genres = set(df.iloc[rec_idx]['genres'].split())
            
            # If they share at least one genre, it's a "hit"
            if not input_genres.isdisjoint(rec_genres):
                correct_matches += 1
            total_checks += 1
            
    precision = correct_matches / total_checks
    return precision

# If precision is 0.85, that is 85% accuracy based on Genre Matching.