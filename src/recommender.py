import pandas as pd

def get_recommendations(movie_title, df, similarity_matrix, top_k=5):

    # Convert input to lowercase for searching
    query = movie_title.lower().strip()

    # Create a temporary column for lowercase searching
    titles_lower = df['title'].str.lower()

    # STRATEGY 1: Exact Match
    if query in titles_lower.values:
        movie_idx = titles_lower[titles_lower == query].index[0]

    # STRATEGY 2: Partial Match (The NEW Logic)
    else:
        # Find all titles that contain the query string
        mask = titles_lower.str.contains(query, regex=False)
        matches = df[mask]['title'].tolist()

        if len(matches) == 0:
            return f"Movie '{movie_title}' not found in the database. Try checking the spelling."
        
        if len(matches) > 1:
            # If we found multiple matches, return them to the user
            # Limit to top 5 to avoid flooding the screen
            suggestions = matches[:5] 
            return f"Did you mean one of these?: {', '.join(suggestions)}"
    
        # If exactly one match found, use it automatically
        movie_idx = df[mask].index[0]
        # Let the user know we auto-corrected them
        print(f"      (Auto-selecting: '{df.iloc[movie_idx]['title']}')")        
    
    # 2. Get similarity scores for this movie
    scores = similarity_matrix[movie_idx]
    
    # 3. Sort scores (highest first) and get indices
    # argsort sorts low-to-high, so we reverse it [::-1]
    sorted_indices = scores.argsort()[::-1]
    
    # 4. Get top K (skip index 0 because that is the movie itself)
    top_indices = sorted_indices[1 : top_k+1]
    
    # 5. Return titles
    return df['title'].iloc[top_indices].tolist()