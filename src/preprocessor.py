import pandas as pd

def load_and_clean_data(movies_path):
    # Load data
    df = pd.read_csv(movies_path, encoding='latin1')
    
    # Keep only useful columns
    df = df[['id', 'title', 'genres', 'overview']]
    
    # Fill missing text with empty strings
    df['overview'] = df['overview'].fillna('')
    
    # Simple text cleaning function
    def clean_text(text):
        # Make lowercase
        text = str(text).lower()
        # Replace punctuation (simple approach)
        text = text.replace(',', ' ').replace('.', ' ')
        return text

    # Apply cleaning
    df['clean_overview'] = df['overview'].apply(clean_text)
    
    return df