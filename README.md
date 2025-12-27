# Custom ML Movie Recommender (Built from Scratch)

A Content-Based Movie Recommendation System implemented using only **Python, NumPy, and Pandas**. This project avoids high-level libraries like scikit-learn to demonstrate a deep understanding of Recommendation Engine mathematics.

## 🚀 Key Features
* **Manual TF-IDF Vectorizer:** Custom implementation of Term Frequency and Inverse Document Frequency.
* **Linear Algebra Similarity:** Manual calculation of Cosine Similarity using NumPy dot products and norms.
* **Smart Search:** Flexible search logic that handles case sensitivity and partial title matches.
* **Performance Metrics:** Built-in evaluation script calculating Precision@5 based on genre relevance.

## 📊 The Math Behind the Project

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
Used to convert movie descriptions into numerical vectors. 
- **TF:** Measures how often a word appears in a description.
- **IDF:** Reduces the weight of common words (like "the") and increases the weight of unique keywords.

### 2. Cosine Similarity
Calculates the "distance" between two movie vectors using the formula:
$$Similarity = \frac{A \cdot B}{||A|| ||B||}$$

## 🛠️ Technology Stack
- **Language:** Python 3.x
- **Data Handling:** Pandas
- **Mathematics:** NumPy
- **Version Control:** Git

## 📈 Evaluation Results
- **Metric:** Precision@5 (Genre-based relevance)
- **Result:** ~98% 
*Note: A recommendation is considered "relevant" if it shares at least one genre with the input movie.*

## ⚙️ Setup & Usage
1. Clone the repository.
2. Ensure you have the dataset `tmdb_5000_movies.csv` in the `data/` folder.
3. Install requirements: `pip install -r requirements.txt`.
4. Run the system: `python main.py`.
