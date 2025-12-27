import numpy as np
import pandas as pd
from collections import Counter
import math

class ManualTFIDF:
    def __init__(self):
        self.vocabulary = {} # Stores word -> index
        self.idf_scores = {} # Stores word -> score
        
    def fit_transform(self, text_list):
        # 1. Build Vocabulary (List of all unique words)
        all_words = set()
        for text in text_list:
            words = text.split()
            all_words.update(words)
        
        # Create a map: word -> index (e.g., 'action': 0, 'comedy': 1)
        self.vocabulary = {word: i for i, word in enumerate(sorted(list(all_words)))}
        vocab_size = len(self.vocabulary)
        n_docs = len(text_list)
        
        # 2. Calculate IDF
        # Count how many documents contain each word
        doc_counts = Counter()
        for text in text_list:
            unique_words_in_doc = set(text.split())
            for word in unique_words_in_doc:
                doc_counts[word] += 1
                
        # Calculate IDF formula: log(Total Docs / Docs with word)
        for word, count in doc_counts.items():
            self.idf_scores[word] = math.log(n_docs / (1 + count))
            
        # 3. Calculate TF-IDF Matrix
        # Create a matrix of zeros: (Number of Movies) x (Number of Words)
        tfidf_matrix = np.zeros((n_docs, vocab_size))
        
        for doc_idx, text in enumerate(text_list):
            words = text.split()
            word_counts = Counter(words)
            total_words = len(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    # TF = count / total words
                    tf = count / total_words
                    # IDF = from our calculation above
                    idf = self.idf_scores[word]
                    
                    word_idx = self.vocabulary[word]
                    tfidf_matrix[doc_idx, word_idx] = tf * idf
                    
        return tfidf_matrix