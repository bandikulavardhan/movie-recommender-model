import numpy as np

def calculate_cosine_similarity(matrix):
    # 1. Dot Product (A . B)
    # We multiply the matrix by its transpose
    dot_product = np.dot(matrix, matrix.T)
    
    # 2. Magnitude (Norm)
    # Calculate length of each vector
    norm = np.linalg.norm(matrix, axis=1)

    norm[norm == 0] = 1e-10
    
    # 3. Divide
    # We need to reshape norm to allow division
    # (Matrix / Norm) / Norm_Transpose
    similarity_matrix = dot_product / np.outer(norm, norm)
    
    return similarity_matrix