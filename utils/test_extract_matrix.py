#import numpy as np

def extract_3x3(matrix, center_row, center_col):
    rows = len(matrix)
    cols = len(matrix[0])
    submatrix = []
    for i in range(center_row - 1, center_row + 2):
        row = []
        for j in range(center_col - 1, center_col + 2):
            if 0 <= i < rows and 0 <= j < cols:
                row.append(matrix[i][j])
            else:
                row.append(3)  # Out of bounds
        submatrix.append(row)
    return submatrix

m = [
    [0, 3, 0, 0, 2],
    [0, 3, 0, 3, 0],    
    [0, 0, 0, 3, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]]

print(extract_3x3(m, 0, 0))  # Top-left corner
print(extract_3x3(m, 0, 4))  # Top-right corner
print(extract_3x3(m, 4, 0))  # Bottom-left corner
print(extract_3x3(m, 4, 4))  # Bottom-right corner
print(extract_3x3(m, 2, 2))  # Center
print(extract_3x3(m, 1, 3))  # Edge case