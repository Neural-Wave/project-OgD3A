def read_adjacency_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            # Strip whitespace and split the line into elements
            elements = line.strip().split()
            if elements:  # Ensure the line is not empty
                row = [int(float(x)) for x in elements]
                matrix.append(row)
    return matrix

def write_adjacency_matrix(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

def main():
    # Read the original matrix
    original_matrix = read_adjacency_matrix('oli/adjacency_matrix.txt')
    
    # Write the integer version of the original matrix
    write_adjacency_matrix(original_matrix, 'adjacency_matrix_int.txt')
    
    # Read the first fifteen rows from head15.txt
    head15 = read_adjacency_matrix('head15.txt')
    
    # Replace the first fifteen rows in the original matrix
    for i in range(min(15, len(head15), len(original_matrix))):
        original_matrix[i] = head15[i]
    
    # Write the modified matrix to a new file
    write_adjacency_matrix(original_matrix, 'adjacency_matrix_modified.txt')

if __name__ == "__main__":
    main()
