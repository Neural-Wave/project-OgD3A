def count_ones_in_matrix(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += line.count('1')
    return count

if __name__ == "__main__":
    file_path = 'misc/lower_triangular_gt.txt'
    ones_count = count_ones_in_matrix(file_path)
    print(f"Number of ones in the matrix: {ones_count}")