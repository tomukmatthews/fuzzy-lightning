#ifndef ARRAY_H
#define ARRAY_H

#include <vector>

struct IntArray2D
{
    // The number of rows in the 2D array
    int num_rows;

    // The number of columns in the 2D array
    int num_columns;

    // The 1D array used for storage
    std::vector<int> data;

    // Constructor that initializes the struct with the given number of rows and columns
    IntArray2D(int num_rows, int num_columns) : num_rows(num_rows), num_columns(num_columns)
    {
        data = std::vector<int>(num_rows * num_columns, 0);
    }

    // Destructor that frees the memory used by the 1D array
    ~IntArray2D()
    {
        // no need to explicitly delete the vector object; it will be
        // automatically destroyed when the struct is destroyed
    }

    // Returns the value at the given row and column, using row-major order
    int get(int row, int col) const
    {
        return data[row * num_columns + col];
    }

    // Sets the value at the given row and column, using row-major order
    void set(int row, int col, int value)
    {
        data[row * num_columns + col] = value;
    }
};

#endif // ARRAY_H