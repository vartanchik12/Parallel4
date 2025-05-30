#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

using Matrix = std::vector<std::vector<double>>;

Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }
    Matrix matrix;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

void writeMatrixToFile(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + filename);
    }
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < K) {
        double sum = 0.0;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

Matrix multiplyMatricesCUDA(const Matrix& A, const Matrix& B) {
    int N = A.size();
    int M = A[0].size();
    int K = B[0].size();

    std::vector<double> h_A(N * M);
    std::vector<double> h_B(M * K);
    std::vector<double> h_C(N * K, 0.0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            h_A[i * M + j] = A[i][j];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_B[i * K + j] = B[i][j];

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * M * sizeof(double));
    cudaMalloc(&d_B, M * K * sizeof(double));
    cudaMalloc(&d_C, N * K * sizeof(double));

    cudaMemcpy(d_A, h_A.data(), N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, N * K * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    Matrix result(N, std::vector<double>(K));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < K; ++j)
            result[i][j] = h_C[i * K + j];
    return result;
}

int main() {
    std::string matrix_A_file = "C:/Users/artan/OneDrive/Desktop/paralel/matrixA.txt";
    std::string matrix_B_file = "C:/Users/artan/OneDrive/Desktop/paralel/matrixB.txt";
    std::string matrix_result_file = "C:/Users/artan/OneDrive/Desktop/paralel/result.txt";

    Matrix A = readMatrix(matrix_A_file);
    Matrix B = readMatrix(matrix_B_file);
    if (A.empty() || B.empty()) {
        throw std::runtime_error("Матрицы пусты или некорректны");
    }

    auto start = chrono::high_resolution_clock::now();
    Matrix result = multiplyMatricesCUDA(A, B);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time = end - start;

    writeMatrixToFile(result, matrix_result_file);

    ofstream fout("C:/Users/artan/OneDrive/Desktop/paralel/info.txt");
    fout << "Execution time (sec): " << time.count() << endl;
    fout.close();

    return 0;
} 