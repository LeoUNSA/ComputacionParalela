#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
using namespace std;
using namespace std::chrono;
const int N = 2048;
const int BLOCK_SIZE = 32;
void initializeMatrix(vector<vector<double>>& matrix) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }
}
void classicMultiplication(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void blockMultiplication(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Multiplicación dentro del bloque
                for (int ii = i; ii < min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE, N); jj++) {
                        for (int kk = k; kk < min(k + BLOCK_SIZE, N); kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}
int main() {
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0.0));
    initializeMatrix(A);
    initializeMatrix(B);
    // Multiplicacion clasica
    auto start = high_resolution_clock::now();
    classicMultiplication(A, B, C);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Tiempo de multiplicacion clasica: " << duration.count() << " milisegundos" << endl;
    fill(C.begin(), C.end(), vector<double>(N, 0.0));
    // Multiplicación por bloques
    start = high_resolution_clock::now();
    blockMultiplication(A, B, C);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Tiempo de multiplicacion por bloques: " << duration.count() << " milisegundos" << endl;
    return 0;
}