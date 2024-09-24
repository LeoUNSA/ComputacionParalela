#include <iostream>
#include <chrono>
#include <vector>
#include <random>
using namespace std;
using namespace std::chrono;
const int N = 1000;

void multiplyMatrixByRows(const vector<vector<double>>& a, const vector<double>& X, vector<double>& Y) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            Y[i] += a[i][j] * X[j];
        }
    }
}

void multiplyMatrixByColumns(const vector<vector<double>>& a, const vector<double>& X, vector<double>& Y) {
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++){
            Y[i] += a[i][j] * X[j];
        }
    }
}

void resetVector(vector<double>& Y) {
    fill(Y.begin(), Y.end(), 0.0);
}

// Función para inicializar una matriz o vector con valores aleatorios
void initializeRandom(vector<vector<double>>& matrix) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = dis(gen);
        }
    }
}

void initializeRandom(vector<double>& vec) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& elem : vec) {
        elem = dis(gen);
    }
}

int main(){
    // Inicialización de matrices y vectores con tamaño N
    vector<vector<double>> a(N, vector<double>(N));
    vector<double> X(N);
    vector<double> Y(N, 0.0);

    // Inicialización aleatoria
    initializeRandom(a);
    initializeRandom(X);

    // Numero de iteraciones
    const int itN = 1;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < itN; i++) {
        resetVector(Y);
        multiplyMatrixByRows(a, X, Y);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Tiempo del primer caso: " << itN << " iteraciones: " << duration.count() << " microsegundos" << endl;

    start = high_resolution_clock::now();
    for (int i = 0; i < itN; i++) {
        resetVector(Y);
        multiplyMatrixByColumns(a, X, Y);
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Tiempo del segundo caso: " << itN << " iteraciones: " << duration.count() << " microsegundos" << endl;

    return 0;
}
