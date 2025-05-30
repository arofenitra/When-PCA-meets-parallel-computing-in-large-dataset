#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

class Timer {
private:
    high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = high_resolution_clock::now();
        return duration_cast<milliseconds>(end_time - start_time).count();
    }
};

class SVDPCA {
private:
    int n;
    MatrixXf matrix, U, S, VT;
    MatrixXf covariance, eigenvalues, eigenvectors;
    MatrixXf centered_data, principal_components;
    vector<double> timing_data;
    vector<string> timing_labels;
    
public:
    SVDPCA(int size) : n(size) {
        // Initialize matrices
        matrix = MatrixXf::Zero(n, n);
        U = MatrixXf::Zero(n, n);
        S = VectorXf::Zero(n);
        VT = MatrixXf::Zero(n, n);
        covariance = MatrixXf::Zero(n, n);
        eigenvalues = VectorXf::Zero(n);
        eigenvectors = MatrixXf::Zero(n, n);
        centered_data = MatrixXf::Zero(n, n);
        principal_components = MatrixXf::Zero(n, n);
    }
    
    void fillMatrixRandom() {
        Timer timer;
        timer.start();
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dis(0.0, 1.0);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix(i, j) = dis(gen);
            }
        }
        
        double elapsed = timer.stop();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Random Fill");
        cout << "Matrix filled with random numbers: " << elapsed << " ms" << endl;
    }
    
    void matrixOperations() {
        Timer timer;
        timer.start();
        
        // Matrix multiplication: A * A^T
        covariance = matrix * matrix.transpose();
        
        double elapsed = timer.stop();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Operations");
        cout << "Matrix operations completed: " << elapsed << " ms" << endl;
    }
    
    void computeEigendecomposition() {
        Timer timer;
        timer.start();
        
        // Compute eigenvalues and eigenvectors of covariance matrix
        SelfAdjointEigenSolver<MatrixXf> eigensolver(covariance);
        if (eigensolver.info() != Success) {
            cerr << "Eigen decomposition failed!" << endl;
            return;
        }
        
        eigenvalues = eigensolver.eigenvalues();
        eigenvectors = eigensolver.eigenvectors();
        
        // Sort eigenvalues and eigenvectors in descending order
        VectorXi indices = VectorXi::LinSpaced(n, 0, n-1);
        std::sort(indices.data(), indices.data()+n, 
            [&](int i, int j){ return eigenvalues(i) > eigenvalues(j); });
        
        VectorXf sorted_eigenvalues(n);
        MatrixXf sorted_eigenvectors(n, n);
        for (int i = 0; i < n; i++) {
            sorted_eigenvalues(i) = eigenvalues(indices[i]);
            sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
        }
        eigenvalues = sorted_eigenvalues;
        eigenvectors = sorted_eigenvectors;
        
        double elapsed = timer.stop();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Eigendecomposition");
        cout << "Eigendecomposition completed: " << elapsed << " ms" << endl;
    }
    
    void computeSVD() {
        Timer timer;
        timer.start();
        
        // Compute full SVD
        BDCSVD<MatrixXf> svd(matrix, ComputeFullU | ComputeFullV);
        if (svd.info() != Success) {
            cerr << "SVD failed!" << endl;
            return;
        }
        
        U = svd.matrixU();
        VectorXf singular_values = svd.singularValues();
        for (int i = 0; i < n; i++) {
            S(i) = singular_values(i);
        }
        VT = svd.matrixV().transpose();
        
        double elapsed = timer.stop();
        timing_data.push_back(elapsed);
        timing_labels.push_back("SVD Computation");
        cout << "SVD completed: " << elapsed << " ms" << endl;
    }
    
    void computePCA() {
        Timer timer;
        timer.start();
        
        // Center the data
        VectorXf means = matrix.colwise().mean();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                centered_data(i, j) = matrix(i, j) - means(j);
            }
        }
        
        // The eigendecomposition was already done
        // The eigenvectors are the principal components
        
        // Project data onto principal components
        principal_components = centered_data * eigenvectors;
        
        double elapsed = timer.stop();
        timing_data.push_back(elapsed);
        timing_labels.push_back("PCA Computation");
        cout << "PCA completed: " << elapsed << " ms" << endl;
    }
    
    void saveResults() {
        // Save timing data
        ofstream timing_file("timing_results.csv");
        timing_file << "Operation,Time_ms" << endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            timing_file << timing_labels[i] << "," << fixed << setprecision(2) << timing_data[i] << endl;
        }
        timing_file.close();
        
        // Save singular values
        ofstream sv_file("singular_values.csv");
        sv_file << "Index,Singular_Value" << endl;
        for (int i = 0; i < n; i++) {
            sv_file << i << "," << fixed << setprecision(6) << S(i) << endl;
        }
        sv_file.close();
        
        // Save eigenvalues
        ofstream eigen_file("eigenvalues.csv");
        eigen_file << "Index,Eigenvalue" << endl;
        for (int i = 0; i < n; i++) {
            eigen_file << i << "," << fixed << setprecision(6) << eigenvalues(i) << endl;
        }
        eigen_file.close();
        
        // Save first few principal components for visualization
        ofstream pc_file("principal_components.csv");
        pc_file << "PC1,PC2,PC3,PC4,PC5" << endl;
        int num_pc = min(5, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < num_pc; j++) {
                pc_file << fixed << setprecision(6) << principal_components(i, j);
                if (j < num_pc - 1) pc_file << ",";
            }
            pc_file << endl;
        }
        pc_file.close();
        
        cout << "Results saved to CSV files for Python visualization" << endl;
    }
    
    void printSummary() {
        cout << "\n=== SVD and PCA Summary ===" << endl;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Total operations: " << timing_labels.size() << endl;
        double total_time = 0;
        for (double time : timing_data) {
            total_time += time;
        }
        cout << "Total computation time: " << total_time << " ms" << endl;
        cout << "\nTiming breakdown:" << endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            cout << "  " << timing_labels[i] << ": " 
                 << fixed << setprecision(2) << timing_data[i] << " ms (" 
                 << setprecision(1) << (timing_data[i] / total_time) * 100 << "%)" << endl;
        }
    }
};

int main() {
    const int MATRIX_SIZE = 5000;
    cout << "Starting Serial SVD and PCA Implementation" << endl;
    cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << endl;
    
    try {
        SVDPCA solver(MATRIX_SIZE);
        // Execute the algorithm pipeline
        solver.fillMatrixRandom();
        solver.matrixOperations();
        solver.computeEigendecomposition();
        solver.computeSVD();
        solver.computePCA();
        // Save results and print summary
        solver.saveResults();
        solver.printSummary();
        cout << "\nAll computations completed successfully!" << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
