
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>

#ifdef USE_INTEL_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#else
// For systems without MKL, you'll need BLAS/LAPACK
extern "C" {
    void sgemm_(const char*, const char*, const int*, const int*, const int*,
                const float*, const float*, const int*, const float*, const int*,
                const float*, float*, const int*);
    void ssyev_(const char*, const char*, const int*, float*, const int*,
                float*, float*, const int*, int*);
    void sgesvd_(const char*, const char*, const int*, const int*, float*, const int*,
                 float*, float*, const int*, float*, const int*, float*, const int*, int*);
}
#endif

class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void startTimer() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stopTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Return milliseconds
    }
};

class OpenMPSVDPCA {
private:
    int n;
    std::vector<float> matrix, U, S, VT;
    std::vector<float> covariance, eigenvalues, eigenvectors;
    std::vector<float> centered_data, principal_components;
    std::vector<float> means;
    
    std::vector<double> timing_data;
    std::vector<std::string> timing_labels;
    std::vector<int> thread_counts;

public:
    OpenMPSVDPCA(int size) : n(size) {
        // Allocate memory
        matrix.resize(n * n);
        U.resize(n * n);
        S.resize(n);
        VT.resize(n * n);
        covariance.resize(n * n);
        eigenvalues.resize(n);
        eigenvectors.resize(n * n);
        centered_data.resize(n * n);
        principal_components.resize(n * n);
        means.resize(n);
    }
    
    void fillMatrixRandom() {
        CPUTimer timer;
        timer.startTimer();
        
        std::random_device rd;
        std::mt19937 gen(1234); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        #pragma omp parallel
        {
            std::mt19937 local_gen(1234 + omp_get_thread_num());
            std::uniform_real_distribution<float> local_dis(0.0f, 1.0f);
            
            #pragma omp for
            for (int i = 0; i < n * n; i++) {
                matrix[i] = local_dis(local_gen);
            }
        }
        
        double elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Random Fill");
        thread_counts.push_back(omp_get_max_threads());
        
        std::cout << "Matrix filled with random numbers: " << elapsed << " ms (threads: " 
                  << omp_get_max_threads() << ")" << std::endl;
    }
    
    void matrixOperations() {
        CPUTimer timer;
        timer.startTimer();
        
        // Matrix multiplication: A * A^T using parallel BLAS or manual implementation
        const float alpha = 1.0f, beta = 0.0f;
        
#ifdef USE_INTEL_MKL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n, n, n, alpha, matrix.data(), n, matrix.data(), n,
                    beta, covariance.data(), n);
#else
        // Manual parallel matrix multiplication
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += matrix[i * n + k] * matrix[j * n + k]; // A * A^T
                }
                covariance[i * n + j] = sum;
            }
        }
#endif
        
        double elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Operations");
        thread_counts.push_back(omp_get_max_threads());
        
        std::cout << "Matrix operations completed: " << elapsed << " ms (threads: " 
                  << omp_get_max_threads() << ")" << std::endl;
    }
    
    void computeEigendecomposition() {
        CPUTimer timer;
        timer.startTimer();
        
        // Copy covariance matrix for eigendecomposition
        eigenvectors = covariance;
        
#ifdef USE_INTEL_MKL
        int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n, 
                                eigenvectors.data(), n, eigenvalues.data());
#else
        // Using LAPACK directly
        char jobz = 'V', uplo = 'U';
        int lwork = 3 * n - 1;
        std::vector<float> work(lwork);
        int info;
        
        ssyev_(&jobz, &uplo, &n, eigenvectors.data(), &n, 
               eigenvalues.data(), work.data(), &lwork, &info);
#endif
        
        if (info != 0) {
            std::cerr << "Eigendecomposition failed with info = " << info << std::endl;
        }
        
        // Sort eigenvalues and eigenvectors in descending order
        std::vector<std::pair<float, int>> eigenvalue_pairs;
        for (int i = 0; i < n; i++) {
            eigenvalue_pairs.push_back({eigenvalues[i], i});
        }
        std::sort(eigenvalue_pairs.rbegin(), eigenvalue_pairs.rend());
        
        std::vector<float> sorted_eigenvalues(n);
        std::vector<float> sorted_eigenvectors(n * n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            sorted_eigenvalues[i] = eigenvalue_pairs[i].first;
            int orig_idx = eigenvalue_pairs[i].second;
            for (int j = 0; j < n; j++) {
                sorted_eigenvectors[i * n + j] = eigenvectors[orig_idx * n + j];
            }
        }
        
        eigenvalues = sorted_eigenvalues;
        eigenvectors = sorted_eigenvectors;
        
        double elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Eigendecomposition");
        thread_counts.push_back(omp_get_max_threads());
        
        std::cout << "Eigendecomposition completed: " << elapsed << " ms (threads: " 
                  << omp_get_max_threads() << ")" << std::endl;
    }
    
    void computeSVD() {
        CPUTimer timer;
        timer.startTimer();
        
        // Copy original matrix for SVD
        std::vector<float> A_copy = matrix;
        
#ifdef USE_INTEL_MKL
        int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n,
                                 A_copy.data(), n, S.data(),
                                 U.data(), n, VT.data(), n, nullptr);
#else
        // Using LAPACK directly
        char jobu = 'A', jobvt = 'A';
        int lwork = 5 * n;
        std::vector<float> work(lwork);
        int info;
        
        sgesvd_(&jobu, &jobvt, &n, &n, A_copy.data(), &n, S.data(),
                U.data(), &n, VT.data(), &n, work.data(), &lwork, &info);
#endif
        
        if (info != 0) {
            std::cerr << "SVD failed with info = " << info << std::endl;
        }
        
        double elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("SVD Computation");
        thread_counts.push_back(omp_get_max_threads());
        
        std::cout << "SVD completed: " << elapsed << " ms (threads: " 
                  << omp_get_max_threads() << ")" << std::endl;
    }
    
    void computePCA() {
        CPUTimer timer;
        timer.startTimer();
        
        // Compute column means
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += matrix[i * n + j];
            }
            means[j] = sum / n;
        }
        
        // Center the data
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                centered_data[i * n + j] = matrix[i * n + j] - means[j];
            }
        }
        
        // Compute covariance matrix: C = (1/(n-1)) * X^T * X
        const float scale = 1.0f / (n - 1);
        
#ifdef USE_INTEL_MKL
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, n, n, scale, centered_data.data(), n,
                    centered_data.data(), n, 0.0f, covariance.data(), n);
#else
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += centered_data[k * n + i] * centered_data[k * n + j];
                }
                covariance[i * n + j] = sum * scale;
            }
        }
#endif
        
        // Project data onto principal components
#ifdef USE_INTEL_MKL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n, n, n, 1.0f, centered_data.data(), n,
                    eigenvectors.data(), n, 0.0f, principal_components.data(), n);
#else
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += centered_data[i * n + k] * eigenvectors[j * n + k];
                }
                principal_components[i * n + j] = sum;
            }
        }
#endif
        
        double elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("PCA Computation");
        thread_counts.push_back(omp_get_max_threads());
        
        std::cout << "PCA completed: " << elapsed << " ms (threads: " 
                  << omp_get_max_threads() << ")" << std::endl;
    }
    
    void saveResults(const std::string& suffix = "") {
        std::string file_suffix = suffix.empty() ? "" : "_" + suffix;
        
        // Save timing data with thread counts
        std::ofstream timing_file("timing_results_openmp" + file_suffix + ".csv");
        timing_file << "Operation,Time_ms,Threads" << std::endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            timing_file << timing_labels[i] << "," << timing_data[i] << "," 
                       << thread_counts[i] << std::endl;
        }
        timing_file.close();
        
        // Save singular values
        std::ofstream sv_file("singular_values_openmp" + file_suffix + ".csv");
        sv_file << "Index,Singular_Value" << std::endl;
        for (int i = 0; i < n; i++) {
            sv_file << i << "," << std::fixed << std::setprecision(6) << S[i] << std::endl;
        }
        sv_file.close();
        
        // Save eigenvalues
        std::ofstream eigen_file("eigenvalues_openmp" + file_suffix + ".csv");
        eigen_file << "Index,Eigenvalue" << std::endl;
        for (int i = 0; i < n; i++) {
            eigen_file << i << "," << std::fixed << std::setprecision(6) 
                      << eigenvalues[i] << std::endl;
        }
        eigen_file.close();
        
        // Save first few principal components
        std::ofstream pc_file("principal_components_openmp" + file_suffix + ".csv");
        pc_file << "PC1,PC2,PC3,PC4,PC5" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < std::min(5, n); j++) {
                pc_file << std::fixed << std::setprecision(6) 
                       << principal_components[i * n + j];
                if (j < std::min(5, n) - 1) pc_file << ",";
            }
            pc_file << std::endl;
        }
        pc_file.close();
        
        std::cout << "Results saved to OpenMP CSV files" << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n=== OpenMP SVD and PCA Summary ===" << std::endl;
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Total operations: " << timing_labels.size() << std::endl;
        std::cout << "Max threads used: " << omp_get_max_threads() << std::endl;
        
        double total_time = 0;
        for (double time : timing_data) {
            total_time += time;
        }
        std::cout << "Total computation time: " << total_time << " ms" << std::endl;
        
        std::cout << "\nTiming breakdown:" << std::endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            std::cout << "  " << timing_labels[i] << ": " 
                     << timing_data[i] << " ms (" 
                     << std::fixed << std::setprecision(1) 
                     << (timing_data[i] / total_time) * 100 << "%) [" 
                     << thread_counts[i] << " threads]" << std::endl;
        }
    }
};

void runScalabilityTest(int matrix_size, const std::vector<int>& thread_counts) {
    std::ofstream scalability_file("scalability_openmp.csv");
    scalability_file << "Threads,Operation,Time_ms,Speedup,Efficiency" << std::endl;
    
    std::vector<double> baseline_times;
    
    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);
        
        std::cout << "\n=== Testing with " << num_threads << " threads ===" << std::endl;
        
        OpenMPSVDPCA solver(matrix_size);
        
        // Run all operations
        solver.fillMatrixRandom();
        solver.matrixOperations();
        solver.computeEigendecomposition();
        solver.computeSVD();
        solver.computePCA();
        
        // Save results for this thread count
        solver.saveResults(std::to_string(num_threads) + "threads");
        
        // Read timing results and compute speedup
        std::ifstream timing_file("timing_results_openmp_" + std::to_string(num_threads) + "threads.csv");
        std::string line;
        std::getline(timing_file, line); // Skip header
        
        int op_index = 0;
        while (std::getline(timing_file, line)) {
            std::stringstream ss(line);
            std::string operation, time_str, threads_str;
            
            std::getline(ss, operation, ',');
            std::getline(ss, time_str, ',');
            std::getline(ss, threads_str, ',');
            
            double time_ms = std::stod(time_str);
            
            if (num_threads == thread_counts[0]) {
                // Store baseline times (first thread count)
                baseline_times.push_back(time_ms);
            }
            
            double speedup = (op_index < baseline_times.size()) ? 
                           baseline_times[op_index] / time_ms : 1.0;
            double efficiency = speedup / num_threads;
            
            scalability_file << num_threads << "," << operation << "," 
                           << time_ms << "," << speedup << "," << efficiency << std::endl;
            
            op_index++;
        }
        timing_file.close();
    }
    
    scalability_file.close();
    std::cout << "\nScalability test completed. Results saved to scalability_openmp.csv" << std::endl;
}

int main() {
    const int MATRIX_SIZE = 1000;
    const std::vector<int> THREAD_COUNTS = {1, 2, 4, 8, 16, 32};
    
    std::cout << "Starting OpenMP SVD and PCA Implementation" << std::endl;
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    std::cout << "Available threads: " << omp_get_max_threads() << std::endl;
    
    try {
        // Run scalability test
        runScalabilityTest(MATRIX_SIZE, THREAD_COUNTS);
        
        std::cout << "\nAll computations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
