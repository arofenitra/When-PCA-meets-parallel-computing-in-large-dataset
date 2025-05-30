
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "CUSOLVER error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// CUDA kernel functions must be outside the class
__global__ void centerDataKernel(float *data, float *centered_data, float *means, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        centered_data[idy * n + idx] = data[idy * n + idx] - means[idx];
    }
}

__global__ void computeMeansKernel(float *data, float *means, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += data[i * n + idx];
        }
        means[idx] = sum / n;
    }
}

class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        CHECK_CUDA(cudaEventRecord(start));
    }
    
    float stopTimer() {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};

class CudaSVDPCA {
private:
    int n;
    float *d_matrix, *d_U, *d_S, *d_VT;
    float *d_covariance, *d_eigenvalues, *d_eigenvectors;
    float *d_centered_data, *d_principal_components;
    
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    curandGenerator_t curandGen;
    
    std::vector<float> timing_data;
    std::vector<std::string> timing_labels;

public:
    CudaSVDPCA(int size) : n(size) {
        // Allocate GPU memory
        CHECK_CUDA(cudaMalloc(&d_matrix, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_U, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_S, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_VT, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_covariance, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_eigenvalues, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_eigenvectors, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_centered_data, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_principal_components, n * n * sizeof(float)));
        
        // Initialize handles
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverHandle));
        CHECK_CURAND(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curandGen, 1234ULL));
    }
    
    ~CudaSVDPCA() {
        cudaFree(d_matrix);
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_VT);
        cudaFree(d_covariance);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cudaFree(d_centered_data);
        cudaFree(d_principal_components);
        
        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);
        curandDestroyGenerator(curandGen);
    }
    
    void fillMatrixRandom() {
        CudaTimer timer;
        timer.startTimer();
        
        CHECK_CURAND(curandGenerateUniform(curandGen, d_matrix, n * n));
        
        float elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Random Fill");
        
        std::cout << "Matrix filled with random numbers: " << elapsed << " ms" << std::endl;
    }
    
    void matrixOperations() {
        CudaTimer timer;
        timer.startTimer();
        
        const float alpha = 1.0f, beta = 0.0f;
        
        // Matrix multiplication: A * A^T (for demonstration)
        CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                n, n, n, &alpha, d_matrix, n, d_matrix, n,
                                &beta, d_covariance, n));
        
        float elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Matrix Operations");
        
        std::cout << "Matrix operations completed: " << elapsed << " ms" << std::endl;
    }
    
    void computeEigendecomposition() {
        CudaTimer timer;
        timer.startTimer();
        
        // Query workspace size
        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_UPPER, n, d_covariance, n,
                                                  d_eigenvalues, &lwork));
        
        float *d_work;
        int *d_info;
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        
        // Copy covariance matrix for eigendecomposition
        CHECK_CUDA(cudaMemcpy(d_eigenvectors, d_covariance, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Compute eigendecomposition
        CHECK_CUSOLVER(cusolverDnSsyevd(cusolverHandle, CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_UPPER, n, d_eigenvectors, n,
                                       d_eigenvalues, d_work, lwork, d_info));
        
        // Check convergence
        int info;
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            std::cerr << "Eigendecomposition failed with info = " << info << std::endl;
        }
        
        cudaFree(d_work);
        cudaFree(d_info);
        
        float elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("Eigendecomposition");
        
        std::cout << "Eigendecomposition completed: " << elapsed << " ms" << std::endl;
    }
    
    void computeSVD() {
        CudaTimer timer;
        timer.startTimer();
        
        // Query workspace size for SVD
        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverHandle, n, n, &lwork));
        
        float *d_work;
        float *d_rwork = nullptr; // For complex matrices only
        int *d_info;
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        
        // Copy original matrix for SVD
        float *d_A_copy;
        CHECK_CUDA(cudaMalloc(&d_A_copy, n * n * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A_copy, d_matrix, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Compute SVD: A = U * S * V^T
        CHECK_CUSOLVER(cusolverDnSgesvd(cusolverHandle, 'A', 'A', n, n,
                                       d_A_copy, n, d_S, d_U, n, d_VT, n,
                                       d_work, lwork, d_rwork, d_info));
        
        // Check convergence
        int info;
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            std::cerr << "SVD failed with info = " << info << std::endl;
        }
        
        cudaFree(d_A_copy);
        cudaFree(d_work);
        cudaFree(d_info);
        
        float elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("SVD Computation");
        
        std::cout << "SVD completed: " << elapsed << " ms" << std::endl;
    }
    
    void computePCA() {
        CudaTimer timer;
        timer.startTimer();
        
        // Compute column means
        float *d_means;
        CHECK_CUDA(cudaMalloc(&d_means, n * sizeof(float)));
        
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        computeMeansKernel<<<gridSize, blockSize>>>(d_matrix, d_means, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Center the data
        dim3 blockSize2D(16, 16);
        dim3 gridSize2D((n + blockSize2D.x - 1) / blockSize2D.x,
                       (n + blockSize2D.y - 1) / blockSize2D.y);
        centerDataKernel<<<gridSize2D, blockSize2D>>>(d_matrix, d_centered_data, d_means, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Compute covariance matrix: C = (1/(n-1)) * X^T * X
        const float alpha = 1.0f / (n - 1);
        const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                n, n, n, &alpha, d_centered_data, n,
                                d_centered_data, n, &beta, d_covariance, n));
        
        // Compute eigendecomposition of covariance matrix (already done above)
        // The eigenvectors are the principal components
        
        // Project data onto principal components
        const float alpha_proj = 1.0f, beta_proj = 0.0f;
        CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                n, n, n, &alpha_proj, d_centered_data, n,
                                d_eigenvectors, n, &beta_proj, d_principal_components, n));
        
        cudaFree(d_means);
        
        float elapsed = timer.stopTimer();
        timing_data.push_back(elapsed);
        timing_labels.push_back("PCA Computation");
        
        std::cout << "PCA completed: " << elapsed << " ms" << std::endl;
    }
    
    void saveResults() {
        // Save timing data
        std::ofstream timing_file("timing_results.csv");
        timing_file << "Operation,Time_ms" << std::endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            timing_file << timing_labels[i] << "," << timing_data[i] << std::endl;
        }
        timing_file.close();
        
        // Save singular values
        std::vector<float> singular_values(n);
        CHECK_CUDA(cudaMemcpy(singular_values.data(), d_S, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream sv_file("singular_values.csv");
        sv_file << "Index,Singular_Value" << std::endl;
        for (int i = 0; i < n; i++) {
            sv_file << i << "," << std::fixed << std::setprecision(6) << singular_values[i] << std::endl;
        }
        sv_file.close();
        
        // Save eigenvalues
        std::vector<float> eigenvalues(n);
        CHECK_CUDA(cudaMemcpy(eigenvalues.data(), d_eigenvalues, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream eigen_file("eigenvalues.csv");
        eigen_file << "Index,Eigenvalue" << std::endl;
        for (int i = 0; i < n; i++) {
            eigen_file << i << "," << std::fixed << std::setprecision(6) << eigenvalues[i] << std::endl;
        }
        eigen_file.close();
        
        // Save first few principal components for visualization
        std::vector<float> pc_data(n * std::min(5, n));
        CHECK_CUDA(cudaMemcpy(pc_data.data(), d_principal_components, 
                             n * std::min(5, n) * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::ofstream pc_file("principal_components.csv");
        pc_file << "PC1,PC2,PC3,PC4,PC5" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < std::min(5, n); j++) {
                pc_file << std::fixed << std::setprecision(6) << pc_data[j * n + i];
                if (j < std::min(5, n) - 1) pc_file << ",";
            }
            pc_file << std::endl;
        }
        pc_file.close();
        
        std::cout << "Results saved to CSV files for Python visualization" << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n=== CUDA SVD and PCA Summary ===" << std::endl;
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "Total operations: " << timing_labels.size() << std::endl;
        
        float total_time = 0;
        for (float time : timing_data) {
            total_time += time;
        }
        std::cout << "Total computation time: " << total_time << " ms" << std::endl;
        
        std::cout << "\nTiming breakdown:" << std::endl;
        for (size_t i = 0; i < timing_data.size(); i++) {
            std::cout << "  " << timing_labels[i] << ": " 
                     << timing_data[i] << " ms (" 
                     << std::fixed << std::setprecision(1) 
                     << (timing_data[i] / total_time) * 100 << "%)" << std::endl;
        }
    }
};

int main() {
    const int MATRIX_SIZE = 1000;
    
    std::cout << "Starting CUDA SVD and PCA Implementation" << std::endl;
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;
    
    try {
        CudaSVDPCA solver(MATRIX_SIZE);
        
        // Execute the algorithm pipeline
        solver.fillMatrixRandom();
        solver.matrixOperations();
        solver.computeEigendecomposition();
        solver.computeSVD();
        solver.computePCA();
        
        // Save results and print summary
        solver.saveResults();
        solver.printSummary();
        
        std::cout << "\nAll computations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
