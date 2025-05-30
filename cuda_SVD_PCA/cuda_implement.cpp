
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err_) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CURAND_CHECK(err) \
    do { \
        curandStatus_t err_ = (err); \
        if (err_ != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error " << __FILE__ << ":" << __LINE__ << " - " \
                      << (int)err_ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUSOLVER_CHECK(err) \
    do { \
        cusolverStatus_t err_ = (err); \
        if (err_ != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "CUSOLVER error " << __FILE__ << ":" << __LINE__ << " - " \
                      << (int)err_ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Kernel to compute column means
__global__ void computeMeansKernel(const float* data, float* means, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for(int i = 0; i < n; ++i) {
            sum += data[i * n + idx];
        }
        means[idx] = sum / n;
    }
}

// Kernel to center data by subtracting mean from each element
__global__ void centerDataKernel(const float* data, float* centered_data, const float* means, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int col = idx % n;
        centered_data[idx] = data[idx] - means[col];
    }
}

class CudaSVDPCA {
    int n;
    float* d_matrix;
    float* d_centered;
    float* d_means;
    curandGenerator_t curandGen;
    cusolverDnHandle_t cusolverH;

public:
    CudaSVDPCA(int size) : n(size) {
        CUDA_CHECK(cudaMalloc(&d_matrix, n * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centered, n * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_means, n * sizeof(float)));
        CURAND_CHECK(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curandGen, 1234ULL));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    }

    ~CudaSVDPCA() {
        CUDA_CHECK(cudaFree(d_matrix));
        CUDA_CHECK(cudaFree(d_centered));
        CUDA_CHECK(cudaFree(d_means));
        CURAND_CHECK(curandDestroyGenerator(curandGen));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    }

    void fillMatrixRandom() {
        CURAND_CHECK(curandGenerateUniform(curandGen, d_matrix, n * n));
    }

    void computeMeans() {
        const int blockSize = 256;
        const int gridSize = (n + blockSize - 1) / blockSize;
        computeMeansKernel<<<gridSize, blockSize>>>(d_matrix, d_means, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void centerData() {
        const int blockSize = 256;
        const int gridSize = (n * n + blockSize - 1) / blockSize;
        centerDataKernel<<<gridSize, blockSize>>>(d_matrix, d_centered, d_means, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void performPCA() {
        // SVD using cuSOLVER
        int lwork = 0;
        float* d_U = nullptr;
        float* d_S = nullptr;
        float* d_VT = nullptr;
        int* devInfo = nullptr;

        CUDA_CHECK(cudaMalloc(&d_U, n * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_S, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_VT, n * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

        CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH, n, n, &lwork));
        float* d_work = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

        signed char jobu = 'A';
        signed char jobvt = 'A';
        float* d_rwork = nullptr; // Not used in single precision

        CUSOLVER_CHECK(cusolverDnSgesvd(
            cusolverH,
            jobu,
            jobvt,
            n,
            n,
            d_centered,
            n,
            d_S,
            d_U,
            n,
            d_VT,
            n,
            d_work,
            lwork,
            d_rwork,
            devInfo));

        // Normally, you would copy results back and use them here

        CUDA_CHECK(cudaFree(d_U));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(d_VT));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(devInfo));
    }
};

int main() {
    const int n = 512; // You can change the matrix size
    CudaSVDPCA pca(n);
    pca.fillMatrixRandom();
    pca.computeMeans();
    pca.centerData();
    pca.performPCA();
    std::cout << "PCA computation completed successfully." << std::endl;
    return 0;
}
