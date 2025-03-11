__kernel void a2b2(
    __global double* A, 
    __global double* B, 
    __global double* C, 
    const int N, const int K) 
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < 2 && col < K) {
        double value = 0;
        for (int i = 0; i < N; i++) {
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

__kernel void a3b2(
    __global double *A,
    __global double *B,
    __global double *C,
    const int D, const int M, const int N, const int P)
{
    int d = get_global_id(0);
    int m = get_global_id(1);
    int p = get_global_id(2);

    if (d < D && m < M && p < P) {
        double value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[d * (M * N) + m * N + k] * B[k * P + p];
        }
        C[d * (M * P) + m * P + p] = value;
    }
}

__kernel void a3b3(
    __global double* A, 
    __global double* B, 
    __global double* C, 
    const int M,  
    const int K,  
    const int N)  
{
    int b = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    double value = 0;
    for (int k = 0; k < K; k++) {
        value += A[b * (M * K) + i * K + k] * 
                 B[b * (K * N) + k * N + j];
    }
    C[b * (M * N) + i * N + j] = value;
}
