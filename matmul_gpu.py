import pyopencl as cl
import numpy as np
import os

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
kernel_src = open("matmulGPU.c").read()
program = cl.Program(context, kernel_src).build()

def a2b2(A, B):
    A, B = A.copy().astype(np.float64), B.copy().astype(np.float64)
    M, N = A.shape
    K = B.shape[1]
    C = np.zeros((M, K), dtype=np.float64)
    mf = cl.mem_flags
    A_g = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=A)
    B_g = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=B)
    C_g = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes, hostbuf=None)
    program.a2b2(queue, C.shape, None, A_g, B_g, C_g, np.int32(N), np.int32(K))
    A_g.release()
    B_g.release()
    cl.enqueue_copy(queue, C, C_g).wait()
    C_g.release()
    return C

def a3b2(A, B):
    A, B = A.copy().astype(np.float64), B.copy().astype(np.float64)
    D, M, N = A.shape
    P = B.shape[-1]
    C = np.zeros((D, M, P), dtype=np.float64)
    mf = cl.mem_flags
    A_g = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=A)
    B_g = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=B)
    C_g = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes, hostbuf=None)
    program.a3b2(queue, C.shape, None, A_g, B_g, C_g, np.int32(D), np.int32(M), np.int32(N), np.int32(P))
    A_g.release()
    B_g.release()
    cl.enqueue_copy(queue, C, C_g).wait()
    C_g.release()
    return C

def a3b3(A, B):
    A, B = A.copy().astype(np.float64), B.copy().astype(np.float64)
    BATCH, M, K = A.shape
    N = B.shape[-1]
    C = np.zeros((BATCH, M, N), dtype=np.float64)
    mf = cl.mem_flags
    A_buf = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(context, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes, hostbuf=None)
    program.a3b3(queue, C.shape, None, A_buf, B_buf, C_buf, np.int32(M), np.int32(K), np.int32(N))
    A_buf.release()
    B_buf.release()
    cl.enqueue_copy(queue, C, C_buf).wait()
    C_buf.release()
    return C

def matmulGPU(A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return a2b2(A, B)
    elif len(A.shape) == 3 and len(B.shape) == 2:
        return a3b2(A, B)
    elif len(A.shape) == 3 and len(B.shape) == 3:
        return a3b3(A, B)
    elif len(A.shape) == 4 and len(B.shape) == 4:
        dim0, dim1 = A.shape[0], A.shape[1]
        dim_a2, dim_a3 = A.shape[2], A.shape[3]
        dim_b2, dim_b3 = B.shape[2], B.shape[3]
        A = A.reshape(dim0 * dim1, dim_a2, dim_a3)
        B = B.reshape(dim0 * dim1, dim_b2, dim_b3)
        return a3b3(A, B).reshape(dim0, dim1, dim_a2, dim_b3)
    else:
        print("tính toán trên CPU")
        return np.matmul(A, B)
    
# A = np.random.rand(2, 3, 6, 3).astype(np.float64)
# B = np.random.rand(2, 3, 3, 6).astype(np.float64)
# print(matmulGPU(A, B))
# print(np.matmul(A, B))