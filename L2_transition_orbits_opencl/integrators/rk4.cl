#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WORKGROUP_SIZE 64
#define KAPPASQ 39.47841760435743447

__kernel void integrator(__global const double* t0, __global const double* h, __global const int* n, __global const int* m, __global const double* x0, __global double* xres)
{
    __local double x[N * WORKGROUP_SIZE];
    __local double k1[N * WORKGROUP_SIZE];
    __local double k2[N * WORKGROUP_SIZE];
    __local double k3[N * WORKGROUP_SIZE];
    __local double k4[N * WORKGROUP_SIZE];
    __local double tmp[N * WORKGROUP_SIZE];
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int shft = local_id * N;
    const int globshft = global_id * N;
    double tt0 = t0[0];
    double hh = h[0];
    int nn = n[0];
    int mm = m[0];
    int nt = nn / mm;
    double* xlocal = x + shft;
    double* k1local = k1 + shft;
    double* k2local = k2 + shft;
    double* k3local = k3 + shft;
    double* k4local = k4 + shft;
    double* tmplocal = tmp + shft;
    int i, j, l, t0;
    double t, t1;

    for(i = 0; i < N; ++i)
    {
        x[shft + i] = x0[global_id * N + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(l = 0; l < N; ++l)
    {
        xres[globshft + l] = xlocal[l];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(i = 0; i < nt; ++i)
    {
        t0 = i * mm;
        for(j = 0; j < mm; ++j)
        {
            t = tt0 + hh * (t0 + j);
            F(t, xlocal, k1local);
            for(l = 0; l < N; ++l)
            {
                k1local[l] *= hh;
                tmplocal[l] = xlocal[l] + k1local[l] / 2.0;
            }
            t1 = t + hh / 2.0;
            F(t1, tmplocal, k2local);
            for(l = 0; l < N; ++l)
            {
                k2local[l] *= hh;
                tmplocal[l] = xlocal[l] + k2local[l] / 2.0;
            }
            F(t1, tmplocal, k3local);
            for(l = 0; l < N; ++l)
            {
                k3local[l] *= hh;
                tmplocal[l] = xlocal[l] + k3local[l];
            }
            t1 = t + hh;
            F(t1, tmplocal, k4local);
            for(l = 0; l < N; ++l)
            {
                k4local[l] *= hh;
                xlocal[l] += (k1local[l] + 2.0 * k2local[l] + 2.0 * k3local[l] + k4local[l]) / 6.0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for(l = 0; l < N; ++l)
        {
            xres[globshft + l + (i + 1) * WORKRANGE_SIZE * N] = xlocal[l];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
