#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WORKGROUP_SIZE 64

#define DBL_EPSILSON 2.5e-16
#define MAX_STEPS 100

#define NUM_BODIES_IN_WORKGROUP WORKGROUP_SIZE / S
#define NUM_BODIES_IN_WORKRANGE WORKRANGE_SIZE / S

void parallel_matrix_multiplication(double*, double*, double*, int);
double cdot(double*, double*);
void parallel_copy_vector(const double*, double*, int);
void parallel_copy_matrix(const double*, double*, int);
void copy_n_vector(int, const double*, double*);
void parallel_copy_n_vector(int, const double*, double*, int, int);
void parallel_set_elements(double*, double, int);
void parallel_calculate_difference(double*, double*, double*, double*, int);

void make_y_matrix(double*, double*, double*, double*, double*, int);
void make_f_matrix(double*, double*, double, double, double*, double*, int);

/*
void print_value_in_one_thread(double);
void print_vector_in_one_thread(int, double*);
void print_matrix_in_one_thread(int, int, double*);
*/

__kernel void integrator(__global const double* t0, __global const double* h, __global const int* n, __global const int* m, __global const double* x0, __global double* xres, __global const double* c_global, __global const double* matrix_ct_inv_transpose_global, __global const double* b_transpose_global)
{
    const int local_matrix_id = get_local_id(0) % S;
    const int local_body_id = get_local_id(0) / S;
    const int global_body_id = get_global_id(0) / S;
    const int X_IN_Y_SHFT = N * (S - 1);
    double tt0 = t0[0];
    double hh = h[0];
    int nn = n[0];
    int mm = m[0];
    __local double c_local[S];
    __local double matrix_ct_inv_transpose_local[S * S];
    __local double b_local[S * S];
    __local double x_local[NUM_BODIES_IN_WORKGROUP * N];
    __local double x_last_local[NUM_BODIES_IN_WORKGROUP * N];
    __local double a_matrices[N * S * NUM_BODIES_IN_WORKGROUP];
    __local double new_a_matrices[N * S * NUM_BODIES_IN_WORKGROUP];
    __local double f_matrices[N * S * NUM_BODIES_IN_WORKGROUP];
    __local double y_matrices[N * S * NUM_BODIES_IN_WORKGROUP];
    __local double differences[NUM_BODIES_IN_WORKGROUP];
    __local double ab_transpose[N * S * NUM_BODIES_IN_WORKGROUP];
    __local double f_matrices_tmp[S * N * NUM_BODIES_IN_WORKGROUP];
    __local double difference_buffer[S * NUM_BODIES_IN_WORKGROUP];
    int i, j, jj, jjj;
    int t_iter;
    int nt = nn / mm;
    double t_cur;
    double* a_local = a_matrices + N * S * local_body_id;
    double* new_a_local = new_a_matrices + N * S * local_body_id;
    double* f_local = f_matrices + N * S * local_body_id;
    double* y_local = y_matrices + N * S * local_body_id;
    double* abt_local = ab_transpose + N * S * local_body_id;
    double* ftmp_local = f_matrices_tmp + N * S * local_body_id;
    double* local_difference_buffer = difference_buffer + S * local_body_id;
    int local_shft = N * local_body_id;

    // Copy data to local memory
    if(local_body_id == 0)
    {
        parallel_copy_vector(c_global, c_local, local_matrix_id);
        parallel_copy_matrix(matrix_ct_inv_transpose_global, matrix_ct_inv_transpose_local, local_matrix_id);
        parallel_copy_matrix(b_transpose_global, b_local, local_matrix_id);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_matrix_id == 0)
    {
        copy_n_vector(N, x0 + N * global_body_id, x_last_local + N * local_body_id);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_matrix_id == 0)
    {
        copy_n_vector(N, x_last_local + N * local_body_id, xres + N * global_body_id);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(i = 0; i < nt; ++i)
    {
        t_iter = i * mm;
        for(j = 0; j < mm; ++j)
        {
            t_cur = tt0 + (t_iter + j) * hh;
            for(jj = 0; jj < N; ++jj)
            {
                parallel_set_elements(a_local + jj * S, 0.0, local_matrix_id);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(local_matrix_id == 0)
            {
                differences[local_body_id] = 1.0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            jjj = 0;
            while(differences[local_body_id] > DBL_EPSILSON)
            {
                make_y_matrix(x_last_local + local_shft, a_local, b_local, y_local, abt_local, local_matrix_id);
                make_f_matrix(y_local, c_local, hh, t_cur, f_local, ftmp_local, local_matrix_id);
                parallel_matrix_multiplication(f_local, matrix_ct_inv_transpose_local, new_a_local, local_matrix_id);
                parallel_calculate_difference(new_a_local, a_local, local_difference_buffer, differences + local_body_id, local_matrix_id);
                for(jj = 0; jj < N; ++jj)
                {
                    parallel_copy_vector(new_a_local + S * jj, a_local + S * jj, local_matrix_id);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if(jjj++ > MAX_STEPS)
                {
                    break;
                }
            }
            make_y_matrix(x_last_local + local_shft, a_local, b_local, y_local, abt_local, local_matrix_id);
            parallel_copy_n_vector(N, y_local + X_IN_Y_SHFT, x_local + local_shft, local_matrix_id, S);
            parallel_copy_n_vector(N, x_local + local_shft, x_last_local + local_shft, local_matrix_id, S);
        }
        parallel_copy_n_vector(N, x_last_local + local_shft, xres + ((i + 1) * NUM_BODIES_IN_WORKRANGE + global_body_id) * N, local_matrix_id, S);
    }
}

void parallel_matrix_multiplication(double* a, double* bt, double* res, int loc_id)
// dim(A) = N x S; dim(B) = S x S; dim(res) = N x S (bt is transposed B)
{
    int i;
    double* b_local = bt + S * loc_id;
    for(i = 0; i < N; ++i)
    {
        res[i * S + loc_id] = cdot(a + i * S, b_local);
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

double cdot(double* u, double* v) // dim(U) = dim(V) = S
{
    int i;
    double res;
    res = 0.0;
    for(i = 0; i < S; ++i)
    {
        res += u[i] * v[i];
    }
    return res;
}

void parallel_copy_vector(const double* orig, double* copy, int loc_id) // dim(orig) = dim(copy) = S
{
    copy[loc_id] = orig[loc_id];
}

void parallel_copy_matrix(const double* orig, double* copy, int loc_id) // dim(orig) = dim(copy) = S x S
{
    const double* cur_orig;
    double* cur_copy;
    int i;
    cur_orig = orig + loc_id * S;
    cur_copy = copy + loc_id * S;
    for(i = 0; i < S; ++i)
    {
        cur_copy[i] = cur_orig[i];
    }
}

void copy_n_vector(int n, const double* orig, double* copy)
{
    int i;
    for(i = 0; i < n; ++i)
    {
        copy[i] = orig[i];
    }
}

void parallel_copy_n_vector(int n, const double* orig, double* copy, int loc_id, int num_threads)
{
    int i;
    if(n <= num_threads)
    {
        if(loc_id < n)
        {
            copy[loc_id] = orig[loc_id];
        }
    }
    else
    {
        if(loc_id == 0)
        {
            for(i = 0; i < n; ++i)
            {
                copy[i] = orig[i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void parallel_set_elements(double* vector, double value, int loc_id) // dim(vector) = S
{
    vector[loc_id] = value;
}

void parallel_calculate_difference(double* a1, double* a2, double* buf, double* res_ptr, int loc_id)
{
    double* a1_cur;
    double* a2_cur;
    int j;
    double res, tmp_res;
    res = 0.0;
    a1_cur = a1 + N * loc_id;
    a2_cur = a2 + N * loc_id;
    for(j = 0; j < N; ++j)
    {
        tmp_res = fabs(a1_cur[j] - a2_cur[j]);
        if(tmp_res > res)
        {
            res = tmp_res;
        }
    }
    buf[loc_id] = res;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(loc_id == 0)
    {
        for(j = 0; j < S; ++j)
        {
            if(buf[j] > res)
            {
                res = buf[j];
            }
        }
        res_ptr[0] = res;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void make_y_matrix(double* y0, double* a, double* bt, double* y, double* abt, int loc_id)
{
    int i;
    parallel_matrix_multiplication(a, bt, abt, loc_id);
    for(i = 0; i < N; ++i)
    {
        y[loc_id * N + i] = y0[i] + abt[i * S + loc_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void make_f_matrix(double* y, double* c, double hh, double t0, double* f, double* ftmp, int loc_id)
{
    int i;
    F(t0 + hh * c[loc_id], y + N * loc_id, ftmp + N * loc_id);
    barrier(CLK_LOCAL_MEM_FENCE);
    for(i = 0; i < N; ++i)
    {
        f[i * S + loc_id] = ftmp[loc_id * N + i] * hh;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
void print_value_in_one_thread(double value)
{
    if(get_global_id(0) == 0)
    {
        printf("%lf\n\n", value);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void print_vector_in_one_thread(int len, double* vector)
{
    int i;
    if(get_global_id(0) == 0)
    {
        for(i = 0; i < len; ++i)
        {
            printf("%lf ", vector[i]);
        }
        printf("\n\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

void print_matrix_in_one_thread(int n, int m, double* matrix)
{
    int i, j;
    if(get_global_id(0) == 0)
    {
        for(i = 0; i < n; ++i)
        {
            for(j = 0; j < m; ++j)
            {
                printf("%lf ", matrix[i * m + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
*/
