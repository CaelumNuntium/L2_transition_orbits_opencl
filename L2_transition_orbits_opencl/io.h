int read_data(const char* filename, int n, double* x);
int text_write(int N, const char* filename, int nt, int n_threads, int n_bodies, double dt, double* x);
int is_big_endian(void);
double change_byte_order(double);
int binary_write(int N, const char* filename, int nt, int n_threads, int n_bodies, double dt, double* x);
