int resolve_string(const char* string, char* parameter, char* value);
int read_config(const char* filename, int num_params, const char** parameters, char** values, int* is_required);
int rewrite_config(const char* filename, int nparam, const char** parameters, const char** values);
