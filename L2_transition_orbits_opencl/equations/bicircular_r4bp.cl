#define N 6

void F(double t, double* x, double* res)
{
    double N_L, DELTA, L, MU2;
    double theta, cos_theta, sin_theta, x_e, y_e, z_e, x_m, y_m, z_m, r1sq, r3sq, r4sq, r1cube_inv, r3cube_inv, r4cube_inv, mudelta_r3cube, mu1mdelta_r4cube;
    N_L = 13.36874661478304;
    DELTA = 0.012150584394709708;
    L = 0.00257;
    MU2 = 3.0034805945421924e-06;
    theta = N_L * t;
    cos_theta = native_cos(theta);
    sin_theta = native_sin(theta);
    x_e = 1.0 - MU2 + L * DELTA * cos_theta;
    y_e = L * DELTA * sin_theta;
    z_e = 0.0;
    x_m = 1.0 - MU2 - L * (1.0 - DELTA) * cos_theta;
    y_m = -L * (1.0 - DELTA) * sin_theta;
    z_m = 0.0;
    r1sq = (x[0] + MU2) * (x[0] + MU2) + x[1] * x[1] + x[2] * x[2];
    r3sq = (x[0] - x_m) * (x[0] - x_m) + (x[1] - y_m) * (x[1] - y_m) + (x[2] - z_m) * (x[2] - z_m);
    r4sq = (x[0] - x_e) * (x[0] - x_e) + (x[1] - y_e) * (x[1] - y_e) + (x[2] - z_e) * (x[2] - z_e);
    r1cube_inv = 1.0 / (r1sq * native_sqrt(r1sq));
    r3cube_inv = 1.0 / (r3sq * native_sqrt(r3sq));
    r4cube_inv = 1.0 / (r4sq * native_sqrt(r4sq));
    mudelta_r3cube = MU2 * DELTA * r3cube_inv;
    mu1mdelta_r4cube = MU2 * (1.0 - DELTA) * r4cube_inv;
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    res[3] = 2.0 * x[4] + x[0] - r1cube_inv * (x[0] - MU2) - mudelta_r3cube * (x[0] - x_m) - mu1mdelta_r4cube * (x[0] - x_e);
    res[4] = -2.0 * x[3] + x[1] - r1cube_inv * x[1] - mudelta_r3cube * (x[1] - y_m) - mu1mdelta_r4cube * (x[1] - y_e);
    res[5] = -r1cube_inv * x[2] - mudelta_r3cube * (x[2] - z_m) - mu1mdelta_r4cube * (x[2] - z_e);
}
