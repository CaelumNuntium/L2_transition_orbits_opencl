#define N 6

void F(double t, double* x, double* res)
{
    double MU2;
    double x_e, y_e, z_e, r1sq, r4sq, r1cube_inv, r4cube_inv, mu1mdelta_r4cube;
    MU2 = 3.0034805945421924e-06;
    x_e = 1.0 - MU2;
    y_e = 0.0;
    z_e = 0.0;
    r1sq = (x[0] + MU2) * (x[0] + MU2) + x[1] * x[1] + x[2] * x[2];
    r4sq = (x[0] - x_e) * (x[0] - x_e) + (x[1] - y_e) * (x[1] - y_e) + (x[2] - z_e) * (x[2] - z_e);
    r1cube_inv = 1.0 / (r1sq * native_sqrt(r1sq));
    r4cube_inv = 1.0 / (r4sq * native_sqrt(r4sq));
    mu1mdelta_r4cube = MU2 * r4cube_inv;
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    res[3] = 2.0 * x[4] + x[0] - r1cube_inv * (x[0] - MU2) - mu1mdelta_r4cube * (x[0] - x_e);
    res[4] = -2.0 * x[3] + x[1] - r1cube_inv * x[1] - mu1mdelta_r4cube * (x[1] - y_e);
    res[5] = -r1cube_inv * x[2] - mu1mdelta_r4cube * (x[2] - z_e);
}
