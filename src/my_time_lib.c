#include "my_time_lib.h"

double arithmetic_mean(double *v, int len) {
    if (len <= 0) return 0.0;
    double mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (double)v[i];
    return mu / (double)len;
}

double geometric_mean(double *v, int len) {
    if (len <= 0) return 0.0;
    double log_sum = 0.0;
    for (int i=0; i<len; i++) {
        log_sum += (v[i] > 0) ? log((double)v[i]) : 0.0;
    }
    return exp(log_sum / (double)len);
}

double sigma_fn_sol(double *v, double mu, int len) {
    if (len <= 0) return 0.0;
    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        sigma += ((double)v[i] - mu) * ((double)v[i] - mu);
    }
    return sqrt(sigma / (double)len);  //sqrt to obtain dev stand
}