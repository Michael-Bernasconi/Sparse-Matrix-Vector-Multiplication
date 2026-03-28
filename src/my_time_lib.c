#include "my_time_lib.h"

/**
 * Calculates the Arithmetic Mean (Average).
 * Formula: $\mu = \frac{1}{n} \sum_{i=1}^{n} v_i$
 */
double arithmetic_mean(double *v, int len) {
    if (len <= 0) return 0.0;
    double mu = 0.0;
    for (int i=0; i<len; i++)
        mu += (double)v[i];
    return mu / (double)len;
}

/**
 * Calculates the Geometric Mean using logarithms to prevent numerical overflow.
 * Formula: $\left( \prod_{i=1}^{n} v_i \right)^{\frac{1}{n}} = \exp\left( \frac{1}{n} \sum \ln(v_i) \right)$
 * Note: Only processes values > 0.
 */
double geometric_mean(double *v, int len) {
    if (len <= 0) return 0.0;
    double log_sum = 0.0;
    for (int i=0; i<len; i++) {
        // Logarithm is only defined for positive numbers
        log_sum += (v[i] > 0) ? log((double)v[i]) : 0.0;
    }
    return exp(log_sum / (double)len);
}

/**
 * Calculates the Population Standard Deviation (Sigma).
 * @param v   The array of values.
 * @param mu  The pre-calculated arithmetic mean.
 * @param len The number of elements.
 * Formula: $\sigma = \sqrt{\frac{1}{n} \sum (v_i - \mu)^2}$
 */
double sigma_fn_sol(double *v, double mu, int len) {
    if (len <= 0) return 0.0;
    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        // Sum of squared differences from the mean
        sigma += ((double)v[i] - mu) * ((double)v[i] - mu);
    }
    // Square root converts variance into standard deviation
    return sqrt(sigma / (double)len);
}