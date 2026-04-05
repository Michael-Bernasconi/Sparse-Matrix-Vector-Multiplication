#include "my_time_lib.h"

/**
 * Calculates the Arithmetic Mean (Average).
 */
double arithmetic_mean(double *v, int len)
{
    if (len <= 0)
        return 0.0;
    double mu = 0.0;
    for (int i = 0; i < len; i++)
        mu += (double)v[i];
    return mu / (double)len;
}

/**
 * Calculates the Geometric Mean using logarithms to prevent numerical overflow.
 * Note: Only processes values > 0.
 */
double geometric_mean(double *v, int len)
{
    if (len <= 0)
        return 0.0;
    double log_sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        // Logarithm is only defined for positive numbers
        log_sum += (v[i] > 0) ? log((double)v[i]) : 0.0;
    }
    return exp(log_sum / (double)len);
}

/**
 * Calculates the Population Standard Deviation (Sigma).
 * v =  The array of values.
 * mu = The pre-calculated arithmetic mean.
 * len = The number of elements.
 */
double sigma_fn_sol(double *v, double mu, int len)
{
    if (len <= 0)
        return 0.0;
    double sigma = 0.0;
    for (int i = 0; i < len; i++)
    {
        // Sum of squared differences from the mean
        sigma += ((double)v[i] - mu) * ((double)v[i] - mu);
    }
    // Square root converts variance into standard deviation
    return sqrt(sigma / (double)len);
}