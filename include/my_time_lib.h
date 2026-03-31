#ifndef MY_TIME_LIB_H
#define MY_TIME_LIB_H

#include <sys/time.h>
#include <math.h>
#include <stdio.h>

/**
 * Defines two timeval structures to store the start and end timestamps.
 * 'n' is the unique identifier for the timer instance.
 */
#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}

/**
 * Captures the current system time and stores it in the start variable.
 */
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)

/**
 * Captures the current system time and stores it in the stop variable.
 */
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)

/**
 * Calculates the elapsed time in microseconds (us).
 * Formula: (Seconds_diff * 1,000,000) + Microseconds_diff.
 */
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

/**
 * Prints the elapsed time converted to seconds (s) to the standard output.
 * Uses a 'do-while(0)' block to ensure macro safety in conditional statements.
 */
#define TIMER_PRINT(n) \
    do { \
        printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6); \
        fflush(stdout); \
    } while (0);

/* --- Statistical Function Prototypes --- */
/** Returns the geometric mean of an array 'v' of length 'len' */
double geometric_mean(double *v, int len);
/** Returns the arithmetic mean of an array 'v' of length 'len' */
double arithmetic_mean(double *v, int len);
/** Returns the standard deviation (sigma) given an array 'v' and its mean 'mu' */
double sigma_fn_sol(double *v, double mu, int len);

#endif