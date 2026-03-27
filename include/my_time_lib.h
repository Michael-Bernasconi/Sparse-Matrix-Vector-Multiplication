#ifndef MY_TIME_LIB_H
#define MY_TIME_LIB_H
#include <sys/time.h>
#include <math.h>
#include <stdio.h>

#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

#define TIMER_PRINT(n) \
    do { \
        printf("Timer elapsed: %lfs\n", TIMER_ELAPSED(n)/1e6); \
        fflush(stdout); \
    } while (0);

double geometric_mean(double *v, int len);
double arithmetic_mean(double *v, int len);
double sigma_fn_sol(double *v, double mu, int len);

#endif