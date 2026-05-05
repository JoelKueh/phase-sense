#pragma once

#include <stdint.h>
#include <math.h>
#include <stdbool.h>

typedef uint64_t rand_state;
typedef struct {
	float f1;
	float f2;
} float_pair_t;

// PCG32 prng https://en.wikipedia.org/wiki/Permuted_congruential_generator
#define PCG_INIT_STATE (uint64_t)0x4d595df4d0f33173
#define PCG_MULT (uint64_t)6364136223846793005u
#define PCG_INC (uint64_t)1442695040888963407u
#define M_PI 3.14159265358979323846

static inline uint64_t splitmix64(uint64_t x) {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint32_t rotr32(rand_state x, uint32_t r)
{
	return x >> r | x << (-r & 31);
}

static inline uint32_t pcg32(rand_state *__restrict__ state)
{
	uint64_t x = *state;
	uint32_t count = (uint32_t)(x >> 59);

	*state = x * PCG_MULT + PCG_INC;
	x ^= x >> 18;
	return rotr32((uint32_t)(x >> 27), count);
}

static inline uint32_t rand_u32(rand_state *__restrict__ state)
{
	return pcg32(state);
}

static inline float rand_uniform_float(rand_state *__restrict__ state, float low, float high)
{
	float uniform_float = pcg32(state) / (float)((uint64_t)1 << 32);
	return uniform_float * (high - low) + low;
}

// generate random float pair according to a gaussian distribution using the box-muller transofrm
static inline float_pair_t rand_norm_pair(rand_state *__restrict__ state, double mu, double sigma)
{
	uint32_t u1, u2;
	double f1, f2;

	// generate two random numbers, the first must be nonzero
	do {
		u1 = rand_u32(state);
	} while (u1 == 0);
	u2 = rand_u32(state);

	// convert the random numbers to floats in the range (0,1]
	f1 = (double)u1 / (double)(~(uint32_t)0);
	f2 = (double)u2 / (double)(~(uint32_t)0);

	// compute z0 and z1
	float mag = sigma * sqrt(-2.0 * log(f1));
    float z0  = mag * cos(2.0 * M_PI * f2) + mu;
    float z1  = mag * sin(2.0 * M_PI * f2) + mu;

    return (float_pair_t){.f1 = z0, .f2 = z1};
}

static inline bool rand_event(rand_state *__restrict__ state, double threshold)
{
    const uint32_t max_u32 = ~(uint32_t)0;
    uint32_t u1 = rand_u32(state);
    return u1 > (double)max_u32 * threshold;
}
