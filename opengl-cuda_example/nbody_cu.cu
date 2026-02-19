#define TILE_SIZE 1024

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_gl_interop.h>

#include "nbody_cu.h"

__device__ float2 ai_from_j(float4 bi, float4 bj, float E2)
{
	float2 d;
	d.x = bj.x - bi.x;
	d.y = bj.y - bi.y;

	//TODO implement collisions?
	//this method ignores collisions and has a softening factor
	//for very close elements so force does not
	//approach infinity
	float d2 = d.x * d.x + d.y * d.y + E2;
	float d6 = d2 * d2 * d2;
	float rd = 1.0f / sqrtf(d6);

	float cont = rd * bj.z;

	float2 a;
	a.x = cont * d.x;
	a.y = cont * d.y;

	return a;
}

__global__ void find_forces(float4 *pos, float2 *accel, int n, float E2)
{
	__shared__ float4 positions[TILE_SIZE];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}
	int stride = blockDim.x;

	float2 ai;
	ai.x = 0;
	ai.y = 0;

	float4 bi = pos[tid];

	for (int i = threadIdx.x; i < n; i += stride) {
		positions[threadIdx.x] = pos[i];
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) {
			float2 pa = ai_from_j(bi, positions[j], E2);
			ai.x += pa.x;
			ai.y += pa.y;
		}

		__syncthreads();
	}

	accel[tid] = ai;
}

__global__ void update(float4 *pos, float2 *accel, float2 *vel, int n, float dt)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	float4 posi = pos[tid];
	float2 veli = vel[tid];
	float2 acceli = accel[tid];

	veli.x += acceli.x * dt;
	veli.y += acceli.y * dt;

	posi.x += veli.x * dt;
	posi.y += veli.y * dt;

	pos[tid] = posi;
	vel[tid] = veli;
}

__global__ void init_rand(float4 *d_pos, float2 *d_vel, int n,
		float minx, float miny, float maxx, float maxy, float maxv, float maxm)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	/*if (tid == 0) {
		float4 pos;
		pos.x = 400;
		pos.y = 300;
		pos.z = 50;
		float2 v;
		v.x = 0;
		v.y = 0;
		d_pos[tid] = pos;
		d_vel[tid] = v;
		return;
	}*/

	curandState_t randstate;
	curand_init(1234, tid, 0, &randstate);
	float rand;
	rand = curand_uniform(&randstate);
	float x = rand * (maxx - minx) + minx;
	rand = curand_uniform(&randstate);
	float y = rand * (maxy - miny) + miny;
	

	float2 basev;

	basev.x = y / 50;
	basev.y = -x / 50;

	rand = curand_uniform(&randstate);
	float vx = rand * maxv * 2 - maxv;
	rand = curand_uniform(&randstate);
	float vy = rand * maxv * 2 - maxv;

	rand = curand_uniform(&randstate);
	float m = rand * maxm;


	float4 pos;
	pos.x = x;
	pos.y = y;
	pos.z = m;
	float2 v;
	v.x = vx + basev.x;
	v.y = vy + basev.y;

	d_pos[tid] = pos;
	d_vel[tid] = v;
}

__global__ void init_galaxy(float4 *d_pos, float2 *d_vel, int n,
		float cx, float cy, float r)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	float minx = cx - r;
	float maxx = cx + r;
	float miny = cy - r/10.f;
	float maxy = cy + r/10.f;
	float maxm = 10;
	float maxv = 1;

	curandState_t randstate;
	curand_init(1234, tid, 0, &randstate);
	float rand;
	rand = curand_uniform(&randstate);
	float x = rand * (maxx - minx) + minx;
	rand = curand_uniform(&randstate);
	float y = rand * (maxy - miny) + miny;
	
	float my_r = sqrt((x - cx) * (x-cx) + (y-cy) * (y-cy));
	float2 basev;

	basev.x = /*(1.f/sqrt(my_r * 10.f))  * (1.f/ my_r) * */ (y-cy) * (1.f / 15.f);
	basev.y = /*(1.f/sqrt(my_r * 10.f))   * (1.f/ my_r) * */ -(x-cx) * (1.f  / 15.f);

	rand = curand_uniform(&randstate);
	float vx = rand * maxv * 2 - maxv;
	rand = curand_uniform(&randstate);
	float vy = rand * maxv * 2 - maxv;

	rand = curand_uniform(&randstate);
	float m = rand * maxm;


	float4 pos;
	pos.x = x;
	pos.y = y;
	pos.z = m;
	float2 v;
	v.x = vx + basev.x;
	v.y = vy + basev.y;

	d_pos[tid] = pos;
	d_vel[tid] = v;
}

extern "C"
void register_gl(void **d_pos_v, int vbo)
{
	cudaGraphicsResource_t vbo_cr;
	cudaGraphicsGLRegisterBuffer(&vbo_cr, vbo, cudaGraphicsRegisterFlagsNone);
	size_t pos_size;
	cudaGraphicsMapResources(1, &vbo_cr);
	cudaGraphicsResourceGetMappedPointer(d_pos_v, &pos_size, vbo_cr);
}

extern "C"
void init_bodies(void *d_pos_v, void **d_accel_v, void **d_vel_v, int n)
{
	float4 *d_pos = (float4 *) d_pos_v;
	float2 **d_accel = (float2 **) d_accel_v;
	float2 **d_vel = (float2 **) d_vel_v;

	//float *h_vel = (float *) malloc(sizeof(float) * n);
	/*for (int i = 0; i < n; i++) {

	}
	*/

	cudaMalloc(d_accel, sizeof(float2) * n);
	cudaMalloc(d_vel, sizeof(float2) * n);
	//cudaMemset(*d_vel, 0, sizeof(float2) *n);
	
	//int tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
	//init_rand<<<tiles, TILE_SIZE>>>(d_pos, *d_vel, n,
	//		-500, -20, 500, 20, 1, 1);

	int tiles = (n/3 + TILE_SIZE - 1) / TILE_SIZE;
	init_galaxy<<<tiles, TILE_SIZE>>>(d_pos, *d_vel, n / 3,
			250, -100, 70);

	init_galaxy<<<tiles, TILE_SIZE>>>(d_pos + n/3, *d_vel + n/3, n / 3,
			-250, -100, 70);

	init_galaxy<<<tiles, TILE_SIZE>>>(d_pos + 2* n/3, *d_vel +2* n/3, n / 3,
			0, 100, 70);
	cudaDeviceSynchronize();
}



extern "C"
void process_bodies(void *d_pos_v, void *d_accel_v, void *d_vel_v, int n, float dt, float E2)
{
	float4 *d_pos = (float4 *) d_pos_v;
	float2 *d_accel = (float2 *) d_accel_v;
	float2 *d_vel = (float2 *) d_vel_v;

	int tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
	find_forces<<<tiles, TILE_SIZE>>>(d_pos, d_accel, n, E2);

	update<<<tiles, TILE_SIZE>>>(d_pos, d_accel, d_vel, n, dt);
	cudaDeviceSynchronize();
}

extern "C"
void free_bodies(void *d_accel, void *d_vel)
{
	cudaFree(d_accel);
	cudaFree(d_vel);
}

/*
__global__ void test_vbo_share_kernel(float4 *d_pos)
{
	int i = threadIdx.x;
	d_pos[i].x = 800 - d_pos[i].x;
}

extern "C"
void test_vbo_share(float4 *d_pos) {
	test_vbo_share_kernel<<<1, 9>>>(d_pos);
}
*/
