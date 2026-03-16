#define TILE_SIZE 1024

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_gl_interop.h>
#include <time.h>

#include "nbody_cu.h"


__global__ void collisions(cu_context_t ctx);
__global__ void update(cu_context_t ctx, float dt);
__global__ void accel_walk(cu_context_t ctx, int cseed);
__global__ void sync_clusters(cu_context_t ctx, int *flag);

extern "C"
void cuda_update(cu_context_t ctx, float dt)
{

	int tiles = (ctx.n + TILE_SIZE - 1) / TILE_SIZE;



	collisions<<<tiles, TILE_SIZE>>>(ctx);
	int *sync_flag;
	cudaMalloc(&sync_flag, sizeof(int));
	sync_clusters<<<tiles, TILE_SIZE>>>(ctx, sync_flag);
	cudaDeviceSynchronize();
	cudaFree(sync_flag);
	int cseed = clock();
	accel_walk<<<tiles, TILE_SIZE>>>(ctx, cseed);
	update<<<tiles, TILE_SIZE>>>(ctx, dt);
	cudaDeviceSynchronize();
}

extern "C"
cu_context_t register_gl(int vbo, int ppv, int n)
{

	cu_context_t ret;

	ret.ppv = ppv;
	ret.n = n;

	cudaGraphicsResource_t vbo_cr;
	cudaGraphicsGLRegisterBuffer(&vbo_cr, vbo, cudaGraphicsRegisterFlagsNone);

	size_t pos_size;
	cudaGraphicsMapResources(1, &vbo_cr);

	cudaGraphicsResourceGetMappedPointer((void **) &(ret.d_vbo), &pos_size, vbo_cr);


	cudaMalloc((void **) &(ret.d_accel), sizeof(float) * 2 * n);
	cudaMemset((void *) ret.d_accel, 0, sizeof(float) * 2 * n);

	int *h_coll = (int *) malloc(sizeof(int) * n);
	cudaMalloc(&ret.d_coll, sizeof(int) * n);
	for (int i = 0; i < n; i++) {
		h_coll[i] = i;
	}
	cudaMemcpy(ret.d_coll, (void *) h_coll, n * sizeof(int), cudaMemcpyHostToDevice);
	

	return ret;
}


extern "C"
void cuda_free(cu_context_t ctx)
{
	cudaFree(ctx.d_accel);
}

/*
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
*/

__global__ void accel_walk(cu_context_t ctx, int cseed)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= ctx.n) {
		return;
	}

	
	if (((int *)ctx.d_coll)[tid] != tid) {
		return;
	}
	

	curandState_t randstate;
	curand_init(cseed, tid, 0, &randstate);


	//particle_t part_a = ctx.d_vbo[tid];
	//float2 accel_a = ((float2 *)ctx.d_accel)[tid];

	float2 accel_new;
	accel_new.x = 0;
	accel_new.y = 0;

	accel_new = curand_normal2(&randstate);

	accel_new.x *= 0.1;
	accel_new.y *= 0.1;

	((float2 *)ctx.d_accel)[tid] = accel_new;

}

__device__ float distsqr(float ax, float ay, float bx, float by)
{
	return (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
}

__global__ void collisions(cu_context_t ctx)
{

	__shared__ particle_t particles[TILE_SIZE];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= ctx.n) {
		return;
	}
	int stride = blockDim.x;

	particle_t part_a = ctx.d_vbo[tid];

	//int par_id = (ctx.d_coll)[tid];
	
	for (int i = threadIdx.x; i < ctx.n; i += stride) {
		particles[threadIdx.x] = ctx.d_vbo[i];
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) {

			particle_t part_b = particles[j];

			//TODO replace this with a check more accurate to the shape of the particles
			float dist = distsqr(part_a.px, part_a.py, part_b.px, part_b.py);
			if (dist <= 0.005) {
				int bid = (i - threadIdx.x) + j;
				//larger particle id gets "stuck" to lower id
				if ((ctx.d_coll)[tid] > (ctx.d_coll)[bid]) {
					(ctx.d_coll)[tid] = (ctx.d_coll)[bid];
				}
			}
		}

		__syncthreads();
	}
}

__global__ void sync_clusters(cu_context_t ctx, int *flag)
{	
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= ctx.n) {
		return;
	}

	if (tid == 0) {
		*flag = 1;
	}
	__syncthreads();

	while (*flag) {
		if (tid == 0) {
			*flag = 0;
		}
		__syncthreads();
	
		int expm = (ctx.d_coll)[tid];
		int expp = (ctx.d_coll)[expm];

		/*
		 * honestly, this sucks in terms of parallelism model
		 * this synchronization would be easier to perform sequentially
		 * but i kinda assume the overhead of copying the whole vbo
		 * would make that not worth this sorta messy fight for the
		 * collision information
		 */
		if (expm != expp) {
			*flag = 1; //this shouldnt be a race condition
			int idm = min(expm, expp);
			if (expm < expp) {
				while (expp > idm) {
					expp = (int) atomicCAS((unsigned int *)ctx.d_coll + expm, (unsigned int) expp, (unsigned int) idm);
				}
			} else {
				while (expm > idm) {
					expm = (int) atomicCAS((unsigned int *)ctx.d_coll + tid, (unsigned int) expm, (unsigned int) idm);
				}
			}

		}
		__syncthreads();
	}
}

__global__ void update(cu_context_t ctx, float dt)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= ctx.n) {
		return;
	}

	int pid = ctx.d_coll[tid];

	particle_t self = ctx.d_vbo[tid];
	particle_t parent = ctx.d_vbo[pid];

	float2 acceli = ((float2 *) ctx.d_accel)[pid];

	self.vx += acceli.x * dt;
	self.vy += acceli.y * dt;

	self.px += parent.vx * dt;
	self.py += parent.vy * dt;

	ctx.d_vbo[tid] = self;	
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
