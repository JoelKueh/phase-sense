#define TILE_SIZE 1024

typedef struct {
	float px;
	float py;
	float vx;
	float vy;
	float rotation;
	int type;
} particle_t;

__kernel void collisions(__global particle_t *d_vbo, __global int *d_coll,
                         __global void *d_accel, int n);
__kernel void update(__global particle_t *d_vbo, __global int *d_coll,
                     __global void *d_accel, float dt, int n);
__kernel void accel_walk(__global int *d_coll, __global void *d_accel, int cseed, int n);
__kernel void sync_clusters(__global int *d_coll, __global int *flag, int n);

// PCG32 prng https://en.wikipedia.org/wiki/Permuted_congruential_generator
__constant ulong pcg_init_state = 0x4d595df4d0f33173;
__constant ulong pcg_mlt = 6364136223846793005u;
__constant ulong pcg_inc = 1442695040888963407u;

ulong splitmix64(ulong x) {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

uint rotr32(uint x, uint r)
{
	return x >> r | x << (-r & 31);
}

uint pcg32(ulong *restrict state)
{
	ulong x = *state;
	uint count = (uint)(x >> 59);

	*state = x * pcg_mlt + pcg_inc;
	x ^= x >> 18;
	return rotr32((uint)(x >> 27), count);
}

// approximate gaussian by adding together random floats
float rand_float(ulong *restrict state)
{
	const float scale = 1.0f / (float)((ulong)1 << 31);
	float a = (int)pcg32(state) * scale / 3.0f;
	float b = (int)pcg32(state) * scale / 3.0f;
	float c = (int)pcg32(state) * scale / 3.0f;
	return a + b + c;
}

__kernel void accel_walk(__global int *d_coll, __global void *d_accel, int cseed, int n)
{
	int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}
	
	if (((__global int *)d_coll)[tid] != tid) {
		return;
	}

	ulong rand_state = splitmix64(pcg_init_state + cseed + tid);

	//particle_t part_a = ctx.d_vbo[tid];
	//float2 accel_a = ((float2 *)ctx.d_accel)[tid];

	float2 accel_new;
	accel_new.x = rand_float(&rand_state) * 0.1;
	accel_new.y = rand_float(&rand_state) * 0.1;

	((__global float2 *)d_accel)[tid] = accel_new;
}

float distsqr(float ax, float ay, float bx, float by)
{
	return (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
}

__kernel void collisions(__global particle_t *d_vbo, __global int *d_coll,
                         __global void *d_accel, int n)
{
	__local particle_t particles[TILE_SIZE];
	int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}
	int stride = get_local_size(0);

	particle_t part_a = d_vbo[tid];

	//int par_id = (ctx.d_coll)[tid];
	
	for (int i = get_local_id(0); i < n; i += stride) {
		particles[get_local_id(0)] = d_vbo[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int j = 0; j < get_local_size(0); j++) {
			particle_t part_b = particles[j];

			//TODO replace this with a check more accurate to the shape of the particles
			float dist = distsqr(part_a.px, part_a.py, part_b.px, part_b.py);
			if (dist <= 0.005) {
				int bid = (i - get_local_id(0)) + j;
				//larger particle id gets "stuck" to lower id
				if ((d_coll)[tid] > (d_coll)[bid]) {
					(d_coll)[tid] = (d_coll)[bid];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void sync_clusters(__global int *d_coll, __global int *flag, int n)
{	
	int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}

	if (tid == 0) {
		*flag = 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	while (*flag) {
		if (tid == 0) {
			*flag = 0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	
		int expm = (d_coll)[tid];
		int expp = (d_coll)[expm];

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
					expp = (int) atomic_cmpxchg((__global unsigned int *)d_coll + expm,
					                            (unsigned int) expp, (unsigned int) idm);
				}
			} else {
				while (expm > idm) {
					expm = (int) atomic_cmpxchg((__global unsigned int *)d_coll + tid,
					                            (unsigned int) expm, (unsigned int) idm);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void update(__global particle_t *d_vbo, __global int *d_coll,
                     __global void *d_accel, float dt, int n)
{
	int tid = get_global_id(0);
	if (tid >= n) {
		return;
	}

	int pid = d_coll[tid];

	particle_t self = d_vbo[tid];
	particle_t parent = d_vbo[pid];

	float2 acceli = ((__global float2 *) d_accel)[pid];

	self.vx += acceli.x * dt;
	self.vy += acceli.y * dt;

	self.px += parent.vx * dt;
	self.py += parent.vy * dt;

	d_vbo[tid] = self;	
}
