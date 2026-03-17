
#include <CL/cl.h>
#include <stdio.h>
#include <time.h>

#include "nbody.h"

#define TILE_SIZE 1024

extern const char cl_nbody_cu[];
extern const int cl_nbody_cu_len;

void nbody_err_cb(const char *error_info, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "nbody: %s\n", error_info);
}

int nbody_ctx_init(nbody_context_t *ctx, int vbo)
{
    cl_platform_id platform_ids[16];
    cl_device_id device_ids[16];
    cl_uint num_platforms;
    cl_uint num_devices;
    int idx;

    char err_buf[1024];
    int buflen = 1024;

	// check for a valid opencl device
	clGetPlatformIDs(16, platform_ids, &num_platforms);
	for (idx= 0; idx < num_platforms; idx++) {
	    clGetDeviceIDs(platform_ids[idx], CL_DEVICE_TYPE_DEFAULT, 16, device_ids, &num_devices);
	    if (num_devices > 0)
	        break;
	}

	if (idx == num_platforms) {
	    fprintf(stderr, "nbody_init: no viable platform\n");
	    return -1;
	}

	// create the opencl context based on the device found above
    cl_context_properties ctx_props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[idx],
        0
    };
    ctx->cl_ctx = clCreateContext(ctx_props, 1, device_ids, nbody_err_cb, NULL, NULL);

    // create a command queue for the context
    cl_command_queue_properties q_props[] = {
        0
    };
    ctx->cl_q = clCreateCommandQueueWithProperties(ctx->cl_ctx, device_ids[0], q_props, NULL);

    // create the program to store our kernels
    const char *cl_src[] = { cl_nbody_cu };
    const size_t cl_len[] = { (size_t)cl_nbody_cu_len };
    ctx->cl_prog = clCreateProgramWithSource(ctx->cl_ctx, 1, cl_src, cl_len, NULL);
    if (clBuildProgram(ctx->cl_prog, 1, &device_ids[0], NULL, NULL, NULL) != CL_SUCCESS) {
        clGetProgramBuildInfo(ctx->cl_prog, device_ids[0], CL_PROGRAM_BUILD_LOG,
                              buflen, err_buf, NULL);
        fprintf(stderr, "clBuildProgram:\n%s\n", err_buf);
    }
    ctx->k_collisions = clCreateKernel(ctx->cl_prog, "collisions", NULL);
    ctx->k_update = clCreateKernel(ctx->cl_prog, "update", NULL);
    ctx->k_accel_walk = clCreateKernel(ctx->cl_prog, "accel_walk", NULL);
    ctx->k_sync_clusters = clCreateKernel(ctx->cl_prog, "sync_clusters", NULL);

    // bind the vbo that is the output of our simulation
    ctx->d_vbo = clCreateFromGLBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, vbo, NULL);

    return 0;
}

int nbody_sim_init(nbody_context_t *ctx, int ppv, int n)
{
    // copy parameters into the context
	ctx->ppv = ppv;
	ctx->n = n;

    // create the buffers that our kernel will act upon
    cl_mem_properties mem_props[] = {
    	0
    };
    ctx->d_accel = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE,
								  sizeof(float) * 2 * n, NULL, NULL);
    ctx->d_coll = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
    ctx->d_sync_flag = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

    // clear the acceleration buffer
    float fill_pattern = 0.0f;
    clEnqueueFillBuffer(ctx->cl_q, ctx->d_accel, &fill_pattern, sizeof(float),
                        0, sizeof(float) * 2 * n, 0, NULL, NULL);

    // populate the coll buffer with ids for each particle
	int *h_coll = (int *) malloc(sizeof(int) * n);
	for (int i = 0; i < n; i++) {
		h_coll[i] = i;
	}
	clEnqueueWriteBuffer(ctx->cl_q, ctx->d_coll, true, 0, sizeof(int) * n, h_coll, 0, NULL, NULL);
	free(h_coll);

	// make sure that all copy and fill operations are done is done before continuing
	clFinish(ctx->cl_q);

	return 0;
}

void nbody_update(nbody_context_t *ctx, float dt)
{
	const size_t tile_size = TILE_SIZE;
	const size_t tiles = (ctx->n + TILE_SIZE - 1) / TILE_SIZE;

	// check for colliisons and update collision heirarhcy
	clSetKernelArg(ctx->k_collisions, 0, sizeof(cl_mem), &ctx->d_vbo);
	clSetKernelArg(ctx->k_collisions, 1, sizeof(cl_mem), &ctx->d_coll);
	clSetKernelArg(ctx->k_collisions, 2, sizeof(cl_mem), &ctx->d_accel);
	clSetKernelArg(ctx->k_collisions, 3, sizeof(int), &ctx->n);
	clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_collisions, 1, NULL,
	                       &tiles, &tile_size, 0, NULL, NULL);

	// synchronize the clusters
	clSetKernelArg(ctx->k_sync_clusters, 0, sizeof(cl_mem), &ctx->d_coll);
	clSetKernelArg(ctx->k_sync_clusters, 1, sizeof(cl_mem), &ctx->d_sync_flag);
	clSetKernelArg(ctx->k_sync_clusters, 2, sizeof(int), &ctx->n);
	clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_sync_clusters, 1, NULL,
	                       &tiles, &tile_size, 0, NULL, NULL);

	// update velocities
	int seed = (uint64_t)rand() << 32 || rand();
	clSetKernelArg(ctx->k_accel_walk, 0, sizeof(cl_mem), &ctx->d_coll);
	clSetKernelArg(ctx->k_accel_walk, 1, sizeof(cl_mem), &ctx->d_accel);
	clSetKernelArg(ctx->k_accel_walk, 2, sizeof(int), &seed);
	clSetKernelArg(ctx->k_accel_walk, 3, sizeof(int), &ctx->n);
	clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_accel_walk, 1, NULL,
	                       &tiles, &tile_size, 0, NULL, NULL);

	// update positions
	clSetKernelArg(ctx->k_sync_clusters, 0, sizeof(cl_mem), &ctx->d_vbo);
	clSetKernelArg(ctx->k_sync_clusters, 1, sizeof(cl_mem), &ctx->d_coll);
	clSetKernelArg(ctx->k_sync_clusters, 2, sizeof(cl_mem), &ctx->d_accel);
	clSetKernelArg(ctx->k_sync_clusters, 3, sizeof(float), &dt);
	clSetKernelArg(ctx->k_sync_clusters, 4, sizeof(int), &ctx->n);
	clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_update, 1, NULL,
	                       &tiles, &tile_size, 0, NULL, NULL);

	// wait for all kernels to complete
	clFinish(ctx->cl_q);
}

void nbody_sim_deinit(nbody_context_t *ctx)
{
	clReleaseMemObject(ctx->d_accel);
	clReleaseMemObject(ctx->d_coll);
}

void nbody_ctx_deinit(nbody_context_t *ctx)
{
	clReleaseKernel(ctx->k_collisions);
	clReleaseKernel(ctx->k_update);
	clReleaseKernel(ctx->k_accel_walk);
	clReleaseKernel(ctx->k_sync_clusters);
	clReleaseProgram(ctx->cl_prog);
	clReleaseCommandQueue(ctx->cl_q);
	clReleaseContext(ctx->cl_ctx);
}
