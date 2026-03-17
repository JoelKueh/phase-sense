
#include <EGL/egl.h>
#include <stdio.h>
#include <time.h>

#include "nbody.h"

#define TILE_SIZE 1024
#define CL_ERR_PANIC(err)                                                        \
	if (err != CL_SUCCESS) {                                                     \
		fprintf(stderr, "opencl error: %d at line %d\n", err, __LINE__);         \
		exit(1);                                                                 \
	}

extern const char cl_nbody_cu[];
extern const int cl_nbody_cu_len;

void CL_CALLBACK nbody_err_cb(const char *error_info, const void *private_info, size_t cb,
			      void *user_data)
{
	fprintf(stderr, "nbody: %s\n", error_info);
}

int nbody_ctx_init(nbody_context_t *ctx)
{
	cl_platform_id platform_ids[16];
	cl_device_id device_ids[16];
	cl_uint num_platforms;
	cl_uint num_devices;
	cl_int errcode;
	int idx;

	char err_buf[1024];
	int buflen = 1024;

	// check for a valid opencl device
	clGetPlatformIDs(16, platform_ids, &num_platforms);
	for (idx = 0; idx < num_platforms; idx++) {
		clGetDeviceIDs(platform_ids[idx], CL_DEVICE_TYPE_DEFAULT, 16, device_ids,
			       &num_devices);
		if (num_devices > 0)
			break;
	}

	if (idx == num_platforms) {
		fprintf(stderr, "nbody_init: no viable platform\n");
		return -1;
	}

	// create the opencl context based on the device found above
	cl_context_properties ctx_props[] = { CL_GL_CONTEXT_KHR,
					      (cl_context_properties)eglGetCurrentContext(),
					      CL_EGL_DISPLAY_KHR,
					      (cl_context_properties)eglGetCurrentDisplay(),
					      CL_CONTEXT_PLATFORM,
					      (cl_context_properties)platform_ids[idx],
					      0 };
	ctx->cl_ctx = clCreateContext(ctx_props, 1, device_ids, nbody_err_cb, NULL, &errcode);
	CL_ERR_PANIC(errcode);

	// create a command queue for the context
	cl_command_queue_properties q_props[] = { 0 };
	ctx->cl_q =
		clCreateCommandQueueWithProperties(ctx->cl_ctx, device_ids[0], q_props, &errcode);
	CL_ERR_PANIC(errcode);

	// create the program to store our kernels
	const char *cl_src[] = { cl_nbody_cu };
	const size_t cl_len[] = { (size_t)cl_nbody_cu_len };
	ctx->cl_prog = clCreateProgramWithSource(ctx->cl_ctx, 1, cl_src, cl_len, &errcode);
	CL_ERR_PANIC(errcode);
	if (clBuildProgram(ctx->cl_prog, 1, &device_ids[0], NULL, NULL, NULL) != CL_SUCCESS) {
		clGetProgramBuildInfo(ctx->cl_prog, device_ids[0], CL_PROGRAM_BUILD_LOG, buflen,
				      err_buf, NULL);
		fprintf(stderr, "clBuildProgram:\n%s\n", err_buf);
	}
	ctx->k_collisions = clCreateKernel(ctx->cl_prog, "collisions", &errcode);
	CL_ERR_PANIC(errcode);
	ctx->k_update = clCreateKernel(ctx->cl_prog, "update", NULL);
	CL_ERR_PANIC(errcode);
	ctx->k_accel_walk = clCreateKernel(ctx->cl_prog, "accel_walk", &errcode);
	CL_ERR_PANIC(errcode);
	ctx->k_sync_clusters = clCreateKernel(ctx->cl_prog, "sync_clusters", &errcode);
	CL_ERR_PANIC(errcode);

	return 0;
}

int nbody_sim_init(nbody_context_t *ctx, int vbo, int ppv, int n)
{
	cl_int errcode;

	// copy parameters into the context
	ctx->ppv = ppv;
	ctx->n = n;

	// create the buffers that our kernel will act upon
	cl_mem_properties mem_props[] = { 0 };
	ctx->d_accel = clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, sizeof(float) * 2 * n, NULL,
				      &errcode);
	CL_ERR_PANIC(errcode);
	ctx->d_coll =
		clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, &errcode);
	CL_ERR_PANIC(errcode);
	ctx->d_sync_flag =
		clCreateBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &errcode);
	CL_ERR_PANIC(errcode);

	// bind the shared vbo as an output buffer
	ctx->d_vbo = clCreateFromGLBuffer(ctx->cl_ctx, CL_MEM_READ_WRITE, vbo, &errcode);
	CL_ERR_PANIC(errcode);

	// clear the acceleration buffer
	float fill_pattern = 0.0f;
	CL_ERR_PANIC(clEnqueueFillBuffer(ctx->cl_q, ctx->d_accel, &fill_pattern, sizeof(float), 0,
					 sizeof(float) * 2 * n, 0, NULL, NULL));

	// populate the coll buffer with ids for each particle
	int *h_coll = (int *)malloc(sizeof(int) * n);
	for (int i = 0; i < n; i++) {
		h_coll[i] = i;
	}
	CL_ERR_PANIC(clEnqueueWriteBuffer(ctx->cl_q, ctx->d_coll, true, 0, sizeof(int) * n, h_coll,
					  0, NULL, NULL));
	free(h_coll);

	// make sure that all copy and fill operations are done is done before continuing
	CL_ERR_PANIC(clFinish(ctx->cl_q));

	return 0;
}

void nbody_update(nbody_context_t *ctx, float dt)
{
	const size_t local_size = TILE_SIZE;
	const size_t work_groups = (ctx->n + TILE_SIZE - 1) / TILE_SIZE;
	const size_t global_size = ctx->n * local_size;

	cl_int errcode;

	// aquire the vbo so we can use it
	CL_ERR_PANIC(clEnqueueAcquireGLObjects(ctx->cl_q, 1, &ctx->d_vbo, 0, NULL, NULL));

	// check for colliisons and update collision heirarhcy
	CL_ERR_PANIC(clSetKernelArg(ctx->k_collisions, 0, sizeof(cl_mem), &ctx->d_vbo));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_collisions, 1, sizeof(cl_mem), &ctx->d_coll));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_collisions, 2, sizeof(cl_mem), &ctx->d_accel));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_collisions, 3, sizeof(int), &ctx->n));
	CL_ERR_PANIC(clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_collisions, 1, NULL, &global_size,
					    &local_size, 0, NULL, NULL));

	// synchronize the clusters
	CL_ERR_PANIC(clSetKernelArg(ctx->k_sync_clusters, 0, sizeof(cl_mem), &ctx->d_coll));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_sync_clusters, 1, sizeof(cl_mem), &ctx->d_sync_flag));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_sync_clusters, 2, sizeof(int), &ctx->n));
	CL_ERR_PANIC(clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_sync_clusters, 1, NULL, &global_size,
					    &local_size, 0, NULL, NULL));

	// update velocities
	int seed = (uint64_t)rand() << 32 || rand();
	CL_ERR_PANIC(clSetKernelArg(ctx->k_accel_walk, 0, sizeof(cl_mem), &ctx->d_coll));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_accel_walk, 1, sizeof(cl_mem), &ctx->d_accel));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_accel_walk, 2, sizeof(int), &seed));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_accel_walk, 3, sizeof(int), &ctx->n));
	CL_ERR_PANIC(clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_accel_walk, 1, NULL, &global_size,
					    &local_size, 0, NULL, NULL));

	// update positions
	CL_ERR_PANIC(clSetKernelArg(ctx->k_update, 0, sizeof(cl_mem), &ctx->d_vbo));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_update, 1, sizeof(cl_mem), &ctx->d_coll));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_update, 2, sizeof(cl_mem), &ctx->d_accel));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_update, 3, sizeof(float), &dt));
	CL_ERR_PANIC(clSetKernelArg(ctx->k_update, 4, sizeof(int), &ctx->n));
	CL_ERR_PANIC(clEnqueueNDRangeKernel(ctx->cl_q, ctx->k_update, 1, NULL, &global_size,
					    &local_size, 0, NULL, NULL));

	// release the vbo so OpenGL can draw it
	CL_ERR_PANIC(clEnqueueReleaseGLObjects(ctx->cl_q, 1, &ctx->d_vbo, 0, NULL, NULL));

	// wait for all kernels to complete
	CL_ERR_PANIC(clFinish(ctx->cl_q));
}

void nbody_sim_deinit(nbody_context_t *ctx)
{
	clReleaseMemObject(ctx->d_accel);
	clReleaseMemObject(ctx->d_coll);
	clReleaseMemObject(ctx->d_sync_flag);
	clReleaseMemObject(ctx->d_vbo);
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
