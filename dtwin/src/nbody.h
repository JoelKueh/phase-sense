#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_egl.h>
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <GL/glx.h>
#include <CL/cl.h>

typedef struct {
	float px;
	float py;
	float vx;
	float vy;
	float rotation;
	int type;
} particle_t;

typedef struct
{
	cl_context cl_ctx;
	cl_command_queue cl_q;
	cl_program cl_prog;

	cl_kernel k_collisions;
	cl_kernel k_update;
	cl_kernel k_accel_walk;
	cl_kernel k_sync_clusters;
	
	int ppv;
	int n;
	cl_mem d_vbo;
	cl_mem d_accel;
	cl_mem d_coll;
	cl_mem d_sync_flag;
} nbody_context_t;

int nbody_ctx_init(nbody_context_t *ctx);
int nbody_sim_init(nbody_context_t *ctx, int vbo, int ppv, int n);
void nbody_update(nbody_context_t *ctx, float dt);
void nbody_sim_deinit(nbody_context_t *ctx);
void nbody_ctx_deinit(nbody_context_t *ctx);
