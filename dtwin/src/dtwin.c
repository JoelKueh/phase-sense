
#include "render.h"
#include "nbody.h"
#include <stdio.h>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;

const char *OUT_FNAME = "./out/test.mp4";

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *render_ctx, nbody_context_t *nbody_ctx, const char path[])
{
	particle_t buf[50];
	int result = 0;

	// open the output file for writing
	if (render_open_output(render_ctx, path) == -1) {
		fprintf(stderr, "render_open: failed to oupen %s for writing\n", path);
		result = -1;
		goto out;
	}

	// create an initial buffer of particle positions as data
	for (int i = 0; i < 50; i++) {
		buf[i].px = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
		buf[i].py = -1.0 + 2.0 * (float)rand() / (float)RAND_MAX;
		buf[i].vx = 0.f;
		buf[i].vy = 0.f;
		buf[i].rotation = (float)rand() / (float)RAND_MAX * 2 * 3.141592;
		buf[i].type = 0;
		// buf[i].position.x = 0.5;
		// buf[i].position.y = 0.5;
		// buf[i].velocity.x = 0.5;
		// buf[i].velocity.y = 0.5;
		// buf[i].rotation = 0.25;
		// buf[i].type = 0;
	}

	// load the data from the host buffer to the gpu
    glBindBuffer(GL_ARRAY_BUFFER, render_ctx->particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(buf), buf, GL_DYNAMIC_DRAW);

    // initialize the nbody_simulation with the current parameters
    if (nbody_sim_init(nbody_ctx, render_ctx->particle_vbo, 6, 50)) {
    	fprintf(stderr, "nbody_sim_init: failed\n");
    	result = -1;
    	goto out_close_output;
    }

    // render the desired number of frames
    for (int i = 0; i < 50; i++) {
		nbody_update(nbody_ctx, 0.2);
		if (render_frame(render_ctx) == -1) {
			fprintf(stderr, "render_frame: failed\n");
			result = -1;
			goto out_close_output;
		}
    }

out_sim_deinit:
	nbody_sim_deinit(nbody_ctx);

out_close_output:
	if (render_close_output(render_ctx) == -1) {
		result = -1;
	}

out:
	return result;
}

int main()
{
	int result = 0;
	render_context_t render_ctx;
	nbody_context_t nbody_ctx;

	if (render_init(&render_ctx, screen_width, screen_height) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}

	if (nbody_ctx_init(&nbody_ctx)) {
		fprintf(stderr, "nbody context initialization failed\n");
		result = 1;
		goto out_render_deinit;
	}
	
	if (simulate(&render_ctx, &nbody_ctx, "./out/test.mp4")  == -1) {
		fprintf(stderr, "simulation error\n");
		result = 1;
		goto out_nbody_deinit;
	}

out_nbody_deinit:
	nbody_ctx_deinit(&nbody_ctx);

out_render_deinit:
	render_deinit(&render_ctx);

out:
	return result;
}

