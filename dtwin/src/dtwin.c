
#include "render.h"
#include "nbody.h"
#include <stdio.h>
#include <sys/time.h>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;
int pcount = 5000;

const char *OUT_FNAME = "./out/test.mp4";

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *render_ctx, const char path[])
{
	struct timeval stop, start;
	nbody_context_t nbody_ctx;
	float comp_time = 0.0f;
	float rend_time = 0.0f;
	int result = 0;

	// open the output file for writing
	if (render_open_output(render_ctx, path) == -1) {
		fprintf(stderr, "render_open: failed to oupen %s for writing\n", path);
		result = -1;
		goto out;
	}

	// initialize the nbody_simulation with the current parameters
	if (nbody_init(&nbody_ctx, pcount)) {
		fprintf(stderr, "nbody_init: failed\n");
		result = -1;
		goto out_close_output;
	}

	// render the desired number of frames
	for (int i = 0; i < 50; i++) {
		// update the simulation data on the cpu
  		gettimeofday(&start, NULL);
		nbody_update(&nbody_ctx, 0.2);
		gettimeofday(&stop, NULL);
		comp_time += (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;

		// load the data from the host buffer to the gpu
  		gettimeofday(&start, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, render_ctx->particle_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(particle_t) * nbody_ctx.pcount, nbody_ctx.pbuf,
			     GL_DYNAMIC_DRAW);

		// render the frame and pass the data to ffmpeg
		if (render_frame(render_ctx) == -1) {
			fprintf(stderr, "render_frame: failed\n");
			result = -1;
			goto out_close_output;
		}
		gettimeofday(&stop, NULL);
		rend_time += (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
	}

	// print out total compute and render times
	fprintf(stderr, "Compute/Render Time: %f/%f\n",
	        comp_time / 1000000, rend_time / 1000000);

out_sim_deinit:
	nbody_deinit(&nbody_ctx);

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

	if (render_init(&render_ctx, screen_width, screen_height, pcount) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}

	if (simulate(&render_ctx, "./out/test.mp4") == -1) {
		fprintf(stderr, "simulation error\n");
		result = 1;
		goto out_render_deinit;
	}

out_render_deinit:
	render_deinit(&render_ctx);

out:
	return result;
}
