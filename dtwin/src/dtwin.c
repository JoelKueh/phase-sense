
#include "render.h"
#include "nbody.h"
#include <stdio.h>
#include <sys/time.h>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;
int pcount = 100;
int num_frames = 200;
int num_videos = 50;

const char *VIDEO_OUT_FNAME = "./out/test.mp4";
const char *META_OUT_FNAME = "./out/test.meta";

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *render_ctx, const char video_path[], const char meta_path[])
{
	struct timeval stop, start;
	nbody_context_t nbody_ctx;
	FILE *meta_output_file;
	float comp_time = 0.0f;
	float rend_time = 0.0f;
	float emergence_idx;
	int result = 0;

	// open the output file for writing
	if (render_open_output(render_ctx, video_path) == -1) {
		fprintf(stderr, "render_open: failed to open %s for writing\n", video_path);
		result = -1;
		goto out;
	}

	// open the metadata output file for writing
	if ((meta_output_file = fopen(meta_path, "w")) == 0) {
		perror("fopen");
		result = -1;
		goto out_close_output;
	}

	// initialize the nbody_simulation with the current parameters
	if (nbody_init(&nbody_ctx, pcount)) {
		fprintf(stderr, "nbody_init: failed\n");
		result = -1;
		goto out_close_meta;
	}

	// render the desired number of frames
	for (int i = 0; i < num_frames; i++) {
		// update the simulation data on the cpu
  		gettimeofday(&start, NULL);
		emergence_idx = nbody_update(&nbody_ctx, 0.2);
		if (fprintf(meta_output_file, "%d,%f\n", i, emergence_idx) < 0) {
			perror("fprintf");
			result = -1;
			goto out_sim_deinit;
		}
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
			goto out_sim_deinit;
		}
		gettimeofday(&stop, NULL);
		rend_time += (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
	}

	// print out total compute and render times
	fprintf(stderr, "Compute/Render Time: %f/%f\n",
	        comp_time / 1000000, rend_time / 1000000);

out_sim_deinit:
	nbody_deinit(&nbody_ctx);

out_close_meta:
	if (fclose(meta_output_file)) {
		result = -1;
	}

out_close_output:
	if (render_close_output(render_ctx) == -1) {
		result = -1;
	}

out:
	return result;
}

int main()
{
	char video_path_buf[128];
	char meta_path_buf[128];
	int result = 0;
	render_context_t render_ctx;

	if (render_init(&render_ctx, screen_width, screen_height, pcount) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}

	for (int i = 0; i < num_videos; i++) {
		fprintf(stderr, "Rendering Video %d/%d\n", i+1, num_videos);
		snprintf(video_path_buf, sizeof(video_path_buf), "./out/ds/%d.mp4", i);
		snprintf(meta_path_buf, sizeof(meta_path_buf), "./out/ds/%d.meta", i);
		if (simulate(&render_ctx, video_path_buf, meta_path_buf) == -1) {
			fprintf(stderr, "simulation error\n");
			result = 1;
			goto out_render_deinit;
		}
	}

out_render_deinit:
	render_deinit(&render_ctx);

out:
	return result;
}
