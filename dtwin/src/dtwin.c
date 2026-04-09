
#include "render.h"
#include "nbody.h"
#include "rand.h"
#include "param.h"
#include <time.h>
#include <stdio.h>
#include <sys/time.h>

const char *VIDEO_OUT_FNAME = "./out/test.mp4";
const char *META_OUT_FNAME = "./out/test.meta";

void gen_particles(nbody_hitbox_t *hbox_buf, render_spine_t *spine_buf, int n)
{
	rand_state s = splitmix64(time(NULL));
	int pidx, vidx;
	float delx, dely;

	for (pidx = 0; pidx < n; pidx++) {
		// random spine of points
		spine_buf[pidx].px[0] = -0.3f;
		spine_buf[pidx].py[0] = 0.0f;
		for (vidx = 1; vidx < MAX_SPINE_LEN-1; vidx++) {
			delx = rand_norm_pair(&s, 0.0f, 0.02).f1;
			dely = rand_norm_pair(&s, 0.0f, 0.02).f2;
			spine_buf[pidx].px[vidx] = -0.3f + 0.6f * vidx / (MAX_SPINE_LEN-1) + delx;
			spine_buf[pidx].py[vidx] = spine_buf[pidx].py[vidx] + dely;
		}
		spine_buf[pidx].px[vidx] = 0.3f;
		spine_buf[pidx].py[vidx] = 0.0f;

		// dumb hitbox
		hbox_buf[pidx].px[0] = -0.3f;
		hbox_buf[pidx].py[0] = 0.0f;
		hbox_buf[pidx].px[1] = 1.0f;
		hbox_buf[pidx].py[1] = 0.0f;
	}
}

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
	if (nbody_init(&nbody_ctx, render_ctx->params)) {
		fprintf(stderr, "nbody_init: failed\n");
		result = -1;
		goto out_close_meta;
	}

	// render the desired number of frames
	for (int i = 0; i < render_ctx->params->frame_cnt; i++) {
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
		glBufferData(GL_ARRAY_BUFFER, sizeof(particle_t) * render_ctx->params->particle_cnt,
		             nbody_ctx.pbuf, GL_DYNAMIC_DRAW);

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
	render_spine_t *spines;
	nbody_hitbox_t *hboxes;
	params_t dtwin_params = {
		.scale = 0.35,
		.res = 1200,

		.pre_onset_aggr = 0.0,
		.post_onset_aggr = 0.05,
		.onset_prob = 0.01,
		
		.accel_distr_mu = 0.0,
		.accel_distr_sig = 0.08,
		.raccel_distr_mu = 0.0,
		.raccel_distr_sig = 0.15,
		.drag_coeff = 0.1,
		.bounce_strength = 0.1,
		.particle_cnt = 250,
		.num_ptypes = 100,

		.frame_cnt = 2000,
		.fps = 60,
		.video_cnt = 1,
		.out_dir = "./out/ds"
	};

	if ((spines = malloc(dtwin_params.num_ptypes * sizeof(render_spine_t))) == 0) {
		perror("malloc");
		result = 1;
		goto out;
	}

	if ((hboxes = malloc(dtwin_params.num_ptypes * sizeof(nbody_hitbox_t))) == 0) {
		perror("malloc");
		result = 1;
		goto out;
	}

	gen_particles(hboxes, spines, dtwin_params.num_ptypes);
	if (render_init(&render_ctx, &dtwin_params, spines) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}

	for (int i = 0; i < dtwin_params.video_cnt; i++) {
		fprintf(stderr, "Rendering Video %d/%d\n", i+1, dtwin_params.video_cnt);
		snprintf(video_path_buf, sizeof(video_path_buf), "%s/%d.mp4",
		         dtwin_params.out_dir, i);
		snprintf(meta_path_buf, sizeof(meta_path_buf), "%s/%d.meta",
		         dtwin_params.out_dir, i);
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
