
#include "render.h"
#include "nbody.h"
#include "rand.h"
#include "param.h"
#include <time.h>
#include <stdio.h>
#include <sys/time.h>

const char *VIDEO_OUT_FNAME = "./out/test.mp4";
const char *META_OUT_FNAME = "./out/test.meta";

void gen_spine(rand_state *s, spine_t *spine, int start_idx, int end_idx, float roughness)
{
	float distance, offset;
	int mid_idx;
	vec2 mid;
	
	if (end_idx <= start_idx + 1)
		return;

	mid_idx = (start_idx + end_idx) / 2;
	distance = fabsf(spine->px[end_idx] - spine->px[start_idx]);
	offset = rand_norm_pair(s, 0.0f, distance * roughness).f1;

	mid[0] = (spine->px[start_idx] + spine->px[end_idx]) / 2.0f;
	mid[1] = (spine->py[start_idx] + spine->py[end_idx]) / 2.0f;

	spine->px[mid_idx] = mid[0];
	spine->py[mid_idx] = mid[1] + offset;

	gen_spine(s, spine, start_idx, mid_idx, roughness);
	gen_spine(s, spine, mid_idx, end_idx, roughness);
}

void gen_particles(params_t *params, spine_t *spine_buf, int n)
{
	rand_state s = splitmix64(time(NULL));
	int pidx, vidx;
	float delx, dely;
	float length;

	for (pidx = 0; pidx < n; pidx++) {
		// compute particle spines via repeated midpoint displacement.
		length = rand_uniform_float(&s, params->particle_min_len, params->particle_max_len);

		// generate list of spines by repeated midpoint displacement.
		spine_buf[pidx].px[0] = -length/2.0f;
		spine_buf[pidx].py[0] = 0.0f;
		spine_buf[pidx].px[SPINE_LEN-1] = length/2.0f;
		spine_buf[pidx].py[SPINE_LEN-1] = 0.0f;
		gen_spine(&s, &spine_buf[pidx], 0, SPINE_LEN-1, params->particle_roughness);
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

	// write the metadata header
	fprintf(meta_output_file, "aggr_prob,%f,avg_len,%f,accel,%f,drag,%f,scale,%f,count,%d\n",
	        render_ctx->params->post_onset_aggr,
	        (render_ctx->params->particle_min_len + render_ctx->params->particle_max_len) * 0.5f,
	        render_ctx->params->accel_distr_sig,
	        render_ctx->params->drag_coeff,
	        render_ctx->params->scale, render_ctx->params->particle_cnt);

	// initialize the nbody_simulation with the current parameters
	if (nbody_init(&nbody_ctx, render_ctx->params, render_ctx->spines)) {
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

int handle_args(int argc, char **argv, char *out_dir, params_t *sim_params);

int main(int argc, char **argv)
{

	int result = 0;
	params_t dtwin_params;
	char out_dir[128];
	switch(handle_args(argc, argv, out_dir, &dtwin_params)) {
		case -1:
			fprintf(stderr, "processing arguments failed\n");
			result = 1;
			goto out;
		case -2:
			result = 0;
			goto out;
		default:
	}

	char video_path_buf[128];
	char meta_path_buf[128];
	render_context_t render_ctx;
	spine_t *spines;
	rand_state s;


	s = splitmix64(time(NULL));

	if ((spines = malloc(dtwin_params.num_ptypes * sizeof(spine_t))) == 0) {
		perror("malloc");
		result = 1;
		goto out;
	}

	gen_particles(&dtwin_params, spines, dtwin_params.num_ptypes);
	if (render_init(&render_ctx, &dtwin_params, spines) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}

	for (int i = 0; i < dtwin_params.video_cnt; i++) {
		// randomize a couple of important video parameters
		dtwin_params.post_onset_aggr = rand_uniform_float(&s, 0.2, 0.8);
		dtwin_params.particle_min_len = rand_uniform_float(&s, 0.1, 0.4)
		                                * rand_uniform_float(&s, 0.01, 0.4);
		dtwin_params.particle_max_len = rand_uniform_float(&s, 0.4, 0.8);
		dtwin_params.accel_distr_sig = rand_uniform_float(&s, 0.01, 0.2);
		dtwin_params.raccel_distr_sig = 3 * dtwin_params.accel_distr_sig;
		dtwin_params.drag_coeff = rand_uniform_float(&s, 0.01, 0.2);
		dtwin_params.scale = rand_uniform_float(&s, 0.1, 0.3);
		dtwin_params.particle_cnt =
			(4.0 + rand_uniform_float(&s, 0.0, 3.0)) / (dtwin_params.scale * dtwin_params.scale);
		printf("%f, %d\n", dtwin_params.scale, dtwin_params.particle_cnt);
		
		fprintf(stderr, "Rendering Video %d/%d\n", i+1, dtwin_params.video_cnt);
		snprintf(video_path_buf, sizeof(video_path_buf), "%s/%d.mp4",
		         out_dir, i);
		snprintf(meta_path_buf, sizeof(meta_path_buf), "%s/%d.meta",
		         out_dir, i);
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

int read_sim_params(char *path, params_t *sim_params)
{
	FILE *par_f = fopen(path, "r");
	if (par_f == NULL) {
		perror("couldn't open provided file");
		return -1;
	}

	char line[256];

	while (fgets(line, sizeof(line), par_f)) {
		char name[64];
		double val;
		if (sscanf(line, "%[^ ] = %lf", name, &val) != 2) {
			continue;
		}


		// window size parameters
		if (!strncmp("scale", name, 63)) 
			sim_params->scale = val;
		if (!strncmp("res", name, 63)) 
			sim_params->res = (int) val;

		// simulation parameters
		if (!strncmp("pre_onset_aggr", name, 63)) 
			sim_params->pre_onset_aggr = val;
		if (!strncmp("post_onset_aggr", name, 63)) 
			sim_params->post_onset_aggr = val;
		if (!strncmp("onset_prob", name, 63)) 
			sim_params->onset_prob = val;

		if (!strncmp("particle_roughness", name, 63)) 
			sim_params->particle_roughness = val;
		if (!strncmp("particle_min_len", name, 63)) 
			sim_params->particle_min_len = val;
		if (!strncmp("particle_max_len", name, 63)) 
			sim_params->particle_max_len = val;
		if (!strncmp("particle_min_intensity", name, 63)) 
			sim_params->particle_min_intensity = val;
		if (!strncmp("particle_max_intensity", name, 63)) 
			sim_params->particle_max_intensity = val;
	
		if (!strncmp("accel_distr_mu", name, 63)) 
			sim_params->accel_distr_mu = val;
		if (!strncmp("accel_distr_sig", name, 63)) 
			sim_params->accel_distr_sig = val;
		if (!strncmp("raccel_distr_mu", name, 63)) 
			sim_params->raccel_distr_mu = val;
		if (!strncmp("raccel_distr_sig", name, 63)) 
			sim_params->raccel_distr_sig = val;

		if (!strncmp("drag_coeff", name, 63)) 
			sim_params->drag_coeff = val;
		if (!strncmp("bounce_strength", name, 63)) 
			sim_params->bounce_strength = val;
		if (!strncmp("particle_cnt", name, 63)) 
			sim_params->particle_cnt = (int) val;
		if (!strncmp("num_ptypes", name, 63)) 
			sim_params->num_ptypes = (int) val;

		// video parameters
		if (!strncmp("frame_cnt", name, 63)) 
			sim_params->frame_cnt = (int) val;
		if (!strncmp("fps", name, 63)) 
			sim_params->fps = (int) val;
		if (!strncmp("video_cnt", name, 63)) 
			sim_params->video_cnt = (int) val;

	}
	return 0;
}

int handle_args(int argc, char **argv, char *out_dir, params_t *sim_params)
{
	--argc, ++argv; //discard self-ref arg
	
	params_t my_params =  {
		.scale = 0.15,
		.res = 1200,

		.pre_onset_aggr = 0.0,
		.post_onset_aggr = 0.3,
		.onset_prob = 0.005,

		.particle_roughness = 0.05,
		.particle_min_len = 0.01,
		.particle_max_len = 0.8,
		.particle_min_intensity = 0.5,
		.particle_max_intensity = 1.5,
		
		.accel_distr_mu = 0.0,
		.accel_distr_sig = 0.1,
		.raccel_distr_mu = 0.0,
		.raccel_distr_sig = 0.3,
		.drag_coeff = 0.1,
		.bounce_strength = 0.001,
		.particle_cnt = 200,
		.num_ptypes = 100,

		.frame_cnt = 1500,
		.fps = 60,
		.video_cnt = 300,
	};

	strncpy(out_dir, "./out/ds", 256);
	
	while(argc) {
		if (!strncmp(*argv, "-o", 3)) {
			if (!(--argc)) {
				fprintf(stderr, "incorrect usage\n");
				return -1;
			}
			strncpy(out_dir, *(++argv), 256);
			fprintf(stderr, "changed outdir, its now %s\n", out_dir);
		} else if (!strncmp(*argv, "-p", 3)) {
			if (!(--argc)) {
				fprintf(stderr, "incorrect usage\n");
				return -1;
			}

			read_sim_params(*(++argv), &my_params);			
		} else if (!strncmp(*argv, "-h", 3)) {
			fprintf(stderr, "usage: dtwin [options]\n"
					"-o <directory> 	write videos and metadata to specified directory (default ./out/ds)\n"
					"-p <filename>		use simulation parameters from the specified file\n"
					"-h			print this dialog\n");
			return -2;
		} else {
			fprintf(stderr, "unknown argument\n");
			return -1;
		}

		--argc, ++argv;
	}

	*sim_params = my_params;
	return 0;
}

	
