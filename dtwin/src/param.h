#pragma once
#include <stdio.h>
#include <string.h>

typedef struct {
	// window size parameters
	double scale;
	int res;

	// simulation parameters
	double pre_onset_aggr;
	double post_onset_aggr;
	double onset_prob;

	double particle_roughness;
	double particle_min_len;
	double particle_max_len;
	double particle_min_intensity;
	double particle_max_intensity;
	
	double accel_distr_mu;
	double accel_distr_sig;
	double raccel_distr_mu;
	double raccel_distr_sig;

	double drag_coeff;
	double bounce_strength;
	int particle_cnt;
	int num_ptypes;

	// video parameters
	int frame_cnt;
	int fps;
	int video_cnt;
} params_t;
