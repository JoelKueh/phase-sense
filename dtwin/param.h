#pragma once

typedef struct {
	// window size parameters
	double scale;
	int res_x;
	int res_y;

	// simulation parameters
	double aggr_prob;
	double accel_distr_mu;
	double accel_distr_sig;
	double mass_vel_scale;
	double vel_decay_rate;

	// video parameters
	int frame_count;
	int video_count;
	char *out_dir;
} params_t;
