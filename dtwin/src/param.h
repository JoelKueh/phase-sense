#pragma once

typedef struct {
	// window size parameters
	double scale;
	int res;

	// simulation parameters
	double aggr_prob;
	double accel_distr_mu;
	double accel_distr_sig;
	double mass_vel_scale;
	double vel_decay_rate;
	double bounce_strength;
	int particle_cnt;
	int num_particle_types;

	// video parameters
	int frame_cnt;
	int video_cnt;
	char *out_dir;
} params_t;
