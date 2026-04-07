#pragma once

#include "param.h"
#include "rand.h"
#include "cglm/cglm.h"

#define MAX_HITBOX_LEN 4

// holds the data for a particle, equivalent to that used in render vbo
typedef struct {
	vec2 pos;
	vec2 vel;
	float rot;
	float rvel;
	int type;
} particle_t;

// holds the data for a particle type, list of vertices
typedef struct {
	float px[MAX_HITBOX_LEN];
	float py[MAX_HITBOX_LEN];
} nbody_hitbox_t;

typedef struct disj_cluster_node_t disj_cluster_node_t;
struct disj_cluster_node_t {
	// disjoint sets book-keeping data
	disj_cluster_node_t *parent; // pointer to the parent of the node
	int rank;                    // distance of the node from the root of the cluster

	// data for the cluster itself
	float mass;   // the mass of the cluster only valid at the center
	vec2 com;     // the center of mass of the particle

	bool updated; // has this cluster been updated this frame
	vec2 vel;     // the velocity of the particle
};

typedef struct {
	rand_state rand_state;
	params_t *params;
	particle_t *pbuf;
	nbody_hitbox_t *ptypebuf;
	disj_cluster_node_t *disj_clusters;
} nbody_context_t;

int nbody_init(nbody_context_t *ctx, params_t *params);
float nbody_update(nbody_context_t *ctx, float dt);
void nbody_deinit(nbody_context_t *ctx);
