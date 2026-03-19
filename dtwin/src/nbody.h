#pragma once

#include "rand.h"

typedef struct {
	float px;
	float py;
	float vx;
	float vy;
	float rotation;
	int type;
} particle_t;

typedef struct disj_cluster_node_t disj_cluster_node_t;
struct disj_cluster_node_t {
	// disjoint sets book-keeping data
	disj_cluster_node_t *parent; // pointer to the parent of the node
	int rank;                    // distance of the node from the root of the cluster

	// data for the cluster itself
	float mass;              // the mass of the cluster only valid at the center
	float com_x;             // the x position of the center of mass of the cluster
	float com_y;             // the y position of the center of mass of the cluster

	bool updated;            // has this cluster been updated this frame
	float vx;             // the x velocity of the particle this frame
	float vy;             // the y velocity of the particel this frame
};

typedef struct
{
	rand_state rand_state;
	int pcount;
	particle_t *pbuf;
	disj_cluster_node_t *disj_clusters;
} nbody_context_t;

int nbody_init(nbody_context_t *ctx, int pcount);
float nbody_update(nbody_context_t *ctx, float dt);
void nbody_deinit(nbody_context_t *ctx);
