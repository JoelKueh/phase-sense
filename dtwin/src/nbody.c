
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "nbody.h"

#define M_PI 3.14159265358979323846
#define MIN_COLLISION_DIST_SQ 0.01
#define AGGREGATION_PROBABILITY 0.5
#define ACCEL_MU 0.0
#define ACCEL_SIGMA 0.05
#define MASS_VEL_SCALE 1.0
#define VEL_DECAY_RATE 0.9

// finds the representetive element of a cluster (performs path compression)
disj_cluster_node_t *disj_cluster_find(disj_cluster_node_t *node)
{
	if (node->parent != node) {
		node->parent = disj_cluster_find(node->parent);
		return node->parent;
	} else {
		return node;
	}
}

// joins the cluster at node_a to the cluster at node_b
// uses the disjoint sets datastructure under the hood, but also maintains
// the total mass and center of mass of each cluster
void disj_cluster_union(disj_cluster_node_t *node_a, disj_cluster_node_t *node_b)
{
	disj_cluster_node_t *temp_node;
	float total_mass;

	node_a = disj_cluster_find(node_a);
	node_b = disj_cluster_find(node_b);

	// check if the inputs are already in the same set
	if (node_a == node_b) {
		return;
	}

	// union by rank, lesser reparented to greater
	if (node_a->rank < node_b->rank) {
		temp_node = node_a;
		node_a = node_b;
		node_b = temp_node;
	}

	// the com of the joined clusters is weighted avg of the com of the individual clusters
	total_mass = node_a->mass + node_b->mass;
	node_a->com_x = (node_a->com_x * node_a->mass + node_b->com_x * node_b->mass) / total_mass;
	node_a->com_y = (node_a->com_y * node_a->mass + node_b->com_y * node_b->mass) / total_mass;
	node_a->mass = total_mass;

	// make node_a the new root
	node_b->parent = node_a;
	if (node_a->rank == node_b->rank) {
		node_a->rank += 1;
	}
}

float dist_sqr(float ax, float ay, float bx, float by)
{
	return (bx - ax) * (bx - ax) + (by - ay) * (by - ay);
}

// handles the collisison of two particles
void handle_collision(nbody_context_t *ctx, int particle_id_a, int particle_id_b)
{
	disj_cluster_node_t *clust_a = &ctx->disj_clusters[particle_id_a];
	disj_cluster_node_t *clust_b = &ctx->disj_clusters[particle_id_b];

	// TODO: Handle the chance that particles do not aggregate (rand_event)
	disj_cluster_union(clust_a, clust_b);
}

// walks through the all particle pairs, computing collisions and interparticle forces
void part_pair_walk(nbody_context_t *ctx)
{
	disj_cluster_node_t *clust_a;
	disj_cluster_node_t *clust_b;
	particle_t *p1;
	particle_t *p2;
	int i, j;
	float dist;

	// test for all distances between particles
	for (i = 0; i < ctx->pcount; i++) {
		for (j = i + 1; j < ctx->pcount; j++) {
			// skip particles that are already in the same cluster
			clust_a = disj_cluster_find(&ctx->disj_clusters[i]);
			clust_b = disj_cluster_find(&ctx->disj_clusters[j]);
			if (clust_a == clust_b) {
				continue;
			}

			p1 = &ctx->pbuf[i];
			p2 = &ctx->pbuf[j];

			// detect collisions and handle them
			dist = dist_sqr(p1->px, p1->py, p2->px, p2->py);
			if (dist <= MIN_COLLISION_DIST_SQ) {
				handle_collision(ctx, i, j);
			}

			// TODO: Interparticle forces?
		}
	}
}

void part_vel_walk(nbody_context_t *ctx, float dt)
{
	disj_cluster_node_t *clust;
	particle_t *part;
	float_pair_t fpair;
	int i;

	// loop over all clusters and set updated flag to 0
	for (i = 0; i < ctx->pcount; i++) {
		ctx->disj_clusters[i].updated = false;
	}

	// loop over all particles and update their positions and velocities
	for (i = 0; i < ctx->pcount; i++) {
		clust = disj_cluster_find(&ctx->disj_clusters[i]);
		part = &ctx->pbuf[i];

		// if the cluster has not been visited before, update velocity
		if (!clust->updated) {
			fpair = rand_norm_pair(&ctx->rand_state, ACCEL_MU, ACCEL_SIGMA);
			clust->vx = (clust->vx * VEL_DECAY_RATE) + fpair.f1 * dt;
			clust->vy = (clust->vy * VEL_DECAY_RATE) + fpair.f2 * dt;
			clust->updated = true;
		}

		// propagate the cluster velocity to the individual particle
		part->vx = clust->vx;
		part->vy = clust->vy;
		part->px += part->vx * dt;
		part->py += part->vy * dt;
	}
}

float emergence_idx(nbody_context_t *ctx)
{
	disj_cluster_node_t *node;
	double total_mass = 0.0;
	int cluster_count = 0;
	int i;

	// Option 1: Walk over all of the clusters and return the average mass
	for (i = 0; i < ctx->pcount; i++) {
		ctx->disj_clusters[i].updated = false;
	}

	for (i = 0; i < ctx->pcount; i++) {
		node = &ctx->disj_clusters[i];
		if (node->updated == false) {
			total_mass += node->mass;
			node->updated = true;
			cluster_count += 1;
		}
	}

	return total_mass / cluster_count;
}

int nbody_init(nbody_context_t *ctx, int pcount)
{
	float_pair_t fpair;

	// initialize the randomizer
	ctx->rand_state = splitmix64(time(NULL));

	// allocate host particle and cluster buffers
	ctx->pcount = pcount;
	if ((ctx->pbuf = malloc(pcount * sizeof(particle_t))) == 0) {
		fprintf(stderr, "nbody_init: out of memory\n");
		return -1;
	}

	if ((ctx->disj_clusters = malloc(pcount * sizeof(disj_cluster_node_t))) == 0) {
		fprintf(stderr, "nbody_init: out of memory\n");
		free(ctx->pbuf);
		return -1;
	}

	// initialize the data in the particle and cluster buffers
	for (int i = 0; i < pcount; i++) {
		// position data belongs to the particle in the cluster
		ctx->pbuf[i].px = rand_uniform_float(&ctx->rand_state, -1.0, 1.0);
		ctx->pbuf[i].py = rand_uniform_float(&ctx->rand_state, -1.0, 1.0);
		ctx->pbuf[i].vx = 0.0f;
		ctx->pbuf[i].vy = 0.0f;
		ctx->pbuf[i].rotation = rand_uniform_float(&ctx->rand_state, -M_PI, M_PI);
		ctx->pbuf[i].type = 0;

		// velocity data belongs to the cluster itself
		ctx->disj_clusters[i].parent = &ctx->disj_clusters[i]; // each particle is a cluster
		ctx->disj_clusters[i].rank = 0; // clusters rank 0
		ctx->disj_clusters[i].mass = 1.0f; // clusters mass 1
		ctx->disj_clusters[i].com_x = 0.0f; // clusters com is (0, 0)
		ctx->disj_clusters[i].com_y = 0.0f; // clusters com is (0, 0)
		ctx->disj_clusters[i].updated = false; // redundant (see part_vel_walk)
		ctx->disj_clusters[i].vx = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
		ctx->disj_clusters[i].vy = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
	}

	return 0;
}

float nbody_update(nbody_context_t *ctx, float dt)
{
	part_pair_walk(ctx);
	part_vel_walk(ctx, dt);
	return emergence_idx(ctx);
}

void nbody_deinit(nbody_context_t *ctx)
{
	free(ctx->pbuf);
	free(ctx->disj_clusters);
}
