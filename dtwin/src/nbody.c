
#include "cglm/vec3.h"
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
#define DRAG_COEFF 0.1
#define BOUNCE_STRENGTH 0.05

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

	// union by rank, lesser reparented to greater, greater is now in node_a
	if (node_a->rank < node_b->rank) {
		temp_node = node_a;
		node_a = node_b;
		node_b = temp_node;
	}

	// total mass is the sum of the masses of the clusters
	total_mass = node_a->mass + node_b->mass;

	// new velocity is weighted sum of previous velocities
	node_a->vel[0] = (node_a->mass * node_a->vel[0] + node_b->mass * node_b->vel[0]) / total_mass;
	node_a->vel[1] = (node_a->mass * node_a->vel[1] + node_b->mass * node_b->vel[1]) / total_mass;

	// the com of the joined clusters is weighted avg of the com of the individual clusters
	node_a->com[0] = (node_a->com[0] * node_a->mass + node_b->com[0] * node_b->mass) / total_mass;
	node_a->com[1] = (node_a->com[1] * node_a->mass + node_b->com[1] * node_b->mass) / total_mass;
	node_a->mass = total_mass;

	// TODO: rotational velocity update would require rotational inertia

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

// detects intersection between two lines
// returns parameterized points in t and u
// t and u range from 0 to 1.
bool line_intersect(vec2 p1, vec2 p2, vec2 p3, vec2 p4, vec2 intersect)
{
	vec2 r, s, del, offset;

	glm_vec2_sub(p2, p1, r);
	glm_vec2_sub(p4, p3, s);
	glm_vec2_sub(p3, p1, del);

	float rxs = r[0] * s[1] - r[1] * s[0];
	float delxr = del[0] * r[1] - del[1] * r[0];
	float delxs = del[0] * s[1] - del[1] * s[0];

	// lines are parallel if cross product is zero
	if (fabs(rxs) < 1e-12f) {
		return false;
	}

	float t = delxs / rxs;
	float u = delxr / rxs;

	// lines do not intersect if parameterized t and u are out of range
	if (t < 0.0f || t > 1.0f || u < 0.0f || u > 1.0f) {
		return false;
	}

	// compute intersection point as offset from p1
	glm_vec2_scale(r, t, offset);
	glm_vec2_add(p1, offset, intersect);
	return true;
}

// handles the collisison (or non-colliison) of two particles
void handle_collision(nbody_context_t *ctx, int particle_id_a, int particle_id_b)
{
	const float size = 0.1f;
	static vec2 hbox_s = { -size, 0.0f };
	static vec2 hbox_f = {  size, 0.0f };

	vec2 p1, p2, p3, p4, intersect;
	vec2 delta;
	vec2 v1, v3;
	float angle;
	
	particle_t *part_a = &ctx->pbuf[particle_id_a];
	particle_t *part_b = &ctx->pbuf[particle_id_b];
	
	disj_cluster_node_t *clust_a = &ctx->disj_clusters[particle_id_a];
	disj_cluster_node_t *clust_b = &ctx->disj_clusters[particle_id_b];

	glm_vec2_rotate(hbox_s, part_a->rot, p1);
	glm_vec2_rotate(hbox_f, part_a->rot, p2);
	glm_vec2_rotate(hbox_s, part_b->rot, p3);
	glm_vec2_rotate(hbox_f, part_b->rot, p4);

	glm_vec2_add(p1, part_a->pos, p1);
	glm_vec2_add(p2, part_a->pos, p2);
	glm_vec2_add(p3, part_b->pos, p3);
	glm_vec2_add(p4, part_b->pos, p4);

	if (!line_intersect(p1, p2, p3, p4, intersect))
		return;

	if (rand_uniform_float(&ctx->rand_state, 0.0f, 1.0f) <= ctx->aggr_prob) {
		disj_cluster_union(clust_a, clust_b);
		return;
	}

	glm_vec2_sub(part_a->pos, intersect, v1);
	glm_vec2_sub(part_b->pos, intersect, v3);
	angle = glm_vec2_dot(v1, v3);
	angle = acos(glm_clamp(angle, -1.0f, 1.0f));

	// TODO: Bounce off with rotation and not just position
	glm_vec2_sub(p1, p3, delta);
	glm_vec2_normalize(delta);
	glm_vec2_scale(delta, BOUNCE_STRENGTH, delta);
	glm_vec2_add(part_a->vel, delta, part_a->vel);
	glm_vec2_sub(part_a->vel, delta, part_b->vel);
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
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		for (j = i + 1; j < ctx->params->particle_cnt; j++) {
			// skip particles that are already in the same cluster
			clust_a = disj_cluster_find(&ctx->disj_clusters[i]);
			clust_b = disj_cluster_find(&ctx->disj_clusters[j]);
			if (clust_a == clust_b) {
				continue;
			}

			// detect collisions and handle them
			handle_collision(ctx, i, j);
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
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		ctx->disj_clusters[i].updated = false;
	}

	// loop over all particles and update their positions and velocities
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		clust = disj_cluster_find(&ctx->disj_clusters[i]);
		part = &ctx->pbuf[i];

		// if the cluster has not been visited before, update velocity
		if (!clust->updated) {
			fpair = rand_norm_pair(&ctx->rand_state, ACCEL_MU, ACCEL_SIGMA);
			clust->vel[0] = (clust->vel[0] * (1.0f - DRAG_COEFF)) + fpair.f1 * dt;
			clust->vel[1] = (clust->vel[1] * (1.0f - DRAG_COEFF)) + fpair.f2 * dt;
			clust->updated = true;
		}

		// propagate the cluster velocity to the individual particle
		part->vel[0] = clust->vel[0];
		part->vel[1] = clust->vel[1];
	}
}

void part_pos_walk(nbody_context_t *ctx, float dt)
{
	disj_cluster_node_t *clust;
	particle_t *part;
	float_pair_t fpair;
	int i;

	// loop over all particles and update positions
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		part = &ctx->pbuf[i];
		part->pos[0] += part->vel[0] * dt;
		part->pos[1] += part->vel[1] * dt;
	}
}

float emergence_idx(nbody_context_t *ctx)
{
	disj_cluster_node_t *node;
	double total_mass = 0.0;
	int cluster_count = 0;
	int i;

	// Option 1: Walk over all of the clusters and return the average mass
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		ctx->disj_clusters[i].updated = false;
	}

	for (i = 0; i < ctx->params->particle_cnt; i++) {
		node = &ctx->disj_clusters[i];
		if (node->updated == false) {
			total_mass += node->mass;
			node->updated = true;
			cluster_count += 1;
		}
	}

	return total_mass / cluster_count;
}

int nbody_init(nbody_context_t *ctx, params_t *params)
{
	float_pair_t fpair;
	float bound;

	// initialize the randomizer
	ctx->rand_state = splitmix64(time(NULL));

	// allocate host particle and cluster buffers
	ctx->params = params;
	if ((ctx->pbuf = malloc(params->particle_cnt * sizeof(particle_t))) == 0) {
		fprintf(stderr, "nbody_init: out of memory\n");
		return -1;
	}

	if ((ctx->disj_clusters = malloc(params->particle_cnt * sizeof(disj_cluster_node_t))) == 0) {
		fprintf(stderr, "nbody_init: out of memory\n");
		free(ctx->pbuf);
		return -1;
	}

	// initialize the data in the particle and cluster buffers
	bound = 1.0 / ctx->params->scale;
	ctx->aggr_prob = params->pre_onset_aggr;
	for (int i = 0; i < params->particle_cnt; i++) {
		// position data belongs to the particle in the cluster
		ctx->pbuf[i].pos[0] = rand_uniform_float(&ctx->rand_state, -bound, bound);
		ctx->pbuf[i].pos[1] = rand_uniform_float(&ctx->rand_state, -bound, bound);
		ctx->pbuf[i].vel[0] = 0.0f;
		ctx->pbuf[i].vel[1] = 0.0f;
		ctx->pbuf[i].rot = rand_uniform_float(&ctx->rand_state, -M_PI, M_PI);
		ctx->pbuf[i].type = rand_u32(&ctx->rand_state) % ctx->params->num_ptypes;

		// velocity data belongs to the cluster itself
		ctx->disj_clusters[i].parent = &ctx->disj_clusters[i]; // each particle is a cluster
		ctx->disj_clusters[i].rank = 0; // clusters rank 0
		ctx->disj_clusters[i].mass = 1.0f; // clusters mass 1
		ctx->disj_clusters[i].com[0] = 0.0f; // clusters com is (0, 0)
		ctx->disj_clusters[i].com[1] = 0.0f; // clusters com is (0, 0)
		ctx->disj_clusters[i].updated = false; // redundant (see part_vel_walk)
		ctx->disj_clusters[i].vel[0] = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
		ctx->disj_clusters[i].vel[1] = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
	}

	return 0;
}

float nbody_update(nbody_context_t *ctx, float dt)
{
	if (rand_uniform_float(&ctx->rand_state, 0.0f, 1.0f) < ctx->params->onset_prob)
		ctx->aggr_prob = ctx->params->post_onset_aggr;
	part_vel_walk(ctx, dt);
	part_pair_walk(ctx);
	part_pos_walk(ctx, dt);
	return emergence_idx(ctx);
}

void nbody_deinit(nbody_context_t *ctx)
{
	free(ctx->pbuf);
	free(ctx->disj_clusters);
}
