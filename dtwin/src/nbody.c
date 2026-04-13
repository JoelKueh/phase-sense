
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "cglm/vec2.h"
#include "nbody.h"

#define M_PI 3.14159265358979323846

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

// Distance to a line segment through points s1 s2.
float seg_dist(vec2 s1, vec2 s2, vec2 p, vec2 intersect)
{
	vec2 pms1, s2ms1, delta;
	float t;
	
	// compute the parameterized minimizatino point
	glm_vec2_sub(p, s1, pms1);
	glm_vec2_sub(s2, s1, s2ms1);
	t = glm_vec2_dot(pms1, s2ms1) / glm_vec2_dot(s2ms1, s2ms1);
	t = t < 0 ? 0.0f : t > 1 ? 1.0f : t;

	// compute the delta vector
	glm_vec2_scale(s2ms1, t, delta);
	glm_vec2_add(s1, delta, delta);
	glm_vec2_sub(delta, p, delta);

	// compute the "intersection" ponit, halfway between the point and the line
	glm_vec2_scale(delta, 0.5, intersect);
	glm_vec2_add(p, intersect, intersect);

	return glm_vec2_norm(delta);
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
	if (fabsf(rxs) < 1e-12f) {
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

	vec2 p1, p2, p3, p4, intersect;
	vec2 delta, delta_a, delta_b;
	vec2 v1, v3;
	vec2 hbox_s;
	vec2 hbox_f;
	float angle;
	
	particle_t *part_a = &ctx->pbuf[particle_id_a];
	particle_t *part_b = &ctx->pbuf[particle_id_b];
	
	disj_cluster_node_t *clust_a = &ctx->disj_clusters[particle_id_a];
	disj_cluster_node_t *clust_b = &ctx->disj_clusters[particle_id_b];

	// hitbox is a stright line from the start point to the end piont
	hbox_s[0] = ctx->spines[part_a->type].px[0];
	hbox_s[1] = ctx->spines[part_a->type].py[0];
	hbox_f[0] = ctx->spines[part_a->type].px[SPINE_LEN-1];
	hbox_f[1] = ctx->spines[part_a->type].py[SPINE_LEN-1];
	glm_vec2_rotate(hbox_s, part_a->rot, p1);
	glm_vec2_rotate(hbox_f, part_a->rot, p2);
	glm_vec2_add(p1, part_a->pos, p1);
	glm_vec2_add(p2, part_a->pos, p2);

	hbox_s[0] = ctx->spines[part_b->type].px[0];
	hbox_s[1] = ctx->spines[part_b->type].py[0];
	hbox_f[0] = ctx->spines[part_b->type].px[SPINE_LEN-1];
	hbox_f[1] = ctx->spines[part_b->type].py[SPINE_LEN-1];
	glm_vec2_rotate(hbox_s, part_b->rot, p3);
	glm_vec2_rotate(hbox_f, part_b->rot, p4);
	glm_vec2_add(p3, part_b->pos, p3);
	glm_vec2_add(p4, part_b->pos, p4);

	if (!line_intersect(p1, p2, p3, p4, intersect))
		return;

	if (rand_uniform_float(&ctx->rand_state, 0.0f, 1.0f) <= ctx->aggr_prob) {
		disj_cluster_union(clust_a, clust_b);
		clust_a = disj_cluster_find(clust_a);
		clust_b = disj_cluster_find(clust_b);
		glm_vec2_copy(clust_a->vel, part_a->vel);
		glm_vec2_copy(clust_b->vel, part_b->vel);
		return;
	}

	glm_vec2_sub(part_a->pos, intersect, v1);
	glm_vec2_sub(part_b->pos, intersect, v3);
	angle = glm_vec2_dot(v1, v3);
	angle = acos(glm_clamp(angle, -1.0f, 1.0f));

	// TODO: Bounce off with rotation and not just position
	glm_vec2_sub(p1, p3, delta);
	glm_vec2_scale(delta, ctx->params->bounce_strength, delta);
	glm_vec2_scale(delta, 1.0f / clust_a->mass, delta_a);
	glm_vec2_scale(delta, 1.0f / clust_b->mass, delta_b);
	glm_vec2_normalize(delta);
	glm_vec2_add(clust_a->vel, delta_a, clust_a->vel);
	glm_vec2_sub(clust_b->vel, delta_b, clust_b->vel);
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
	vec2 delta;
	int i;

	// loop over all clusters and set updated flag to 0
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		ctx->disj_clusters[i].updated = false;
	}

	// loop over all particles and update their positions and velocities
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		clust = disj_cluster_find(&ctx->disj_clusters[i]);
		part = &ctx->pbuf[i];

		// if the cluster has not been visited before, update velocity and com
		if (!clust->updated) {
			fpair = rand_norm_pair(&ctx->rand_state, ctx->params->accel_distr_mu,
			                       ctx->params->accel_distr_sig);
			clust->vel[0] = (clust->vel[0] * (1.0f - ctx->params->drag_coeff))
			                + fpair.f1 * dt / clust->mass;
			clust->vel[1] = (clust->vel[1] * (1.0f - ctx->params->drag_coeff))
			                + fpair.f2 * dt / clust->mass;
			glm_vec2_scale(clust->vel, dt, delta);
			glm_vec2_add(clust->com, delta, clust->com);

			fpair = rand_norm_pair(&ctx->rand_state, ctx->params->raccel_distr_mu,
			                       ctx->params->raccel_distr_sig);
			clust->rvel = (clust->rvel * (1.0f - ctx->params->drag_coeff))
							+ fpair.f1 * dt / clust->mass;
			clust->updated = true;
		}
	}
}

void part_pos_walk(nbody_context_t *ctx, float dt)
{
	disj_cluster_node_t *clust;
	particle_t *part;
	float_pair_t fpair;
	vec2 delta;
	int i;

	// loop over all particles and update positions
	for (i = 0; i < ctx->params->particle_cnt; i++) {
		clust = disj_cluster_find(&ctx->disj_clusters[i]);
		part = &ctx->pbuf[i];

		// handle cluster rotation
		glm_vec2_sub(part->pos, clust->com, delta);
		glm_vec2_rotate(delta, clust->rvel * dt, delta);
		glm_vec2_add(clust->com, delta, part->pos);
		part->rot += clust->rvel * dt;

		// handle cluster velocity
		glm_vec2_scale(clust->vel, dt, delta);
		glm_vec2_add(part->pos, delta, part->pos);
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
		node = disj_cluster_find(&ctx->disj_clusters[i]);
		if (node->updated == false) {
			total_mass += node->mass;
			node->updated = true;
			cluster_count += 1;
		}
	}

	return total_mass / cluster_count;
}

int nbody_init(nbody_context_t *ctx, params_t *params, spine_t *spines)
{
	float_pair_t fpair;
	float bound;

	ctx->rand_state = splitmix64(time(NULL));
	ctx->spines = spines;

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
		ctx->pbuf[i].rvel = 0.0f;
		ctx->pbuf[i].type = rand_u32(&ctx->rand_state) % ctx->params->num_ptypes;

		// velocity data belongs to the cluster itself
		ctx->disj_clusters[i].parent = &ctx->disj_clusters[i]; // each particle is a cluster
		ctx->disj_clusters[i].rank = 0; // clusters rank 0
		ctx->disj_clusters[i].mass = 1.0f; // clusters mass 1
		ctx->disj_clusters[i].com[0] = ctx->pbuf[i].pos[0];
		ctx->disj_clusters[i].com[1] = ctx->pbuf[i].pos[1];
		ctx->disj_clusters[i].updated = false; // redundant (see part_vel_walk)
		ctx->disj_clusters[i].vel[0] = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
		ctx->disj_clusters[i].vel[1] = rand_uniform_float(&ctx->rand_state, -0.1, 0.1);
		ctx->disj_clusters[i].rvel = 0.0f;
	}

	return 0;
}

float nbody_update(nbody_context_t *ctx, float dt)
{
	if (rand_uniform_float(&ctx->rand_state, 0.0f, 1.0f) < ctx->params->onset_prob)
		ctx->aggr_prob = ctx->params->post_onset_aggr;
	part_pair_walk(ctx);
	part_vel_walk(ctx, dt);
	part_pos_walk(ctx, dt);
	return emergence_idx(ctx);
}

void nbody_deinit(nbody_context_t *ctx)
{
	free(ctx->pbuf);
	free(ctx->disj_clusters);
}
