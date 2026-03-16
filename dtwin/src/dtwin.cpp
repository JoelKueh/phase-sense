
#include "render.h"
#include "nbody_cu.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <time.h>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;
int n = 2000;

const char *OUT_FNAME = "./out/test.mp4";

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *context, const char path[])
{
	particle_t buf[n];
	int result = 0;
	if (render_open_output(context, path) == -1) {
		result = -1;
		//goto out;
		return result;
	}

	for (int i = 0; i < n; i++) {
		buf[i].px = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].py = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].vx = 0.f;
		buf[i].vy = 0.f;
		buf[i].rp = (float)std::rand() / (float)RAND_MAX * 2 * M_PI;
		buf[i].rv = 0.f;
		buf[i].type = 0;
		// buf[i].position.x = 0.5;
		// buf[i].position.y = 0.5;
		// buf[i].velocity.x = 0.5;
		// buf[i].velocity.y = 0.5;
		// buf[i].rotation = 0.25;
		// buf[i].type = 0;
	}
    glBindBuffer(GL_ARRAY_BUFFER, context->particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(buf), buf, GL_DYNAMIC_DRAW);

    cu_context_t cu_ctx = register_gl(context->particle_vbo, 7, n);
    fprintf(stderr, "registered cuda context\n");

    for (int i = 0; i < 50; i++) {

	    time_t start = clock();
		cuda_update(cu_ctx, 0.2);

		time_t cuda = clock();
		fprintf(stderr, "cuda update time: %fms, ", 1000.0*(cuda - start) / CLOCKS_PER_SEC); 
		if (render_frame(context) == -1) {
			result = -1;
			goto out_close_output;
		}
		time_t render = clock();

		fprintf(stderr, "render time: %fms\n", 1000.0*(render - cuda) / CLOCKS_PER_SEC);
    }

out_close_output:
	if (render_close_output(context) == -1) {
		result = -1;
	}

out:
	return result;
}

int main()
{
	int result = 0;
	render_context_t context;

	if (render_init(&context, screen_width, screen_height) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}
	
	if (simulate(&context, "./out/test.mp4")  == -1) {
		fprintf(stderr, "simulation error\n");
		result = 1;
		goto out_render_deinit;
	}

out_render_deinit:
	render_deinit(&context);

out:
	return result;
}

