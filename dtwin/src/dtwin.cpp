
#include "render.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;

const char *OUT_FNAME = "./out/test.mp4";

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *context, const char path[])
{
	particle_t buf[50];
	int result = 0;
	
	if (render_open_output(context, path) == -1) {
		result = -1;
		goto out;
	}

	for (int i = 0; i < 50; i++) {
		buf[i].position.x = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].position.y = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].velocity.x = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].velocity.y = -1.0 + 2.0 * (float)std::rand() / (float)RAND_MAX;
		buf[i].rotation = (float)std::rand() / (float)RAND_MAX * 2 * M_PI;
		buf[i].type = 0;
		// buf[i].position.x = 0.5;
		// buf[i].position.y = 0.5;
		// buf[i].velocity.x = 0.5;
		// buf[i].velocity.y = 0.5;
		// buf[i].rotation = 0.25;
		// buf[i].type = 0;
	}
    glBindBuffer(GL_ARRAY_BUFFER, context->particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(buf),
                 buf, GL_STATIC_DRAW);
    for (int i = 0; i < 50; i++) {
		if (render_frame(context) == -1) {
			result = -1;
			goto out_close_output;
		}
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

	if (render_init(&context, 1280, 760) == -1) {
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

