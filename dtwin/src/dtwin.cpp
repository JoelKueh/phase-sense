
#include "render.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <cstdint>
#include <cstring>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;

const char *OUT_FNAME = "./out/test.mp4";

// TODO: Remove me
const GLfloat vao_centered_point[] = {
    0.5f, 0.5, 0.0f
};

#define RES_X 1280
#define RES_Y 720
#define RES "RES_XxRES_Y"

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(render_context_t *context, const char path[],
             uint32_t res_x, uint32_t res_y)
{
	int result = 0;
	
	if (render_open_output(context, path, res_x, res_y) == -1) {
		result = -1;
		goto out;
	}
	
    glBindBuffer(GL_ARRAY_BUFFER, context->particle_vao);
    glBufferData(context->particle_vao, sizeof(vao_centered_point),
                 vao_centered_point, GL_STATIC_DRAW);
	if (render_frame(context) == -1) {
		result = -1;
		goto out_close_output;
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

	if (render_init(&context) == -1) {
		fprintf(stderr, "opengl context initialization failed\n");
		result = 1;
		goto out;
	}
	
	if (simulate(&context, "./out/test.mp4", 1280, 720)  == -1) {
		fprintf(stderr, "simulation error\n");
		result = 1;
		goto out_render_deinit;
	}

out_render_deinit:
	render_deinit(&context);
out:
	return result;
}

