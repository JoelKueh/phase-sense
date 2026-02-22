
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


/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(const char path[])
{
	uint8_t frame_buf[1280][720][3];

	// Open a pipe to stream the data to ffmpeg
	ffmpeg_handle_t handle;
	if (ffmpeg_open(&handle, "1280x720", path) == -1)
		return -1;
	
	// Write a couple of frames to ffmpeg
	for (int i = 0; i < 120; i++) {
		for (int j = 0; j < 1280; j++) {
			for (int k = 0; k < 720; k++) {
				frame_buf[j][k][0] = 0xFF;
				frame_buf[j][k][1] = 0x00;
				frame_buf[j][k][2] = 0x00;
			}
		}
		if (ffmpeg_write(&handle, frame_buf, sizeof(frame_buf)) == -1)
			return -1;

		for (int j = 0; j < 1280; j++) {
			for (int k = 0; k < 720; k++) {
				frame_buf[j][k][0] = 0x00;
				frame_buf[j][k][1] = 0xFF;
				frame_buf[j][k][2] = 0x00;
			}
		}
		if (ffmpeg_write(&handle, frame_buf, sizeof(frame_buf)) == -1)
			return -1;
	}

	// Close ffmpeg
	if (ffmpeg_close(&handle) == -1)
		return -1;

	return 0;
}

int main()
{
	// Check if we can access ffmpeg.
	// if (access(FFMPEG_PATH, X_OK) != 0) {
	// 	std::cerr << "ERROR: ffmpeg was not found in the path" << std::endl;
	// }
	
	if (simulate(OUT_FNAME) == -1)
		perror("simulate");
}

