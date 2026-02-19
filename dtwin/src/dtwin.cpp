
#include "glad/glad.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>

#include <iostream>
#include <cstdint>
#include <cstring>

float timePast = 0;
int screen_width = 1200;
int screen_height = 1200;

const char *OUT_FNAME = "./test.mp4";
const char *FFMPEG_PATH = "ffmpeg";

typedef struct {
	int pid;
	int pipefds[2];
} ffmpeg_handle_t;

/**
 * @brief Launches ffmpeg to handle a video stream.
 * @param handle Populated with info to interface with subprocess.
 * @param resolution Resolution string passed to ffmpeg (e.g., 1280x720)
 * @return 0 on success or -1 on error.
 */
int ffmpeg_open(ffmpeg_handle_t *handle, const char *const resolution)
{
	// Open the pipe for streaming data to ffmepg
	if (pipe(handle->pipefds))
		return -1;
	
	// Spawn the ffmpeg subprocess.
	handle->pid = fork();
	if (handle->pid == -1) {
		close(handle->pipefds[0]);
		close(handle->pipefds[1]);
		return -1;
	} else if (handle->pid == 0) {
		close(handle->pipefds[1]);
		execl(FFMPEG_PATH, FFMPEG_PATH, "-r", "0", "-f", "awvideo", "-pix_fmt",
		      "rgba", "-s", resolution, "-i", "-", "-threads", "-preset",
		      "ast", "-y", "-pix_fmt", "uv420p", "-crf", "1", "-vf", "flip",
		      OUT_FNAME);
		exit(-1);
	} else {
		close(handle->pipefds[0]);
		return 0;
	}
}

/**
 * @brief Writes a buffer to ffmpeg over a pipe.
 * @param handle Populated with info to interface with subprocess.
 * @param buf The buffer to write.
 * @param sz The size to write.
 * @return 0 on success or -1 on error.
 */
int ffmpeg_write(ffmpeg_handle_t *handle, void *const buf, size_t sz)
{
	if (sz != write(handle->pipefds[1], buf, sz))
		return -1;
	return 0;
}

/**
 * @breif Closes ffmpeg and waits for it to exit.
 * @param handle The handle to the ffmpeg instance.
 * @return 0 on success or -1 on error.
 */
int ffmpeg_close(ffmpeg_handle_t *handle)
{
	close(handle->pipefds[1]);
	if (waitpid(handle->pid, nullptr, 0) == -1)
		return -1;
	return 0;
}

/**
 * @brief Runs a single simulation, rendering the output to an mp4 file.
 * @param path The path to render to.
 * @param params Simulator parameters struct.
 * @return 0 on success or -1 on error.
 */
int simulate(const char path[])
{
	uint32_t frame_buf[1280][720];

	// Open a pipe to stream the data to ffmpeg
	ffmpeg_handle_t handle;
	if (ffmpeg_open(&handle, "1280x720") == -1)
		return -1;
	
	// Write a couple of frames to ffmpeg
	for (int i = 0; i < 120; i++) {
		memset(frame_buf, 0xFF00FF88, sizeof(frame_buf));
		if (ffmpeg_write(&handle, frame_buf, sizeof(frame_buf)) == -1)
			return -1;
		memset(frame_buf, 0x00FF0088, sizeof(frame_buf));
		if (ffmpeg_write(&handle, frame_buf, sizeof(frame_buf)) == -1)
			return -1;
	}

	// Close ffmpeg
	if (ffmpeg_close(&handle) == -1)
		return -1;
}

int main()
{
	// Check if we can access ffmpeg.
	if (access(FFMPEG_PATH, X_OK) != 0) {
		std::cerr << "ERROR: ffmpeg was not found in the path" << std::endl;
	}
	
	simulate(OUT_FNAME);
}

