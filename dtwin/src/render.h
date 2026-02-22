#pragma once

#include "glad/glad.h"
#include <stdlib.h>

// TODO: Move me to somewhere shared by cuda and OpenGL
typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    int type;
} particle_t;

typedef struct {
    int pid;
    int pipefds[2];
} ffmpeg_handle_t;

typedef struct {
    ffmpeg_handle_t h_ffmpeg;
    GLuint particle_buffer;
    GLuint frame_buf;
    GLuint frame_buf_tex;

    // First pass: Take input particle positions and pass them to the geometry
    // shader to create vertices along the particle boundaries.
    GLuint render_program;
    GLuint render_vert;
    GLuint render_geometry;
    GLuint render_frag;

    // Second pass: Point Spreading Function for microscope optics.
    GLuint psf_program;
    GLuint psf_vert;
    GLuint psf_frag;

    // Third pass: Motion blur? Discoloration?
} render_context_t;

#define RES_X 1280
#define RES_Y 720
#define RES "RES_XxRES_Y"

/**
 * @brief Launches ffmpeg to handle a video stream.
 * @param handle Populated with info to interface with subprocess.
 * @param resolution Resolution string passed to ffmpeg (e.g., 1280x720)
 * @return 0 on success or -1 on error.
 */
int ffmpeg_open(ffmpeg_handle_t *handle, const char *const resolution,
                const char *const fname);

/**
 * @brief Writes a buffer to ffmpeg over a pipe.
 * @param handle Populated with info to interface with subprocess.
 * @param buf The buffer to write.
 * @param sz The size to write.
 * @return 0 on success or -1 on error.
 */
int ffmpeg_write(ffmpeg_handle_t *handle, void *const buf, size_t sz);

/**
 * @breif Closes ffmpeg and waits for it to exit.
 * @param handle The handle to the ffmpeg instance.
 * @return 0 on success or -1 on error.
 */
int ffmpeg_close(ffmpeg_handle_t *handle);

/**
 * @brief Initializes the render pipeline to feed data to out_path.
 * @param context A renderer context that can be used later.
 * @param out_path File path that the simulation should output to.
 * @return 0 on success or -1 on error.
 */
int render_init(render_context_t *context, char *const out_path);

/**
 * @brief Renderes a frame and encodes it using ffmpeg.
 * @param context The renderer context to use.
 * @param part_buf Particle buffer holding the current state or the sim.
 * @return 0 on success or -1 on error.
 */
int render_frame(render_context_t *context, particle_t *part_buf);

/**
 * @brief Frees all resources associated with a render context.
 * @param context The render context to free.
 * @return 0 on success or -1 on error.
 */
int render_deinit(render_context_t *context)
