#pragma once

#include "param.h"
#include "glad/glad.h"
#include <stdlib.h>
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <GL/glx.h>
#include <EGL/egl.h>

#define MAX_RENDER_HBOX_LEN 8

typedef struct {
    int pid;
    int pipefds[2];
} ffmpeg_handle_t;

// holds the data for a particle type, list of vertices
typedef struct {
	float px[MAX_RENDER_HBOX_LEN];
	float py[MAX_RENDER_HBOX_LEN];
} render_spine_t;

typedef struct {
    ffmpeg_handle_t h_ffmpeg;
    uint8_t *ffmpeg_buf;
    params_t *params;
    render_spine_t *particle_spines;

    // GLFW context and buffer outputs
    GLFWwindow *window;
    GLuint inst_frame_buf;
    GLuint inst_out_tex;
    GLuint inst_vel_tex;
    GLuint draw_frame_bufs[2];
    GLuint draw_out_texs[2];

    // Particle attriube buffers
    GLuint particle_vao;     // Particle vertex array object.
    GLuint particle_vbo;     // Particle vertex buffer object.
    GLuint particle_ubo;     // Particle uniform buffer object.
    GLuint particle_tree;    // Particle quad tree?
    GLuint empty_vao;        // Empty vao for a fullscreen quad.

    // First pass: Take input particle positions and pass them to the geometry
    // shader to create vertices along the particle boundaries.
    GLuint particle_program;

    // Second pass: Point Spreading Function for microscope optics.
    GLuint psf_program;

    // Third pass: Motion blur? Discoloration?
    // TODO: This might be done at least partly in the particle_program.
    // GPU Gems 3 Chapter 27 talks about storing motion blur data in the depth
    // buffer (see https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect).
} render_context_t;

/**
 * @brief Initializes the render pipeline.
 * @param context A renderer context that can be used later.
 * @return 0 on success or -1 on error.
 */
int render_init(render_context_t *context, params_t *params);

/**
 * @brief Initializes the render pipeline to feed data to out_path.
 * @param context The render context to use.
 * @param out_path File path that the simulation should output to.
 * @return 0 on success or -1 on error.
 */
int render_open_output(render_context_t *context, const char fname[]);

/**
 * @brief Closes ffmpeg and frees up resources so that the pipeline can
 * be reconfigured to output to a different file.
 * @param context The render context.
 */
int render_close_output(render_context_t *context);

/**
 * @brief Renderes a frame and encodes it using ffmpeg.
 * @param context The renderer context to use.
 * @param part_buf Particle buffer holding the current state or the sim.
 * @return 0 on success or -1 on error.
 */
int render_frame(render_context_t *context);

/**
 * @brief Frees all resources associated with a render context.
 * @param context The render context to free.
 */
void render_deinit(render_context_t *context);
