#pragma once

#include "glad/glad.h"
#include <stdlib.h>
#include <GLFW/glfw3.h>

// TODO: Should not be an array of structs. Want struct of attribute buffers.
// TODO: Move me to somewhere shared by cuda and OpenGL
typedef struct {
    GLfloat x;
    GLfloat y;
    GLfloat vx;
    GLfloat vy;
    GLfloat rot;
    GLint type;
} particle_t;

typedef struct {
    int pid;
    int pipefds[2];
} ffmpeg_handle_t;

typedef struct {
    ffmpeg_handle_t h_ffmpeg;
    uint8_t *ffmpeg_buf;
    uint32_t res_x;
    uint32_t res_y;

    // GLFW context
    GLFWwindow *window;

    // Particle attriube buffers
    GLuint particle_vao;     // Particle vertex array object.
    GLuint particle_vbo;     // Particle vertex buffer object.
    GLuint particle_pos;     // Probably a ubo window to the vao.
    GLuint particle_vel;     // Particle velocities.
    GLuint particle_types;   // Particle types.
    GLuint particle_tree;    // Particle quad tree?

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
int render_init(render_context_t *context);

/**
 * @brief Initializes the render pipeline to feed data to out_path.
 * @param context The render context to use.
 * @param out_path File path that the simulation should output to.
 * @return 0 on success or -1 on error.
 */
int render_open_output(render_context_t *context, const char fname[],
                       uint32_t res_x, uint32_t res_y);

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
