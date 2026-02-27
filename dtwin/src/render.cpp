
#include "render.h"
#include "glad/glad.h"

#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <GLFW/glfw3.h>

const char *FFMPEG_PATH = "ffmpeg";

// Shader definitions in src/shaders.cpp
extern const char vert_src_nop[];
extern const char vert_src_quad[];
extern const char geom_src_particles[];
extern const char frag_src_particles[];

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
}

int ffmpeg_open(ffmpeg_handle_t *handle, const char *const resolution,
                const char *const fname) {
    // Open the pipe for streaming data to ffmepg
    if (pipe(handle->pipefds)) {
        perror("pipe");
        return -1;
    }

    // Spawn the ffmpeg subprocess.
    handle->pid = fork();
    if (handle->pid == -1) {
        close(handle->pipefds[0]);
        close(handle->pipefds[1]);
        perror("fork");
        return -1;
    } else if (handle->pid == 0) {
        close(handle->pipefds[1]);
        dup2(handle->pipefds[0], STDIN_FILENO);
        execlp(FFMPEG_PATH, FFMPEG_PATH, "-f", "rawvideo", "-pix_fmt", "rgba",
               "-framerate", "4", "-s", resolution, "-i", "-", fname);
        perror("execlp");
        fprintf(stderr, "Is ffmpeg in the path?\n");
        exit(-1);
    } else {
        close(handle->pipefds[0]);
        return 0;
    }
}

int ffmpeg_write(ffmpeg_handle_t *handle, void *const buf, size_t sz) {
    if (sz != write(handle->pipefds[1], buf, sz)) {
        perror("write");
        return -1;
    }
    return 0;
}

int ffmpeg_close(ffmpeg_handle_t *handle) {
    int status;

    close(handle->pipefds[1]);
    if (waitpid(handle->pid, &status, 0) == -1) {
        perror("waitpid");
        return -1;
    }

    if (status != 0) {
        fprintf(stderr, "ffmpeg had non-zero exit code\n");
        return -1;
    }

    return 0;
}

int compile_shader_program(GLuint *program, const char vert_src[],
                           const char geom_src[], const char frag_src[]) {
    int result = 0;
    int success;
    char info_log[512];

    GLuint vert_shader;
    GLuint geom_shader;
    GLuint frag_shader;

    // Create the vertex shader
    vert_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert_shader, 1, &vert_src, NULL);
	glCompileShader(vert_shader);
	glGetShaderiv(vert_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vert_shader, 512, NULL, info_log);
		fprintf(stderr, "vert_shader compile failed\n%s", info_log);
		result = -1;
		goto out;
	}

	// Create the geometry shader
    geom_shader = glCreateShader(GL_GEOMETRY_SHADER);
	glShaderSource(geom_shader, 1, &geom_src, NULL);
	glCompileShader(geom_shader);
	glGetShaderiv(geom_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(geom_shader, 512, NULL, info_log);
		fprintf(stderr, "geom_shader compile failed\n%s", info_log);
		result = -1;
		goto out_delete_vert;
	}

	// Create the fragment shader
	frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag_shader, 1, &frag_src, NULL);
	glCompileShader(frag_shader);
	glGetShaderiv(frag_shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(frag_shader, 512, NULL, info_log);
		fprintf(stderr, "frag_shader compile failed\n%s", info_log);
		result = -1;
		goto out_delete_geom;
	}

	// Link the shaders into the shader program
	if ((*program = glCreateProgram()) == 0) {
	    fprintf(stderr, "glCreateProgram: failed");
	    result = -1;
	    goto out_delete_frag;
	}
	glAttachShader(*program, vert_shader);
	glAttachShader(*program, geom_shader);
	glAttachShader(*program, frag_shader);
	glLinkProgram(*program);

	// Check for errors in the link stage
	glGetProgramiv(*program, GL_LINK_STATUS, &success);
	if (!success) {
	    glGetProgramInfoLog(*program, 512, NULL, info_log);
	    fprintf(stderr, "program link failed\n%s", info_log);
	    result = -1;
	    goto out_delete_frag;
	}

out_delete_frag:
    glDeleteShader(frag_shader);
out_delete_geom:
    glDeleteShader(geom_shader);
out_delete_vert:
    glDeleteShader(vert_shader);

out:
    return result;
}

int render_init(render_context_t *context, uint32_t res_x, uint32_t res_y) {
    int result = 0;

    context->res_x = res_x;
    context->res_y = res_y;

    // Create the GLFW context with no no visible window.
    glfwInit();
    glfwSetErrorCallback(glfw_error_callback);
    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
    if (!glfwInit()) {
        fprintf(stderr, "error creating glfw context\n");
        goto err;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    context->window = glfwCreateWindow(res_x, res_y, "phase-sense", NULL, NULL);
    if (!context->window)
        goto err_close_glfw;
    glfwMakeContextCurrent(context->window);
    gladLoadGL();

    // PASS 1: Geometry Shader
    glGenVertexArrays(1, &context->particle_vao);
    glGenBuffers(1, &context->particle_vbo);
    glBindVertexArray(context->particle_vao);
    glBindBuffer(GL_ARRAY_BUFFER, context->particle_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(particle_t),
                          (GLvoid*)offsetof(particle_t, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(particle_t),
                          (GLvoid*)offsetof(particle_t, velocity));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(particle_t),
                          (GLvoid*)offsetof(particle_t, rotation));
    glEnableVertexAttribArray(3);
    glVertexAttribIPointer(3, 1, GL_INT, sizeof(particle_t),
                           (GLvoid*)offsetof(particle_t, type));
    if (compile_shader_program(&context->particle_program, vert_src_nop,
                               geom_src_particles, frag_src_particles)) {
        goto err_close_glfw;
    }

    // Skip the termination on error.
    goto out;

err_close_glfw:
    glfwTerminate();
err:
    result = -1;

out:
    return 0;
}

int render_open_output(render_context_t *context, const char fname[]) {
    char res_buf[16] = "";

    if ((context->ffmpeg_buf =
            (uint8_t*)malloc(context->res_x * context->res_y * 4)) == 0)
        return -1;

    snprintf(res_buf, sizeof(res_buf), "%dx%d",
             context->res_x, context->res_y);
    if (ffmpeg_open(&context->h_ffmpeg, res_buf, fname) == -1)
        return -1;

    return 0;
}

int render_close_output(render_context_t *context) {
    free(context->ffmpeg_buf);
    return ffmpeg_close(&context->h_ffmpeg);
}

int render_frame(render_context_t *context) {
    // Prepare the rendering buffer.
    glViewport(0, 0, context->res_x, context->res_y);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // PASS 1: Particle instantiation
    glUseProgram(context->particle_program);
    glBindVertexArray(context->particle_vao);
    glDrawArrays(GL_POINTS, 0, 50);

    // PASS 2: PSF Blurring
    // TODO

    // Read the buffer back from the GPU and write it to ffmpeg.
    glReadBuffer(GL_BACK);
    glFinish();
    memset(context->ffmpeg_buf, 0xFF, context->res_x * context->res_y * 4);
    glReadPixels(0, 0, context->res_x, context->res_y, GL_RGBA,
                 GL_UNSIGNED_BYTE, context->ffmpeg_buf);
    if (ffmpeg_write(&context->h_ffmpeg, context->ffmpeg_buf,
                     context->res_x * context->res_y * 4) == -1)
        return -1;

    return 0;
}

void render_deinit(render_context_t *context) {
    glfwTerminate();
}
