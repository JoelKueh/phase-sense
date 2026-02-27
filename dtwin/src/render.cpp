
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
extern const char frag_src_blur[];

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
	if (geom_src != nullptr) {
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
	if (geom_src != nullptr)
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
    if (geom_src != nullptr)
        glDeleteShader(geom_shader);
out_delete_vert:
    glDeleteShader(vert_shader);

out:
    return result;
}

int render_init(render_context_t *context, uint32_t res_x, uint32_t res_y) {
    const GLenum inst_buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    const GLenum draw_buffers[2] = {GL_COLOR_ATTACHMENT0};
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
    if (!context->window) {
        fprintf(stderr, "error creating glfw window\n");
        goto err_close_glfw;
    }
    glfwMakeContextCurrent(context->window);
    gladLoadGL();

    // Create the particle instantiation output texture
    glGenTextures(1, &context->inst_out_tex);
    glBindTexture(GL_TEXTURE_2D, context->inst_out_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, context->res_x, context->res_y, 0,
                 GL_RGBA8, GL_UNSIGNED_BYTE, 0);

    // Create the particle instantiation velocity texture
    glGenTextures(1, &context->inst_vel_tex);
    glBindTexture(GL_TEXTURE_2D, context->inst_vel_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, context->res_x, context->res_y, 0,
                 GL_RG16F, GL_HALF_FLOAT, 0);

    // Create the particle instantiation framebuffer
    glGenFramebuffers(1, &context->inst_frame_buf);
    glBindFramebuffer(GL_FRAMEBUFFER, context->inst_frame_buf);
    glBindTexture(GL_TEXTURE_2D, context->inst_out_tex);
    glBindTexture(GL_TEXTURE_2D, context->inst_vel_tex);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, context->inst_out_tex, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, context->inst_vel_tex, 0);
    glDrawBuffers(2, inst_buffers);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "error creating particle instantiation framebuffer\n");
        goto err_close_glfw;
    }

    // Create the draw output texture
    glGenTextures(2, context->draw_out_texs);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, context->draw_out_texs[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, context->res_x, context->res_y, 0,
                     GL_RGBA8, GL_UNSIGNED_BYTE, 0);
    }

    // Create the draw framebuffer
    glGenFramebuffers(2, context->draw_frame_bufs);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, context->draw_frame_bufs[i]);
        glBindTexture(GL_TEXTURE_2D, context->draw_out_texs[i]);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, context->draw_out_texs[i], 0);
        glDrawBuffers(1, draw_buffers);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "error creating draw framebuffers\n");
            goto err_close_glfw;
        }
    }

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

    // PASS 2: Blurring Kernel
    glGenVertexArrays(1, &context->empty_vao);
    if (compile_shader_program(&context->psf_program, vert_src_quad, nullptr, frag_src_blur)) {
        goto err_close_glfw;
    }

    // Do not terminate glfw on success
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
    glBindFramebuffer(GL_FRAMEBUFFER, context->inst_frame_buf);
    glViewport(0, 0, context->res_x, context->res_y);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // PASS 1: Particle instantiation
    glUseProgram(context->particle_program);
    glBindVertexArray(context->particle_vao);
    glDrawArrays(GL_POINTS, 0, 50);

    // PASS 2: PSF Blurring
    glBindFramebuffer(GL_FRAMEBUFFER, context->draw_frame_bufs[0]);
    glViewport(0, 0, context->res_x, context->res_y);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(context->psf_program);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      
    // Read the buffer back from the GPU and write it to ffmpeg.
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glFinish();
    memset(context->ffmpeg_buf, 0xFF, context->res_x * context->res_y * 4);
    // glReadPixels(0, 0, context->res_x, context->res_y, GL_RGBA,
    //              GL_UNSIGNED_BYTE, context->ffmpeg_buf);
    glReadPixels(0, 0, context->res_x, context->res_y, GL_RG,
                 GL_HALF_FLOAT, context->ffmpeg_buf);
    if (ffmpeg_write(&context->h_ffmpeg, context->ffmpeg_buf,
                     context->res_x * context->res_y * 4) == -1)
        return -1;

    return 0;
}

void render_deinit(render_context_t *context) {
    glfwTerminate();
}
