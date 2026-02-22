
#include "render.h"
#include "glad/glad.h"

#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

const char *FFMPEG_PATH = "ffmpeg";

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
               "-s", resolution, "-i", "-", fname);
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

int render_init(render_context_t *context, char *const resolution,
                char *const out_path) {
    int result = 0;

    // Open a pipe to stream the data to ffmpeg
    ffmpeg_handle_t handle;
    if (ffmpeg_open(&handle, resolution, out_path) == -1) {
        fprintf(stderr, "error in ffmpeg process creation\n");
        goto err;
    }

    // Initialize the particle buffer as a uniform buffer.
    glGenBuffers(1, &context->particle_buffer);
    glBindBuffer(GL_TEXTURE_1D, context->particle_buffer);

    // Initialize the frame buffer.
    glGenFramebuffers(1, &context->frame_buf);
    glBindFramebuffer(GL_FRAMEBUFFER, context->frame_buf);
    glGenTextures(1, &context->frame_buf_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, RES_X, RES_Y, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         context->frame_buf_tex, 0);

    // Check for errors in the framebuffer creation.
    if (glCheckFramebufferStatus(context->frame_buf) !=
        GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "error in framebuffer creation\n");
        return -1;
    }

    // PASS 1: Geometry Shader


err_close_ffmpeg:
    close(context->h_ffmpeg.pipefds[1]);

err:
    result = -1;

out:
    return 0;
}

int render_frame(render_context_t *context, particle_t *part_buf) {}

int render_deinit(render_context_t *context) {}
