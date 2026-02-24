
# Digital-Twin Simulator

This is a digital-twin simulator for the phase-sense ML pipeline.

## Dependencies

This project depends on the following compile-time libraries:
1. OpenGL 4.3 - Graphics Rendering API
2. GLFW3 - OpenGL Context Creation API
3. glm - OpenGL Math API
4. CUDA - NVIDIA GPGPU Compute API

This project makes use of an ffmpeg executable (not libffmpeg) to encode rendered simulation frames. To run this program, you MUST have ffmpeg added to your PATH.

## Design Philosophy

This repository implements a Brownian motion particle simulator with tunable sticking parameters to allow for the creation of a near-unlimited amount of data for our colloidal aggregation detection machine learning model.

### Phyisics Simulation Backend

The simulator backend is written in CUDA. The backend handles the physical motion and aggregation of particles. Particles randomly move around in 2D space according to a Brownian motion simulation. The simulator tests for collisions between these particles and randomly decides whether the particles will clump together.

For each frame in the simulation, the simulator backend outputs three things.
1. Positions, velocities, and rotations of particles packed as in a GPU buffer.
2. An emergence-index that tracks the progress of the aggregation.
3. The current simulation time.

The positions, velocities, and rotations of the particles are passed to the rendering frontend as a single vertex buffer object. This vertex buffer object is packed as an array of structs in the following format.

```cpp
typedef struct {
    glm::vec2 position;
    glm::vec2 velocity;
    GLfloat rotation;
    GLint type;
} particle_t;
```

The simulator backend outputs a .meta file that is associated with the video. The .meta file uses the CSV format. The first line of the .meta file is a header with information about the name, real-time framerate, particle count, and resolution of the simulated video. This header is followed by a series of rows that hold the simulation time and emergence-index for each frame.

For example, the following .meta file describes a simulation video named simulation_1.mp4. This simulation contains 1280x720 timelapse footage with a framerate of 1 frame every four seconds.

```
simulation_1.mp4,0.25,1280x720
0,0.001
4,0.004
8,0.020
12,0.100
...
```
  
### Rendering Front End

The rendering frontend renders frames based on the particle data supplied by the CUDA backend. The rendering front end consists of the following phases:

1. Particle Intantiation
  - Uses a geometry shader to instantiate particles provided their positions.
  - Uses a fragment shader to draw the interior of these particles.
  - Velocity data is encoded in a velocity buffer to be used later.
2. Point Spreading Function (PSF) Blurring
  - A simple convolution-based blur is applied to the whole image.
3. Motion Blur
  - Motion blur is applied to the whole image using the velocity buffer.

The rendering front end copies the rendered frames back to the CPU and writes them over a pipe to ffmpeg.
