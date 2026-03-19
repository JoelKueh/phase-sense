
# Digital-Twin Simulator

This is a digital-twin simulator for the phase-sense ML pipeline.

## Dependencies

This project has 4 main dependencies:
1. OpenGL 4.3 - Graphics Rendering API/Runtime
  - See your distributions package manager for development headers.
2. GLFW3 - OpenGL Context Creation API
  - See your distributions package manager for development headers.
3. Compiler that meets the C23 standard with the #embed preprocessor directive
  - If your C compiler does not support C23, try [zig-cc](https://andrewkelley.me/post/zig-cc-powerful-drop-in-replacement-gcc-clang.html)
4. Runtime executalbe of ffmpeg added to the PATH
  - Packeged (and likely already installed) on most systems
  - Binaries are avaliable on the [FFMPEG website](https://www.ffmpeg.org/download.html)

## Design Philosophy

This repository implements a Brownian motion particle simulator with tunable sticking parameters to allow for the creation of a near-unlimited amount of data for our colloidal aggregation detection machine learning model.

### Phyisics Simulation Backend

The simulation backend runs entirely on the CPU. The backend handles the physical motion and aggregation of particles. Particles randomly move around in 2D space according to a Brownian motion simulation. The simulator tests for collisions between these particles and randomly decides whether the particles will clump together.

For each frame in the simulation, the simulator backend outputs three things.
1. Positions, velocities, and rotations of particles packed as in a GPU buffer.
2. An emergence-index that tracks the progress of the aggregation.
3. The current simulation time.

The positions, velocities, and rotations of the particles are passed to the rendering frontend as a single vertex buffer object. This vertex buffer object is packed as an array of structs in the following format.

```c
typedef struct {
	float px;
	float py;
	float vx;
	float vy;
	float rotation;
	int type;
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

The rendering frontend renders frames based on the particle data supplied by the backend. The rendering front end consists of the following phases:

1. Particle Intantiation
  - Uses a geometry shader to instantiate particles provided their positions.
  - Particles are defined by their "spine" (strip of line segments) and a radius.
  - The way the particles manipulate light is modeled by a function on the distance from the spine.
  - Velocity data is encoded in a velocity buffer to be used later.
2. Motion Blur
  - Motion blur is applied to the whole image using the velocity buffer.

The rendering front end copies the rendered frames back to the CPU and writes them over a pipe to ffmpeg.
