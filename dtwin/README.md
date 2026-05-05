
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
  - Packaged (and likely already installed) on most systems
  - Binaries are available on the [FFMPEG website](https://www.ffmpeg.org/download.html)

## Design Philosophy

This repository implements a Brownian motion particle simulator with tunable sticking parameters to allow for the creation of a near-unlimited amount of data for our colloidal aggregation detection machine learning model.

The simulation region ranges from -1.0f to 1.0f in both x and y. This is true for both the physics simulation backend and the rendering frontend.

### Phyisics Simulation Backend

The simulation backend runs entirely on the CPU. The backend handles the physical motion and aggregation of particles. Particles randomly move around in 2D space according to a Brownian motion simulation. The simulator tests for collisions between these particles and randomly decides whether the particles will clump together.

For each frame in the simulation, the simulator backend outputs three things.
1. Positions, velocities, and rotations of particles packed as in a GPU buffer.
2. An emergence-index that tracks the progress of the aggregation.
3. The current simulation time.

The positions, velocities, and rotations of the particles are passed to the rendering frontend as a single vertex buffer object. This vertex buffer object is packed as an array of structs in the following format.

```c
typedef struct {
	vec2 pos;
	vec2 vel;
	float rot;
	float rvel;
	int type;
	float intensity;
} particle_t;
```

The simulator backend outputs a .meta file that is associated with the video. The .meta file uses the CSV format. The first line of the .meta file is a header with information about the parameters of the simulation. This header is followed by a series of rows that hold the simulation frame and emergence-index for each frame.

For example, the following .meta file describes a simulation video named simulation_1.mp4. This simulation contains 1280x720 timelapse footage with a framerate of 1 frame every four seconds.

```
aggr_prob,0.732320,avg_len,0.421229,accel,0.125950,drag,0.168514,scale,0.200000,count,122
0,1.000000
1,1.000000
2,1.000000
...
```
  
### Rendering Front End

The rendering frontend renders frames based on the particle data supplied by the backend. The rendering uses the following rendering techniques:

- Uses a geometry shader to instantiate particles provided their positions.
- Particles are defined by their "spine" (strip of line segments) and a radius.
- The way the particles manipulate light is modeled by a function on the distance from the spine.
- Velocity data is encoded in a velocity buffer to be used later.

The rendering front end copies the rendered frames back to the CPU and writes them over a pipe to ffmpeg.

## Running

### Output

The pipeline will output simulation data to `./out/ds` in a list of `.meta` and `.avi` files.

### Parameter Tuning

All parameters in the simulator are tunable via a `.toml` file.

Command
```
./bin/dtwin -p ./params.toml
```

params.toml
```
scale = 0.15
res = 380
pre_onset_aggr = 0.0
post_onset_aggr = 0.3
onset_prob = 0.0025
particle_roughness = 0.05
particle_min_len = 0.01
particle_max_len = 0.8
particle_min_intensity = 0.5
particle_max_intensity = 1.5
accel_distr_mu = 0.0
accel_distr_sig = 0.1
raccel_distr_mu = 0.0
raccel_distr_sig = 0.3
drag_coeff = 0.1
bounce_strength = 0.0005
particle_cnt = 200
num_ptypes = 100
frame_cnt = 3000
fps = 6
video_cnt = 1
```
