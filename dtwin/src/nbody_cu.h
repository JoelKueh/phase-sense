extern "C" void register_gl(void **d_pos_v, int vbo);
extern "C" void init_bodies (void *d_pos, void **d_accel, void **d_vel, int n);
extern "C" void process_bodies(void *d_pos, void *d_accel, void *d_vel, int n, float dt, float E2);
extern "C" void free_bodies(void *d_accel, void *d_vel);
