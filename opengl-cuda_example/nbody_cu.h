extern "C" void init_bodies (float4 *d_pos, float2 **d_accel, float2 **d_vel, int n);
extern "C" void process_bodies(float4 *d_pos, float2 *d_accel, float2 *d_vel, int n, float dt, float E2);
extern "C" void test_vbo_share(float4 *d_pos);
extern "C" void free_bodies(float2 *d_accel, float2 *d_vel);
