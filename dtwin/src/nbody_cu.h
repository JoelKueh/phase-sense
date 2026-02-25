typedef struct
{
	int ppv;
	int n;
	void *d_vbo;
} cu_context_t;
extern "C" cu_context_t register_gl(int vbo, int ppv, int n);
extern "C" void init_bodies (cu_context_t ctx);
extern "C" void process_bodies(cu_context_t ctx, float dt);
extern "C" void free_bodies(cu_context_t ctx);
