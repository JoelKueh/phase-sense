

typedef struct {
	float px;
	float py;
	float vx;
	float vy;
	float rotation;
	int type;
} particle_t;

//i am assuming this is structurally identical to the following
/*
typedef struct {
    glm::vec2 position;
    glm::vec2 velocity;
    GLfloat rotation;
    GLint type;
} particle_t;
*/

typedef struct
{
	int ppv;
	int n;
	particle_t *d_vbo;
	void *d_accel;
} cu_context_t;

extern "C" cu_context_t register_gl(int vbo, int ppv, int n);
//extern "C" void init_bodies (cu_context_t ctx);
extern "C" void cuda_update(cu_context_t ctx, float dt);
extern "C" void cuda_free(cu_context_t ctx);
