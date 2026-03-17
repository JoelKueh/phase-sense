const char inst_vert_glsl[] = {
#embed "shaders/inst_vert.glsl"
};
const unsigned int inst_vert_glsl_len = sizeof(inst_vert_glsl);

const char inst_geom_glsl[] = {
#embed "shaders/inst_geom.glsl"
};
const unsigned int inst_geom_glsl_len = sizeof(inst_geom_glsl);

const char inst_frag_glsl[] = {
#embed "shaders/inst_frag.glsl"
};
const unsigned int inst_frag_glsl_len = sizeof(inst_frag_glsl);

const char quad_vert_glsl[] = {
#embed "shaders/quad_vert.glsl"
};
const unsigned int quad_vert_glsl_len = sizeof(quad_vert_glsl);

const char gaus_frag_glsl[] = {
#embed "shaders/gaus_frag.glsl"
};
const unsigned int gaus_frag_glsl_len = sizeof(gaus_frag_glsl);

const char cl_nbody_cu[] = {
#embed "nbody.cl"
};
const unsigned int cl_nbody_cu_len = sizeof(cl_nbody_cu);
