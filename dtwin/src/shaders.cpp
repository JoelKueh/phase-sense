
// Do-nothing vertex shader.
const char *vtx_src_nop = R"(
#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aRot;
layout (location = 2) in int aPartIdx;

flat out vRot;
flat out vPartIdx;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    vRot = aRot;
    vPartIdx = aPartIdx;
}
)";

// Vertex shader that instantiates a quad that covers the whole screen.
const char *vtx_src_quad = R"(
#version 430 core
out vec2 uv;

void main()
{
	vec2 vertices[3] = vec2[3] (vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
	gl_Position = vec4(vertices[gl_VertexID], 0, 1);
	uv = 0.5 * gl_Position.xy + vec2(0.5);
}
)";

// Particle instantiation geometry shader.
// TODO: Have this read from a UBO for particle shapes.
// - Currently it just renders squares.
const char *geom_src_particles = R"(
#version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

const vec2 OFFSETS[4] = vec2[]{
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5),
    vee2(-0.5,  0.5),
    vec2( 0.5,  0.5)
};

void main()
{
    for (int i = 0; i < 4; i++) {
        gl_Position = gl_in[0].gl_Position + vec4(OFFSETS[i], 0.0, 0.0);
        EmitVeretx();
    }
    EndPrimitive();
}
)";

// Particle instantiation fragment shader.
const char *frag_src_particles = R"(
#version 430 core
out vec3 FragColor;

void main()
{
    FragColor = vec3(0.6, 0.2, 0.2);
}
)";
