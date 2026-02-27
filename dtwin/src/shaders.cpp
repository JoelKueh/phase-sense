
// Do-nothing vertex shader.
extern const char vert_src_nop[] = R"(
#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aVel;
layout (location = 2) in float aRot;
layout (location = 3) in int aPartIdx;

flat out vec2 vVel;
flat out float vRot;
flat out int vPartIdx;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    vVel = aVel;
    vRot = aRot;
    vPartIdx = aPartIdx;
}
)";

// Vertex shader that instantiates a quad that covers the whole screen.
extern const char vert_src_quad[] = R"(
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
extern const char geom_src_particles[] = R"(
#version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

flat in vec2 vVel[];
flat in float vRot[];
flat in int vPartIdx[];
flat out vec2 fVel;

const vec2 OFFSETS[4] = vec2[](
    vec2(-0.05, -0.01),
    vec2( 0.05, -0.01),
    vec2(-0.05,  0.01),
    vec2( 0.05,  0.01)
);

void main()
{
    const mat2 rot = mat2(
        cos(vRot[0]), -sin(vRot[0]),
        sin(vRot[0]),  cos(vRot[0])
    );

    for (int i = 0; i < 4; i++) {
        gl_Position = gl_in[0].gl_Position + vec4(rot * OFFSETS[i], 0.0, 0.0);
        fVel = vVel[0];
        EmitVertex();
    }
    EndPrimitive();
}
)";

// Particle instantiation fragment shader.
extern const char frag_src_particles[] = R"(
#version 430 core
layout (location = 0) out vec3 FragColor;
layout (location = 1) out vec2 FragVelocity;

flat in vec2 fVel;

void main()
{
    FragColor = vec3(0.6, 0.2, 0.2);
    FragVelocity = fVel;
}
)";

// Gaussian blur fragment shader.
extern const char frag_src_blur[] = R"(
#version 430 core

in vec2 uv;
out vec4 gl_FragColor;

void main()
{
    gl_FragColor=vec4(uv, 0.0, 1.0);
} 
)";
