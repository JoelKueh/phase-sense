#version 430 core

layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

flat in vec2 vVel[];
flat in float vRot[];
flat in int vPartIdx[];

out vec2 uv;
flat out vec2 fVel;
flat out int fPartIdx;

uniform float scale;

const vec2 OFFSETS[4] = vec2[](
    vec2(-0.20, -0.20),
    vec2( 0.20, -0.20),
    vec2(-0.20,  0.20),
    vec2( 0.20,  0.20)
);

const vec2 UV_COORDS[4] = vec2[](
    vec2(0, 0),
    vec2(1, 0),
    vec2(0, 1),
    vec2(1, 1)
);

void main()
{
    const mat2 ROT = mat2(
        cos(vRot[0]), -sin(vRot[0]),
        sin(vRot[0]),  cos(vRot[0])
    );

    fVel = vVel[0];
    for (int i = 0; i < 4; i++) {
        gl_Position = gl_in[0].gl_Position + scale * vec4(ROT * OFFSETS[i], 0.0, 0.0);
        uv = UV_COORDS[i];
        EmitVertex();
    }
    EndPrimitive();
}
