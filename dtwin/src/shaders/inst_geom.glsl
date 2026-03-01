#version 430 core

layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

flat in vec2 vVel[];
flat in float vRot[];
flat in int vPartIdx[];
flat out vec2 fVel;

const vec2 OFFSETS[4] = vec2[](
    vec2(-0.20, -0.01),
    vec2( 0.20, -0.01),
    vec2(-0.20,  0.01),
    vec2( 0.20,  0.01)
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
