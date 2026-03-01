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
    gl_Position = vec4(aPos, 0.0, 1.);
    vVel = aVel;
    vRot = aRot;
    vPartIdx = aPartIdx;
}
