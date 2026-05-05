#version 430 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aVel;
layout (location = 2) in float aRot;
layout (location = 3) in float aRvel;
layout (location = 4) in int aPartIdx;
layout (location = 5) in float aIntensity;

flat out vec2 vVel;
flat out float vRot;
flat out float vRvel;
flat out int vPartIdx;
flat out float vIntensity;

uniform float scale;

void main()
{
    gl_Position = vec4(aPos[0] * scale, -aPos[1] * scale, 0.0, 1.0);
    vVel = aPos;
    vRot = aRot;
    vRvel = aRvel;
    vPartIdx = aPartIdx;
    vIntensity = aIntensity;
}
