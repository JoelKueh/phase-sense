#version 430 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec2 FragVelocity;

flat in vec2 fVel;

void main()
{
    FragColor = vec4(0.0, 0.0, 0.0, 0.5);
    FragVelocity = fVel;
}
