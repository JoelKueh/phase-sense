#version 430 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec2 FragVelocity;

in vec2 uv;
flat in vec2 fVel;
flat in int fPartIdx;

const float a = 20.0f;
const float b = 10.0f;
const float c = 2.0f;

float sdf(float d)
{
    // float val = 1.0f + cos(a * c * dist) * exp2(-b * c * c * dist * dist);
    // val = clamp(val, 0.0f, 1.0f);
    // return val;

    return -uv.x;
}

// Distance to a line segment through points s1 s2.
float seg_dist(vec2 s1, vec2 s2, vec2 p)
{
    float t = dot(p - s1, s2 - s1) / dot(s2 - s1, s2 - s1);
    t = min(max(t, 0), 1);
    return length((s1 + t * (s2 - s1)) - p);
}

void main()
{
    vec2 s1 = vec2(1.0f, 0.0f);
    vec2 s2 = vec2(0.0f, 1.0f);
    // FragColor = vec4(vec3(sdf(seg_dist(s1, s2, uv))), 1.0);
    FragColor = vec4(vec3(sdf(seg_dist(s1, s2, uv))), 1.0);
    // FragColor = vec4(seg_dist(s1, s2, uv), 0.0, 0.0, 1);
    // FragColor = vec4(0.0f, 1.0f, 1.0f, 1.0f);
    FragVelocity = fVel;
}
