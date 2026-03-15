#version 430 core

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec2 FragVelocity;

in vec2 uv;
flat in vec2 fVel;
flat in int fPartIdx;

const float k_freq = 20.0f;
const float k_decay = 20.0f;
const float k_time = 3.0f;
const float k_bright = 0.25f;

float sdf(float d)
{
    float d_scaled = d * k_time;
    float d2_scaled = d_scaled * d_scaled;
    float val = k_bright * cos(k_decay * d_scaled) * exp2(-k_decay * d2_scaled);
    return val;
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
    vec2 s1 = vec2(0.25f, 0.25f);
    vec2 s2 = vec2(0.75f, 0.75f);
    // FragColor = vec4(vec3(sdf(seg_dist(s1, s2, uv))), 1.0);
    FragColor = vec4(vec3(sdf(seg_dist(s1, s2, uv))), 0.0);
    // FragColor = vec4(seg_dist(s1, s2, uv), 0.0, 0.0, 1);
    // FragColor = vec4(0.1f, 0.1f, 0.1f, 1.0f);
    FragVelocity = fVel;
}
