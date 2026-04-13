#version 430 core

#define SPINE_LEN 17

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec2 FragVelocity;

struct render_spine_t {
	float px[SPINE_LEN];
	float py[SPINE_LEN];
};

layout (std430, binding = 0) readonly buffer SpineUBO
{
    render_spine_t spines[];
};

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
    float dist = 100.0f, new_dist;
    vec2 s1, s2;
    int i;

    for (i = 0; i < SPINE_LEN-1; i++ ) {
        s1 = vec2(spines[fPartIdx].px[i], spines[fPartIdx].py[i]);
        s2 = vec2(spines[fPartIdx].px[i+1], spines[fPartIdx].py[i+1]);
        new_dist = seg_dist(s1, s2, uv);
        dist = new_dist < dist ? new_dist : dist;
    }
    FragColor = vec4(vec3(sdf(dist)), 0.0);
    FragVelocity = fVel;
}
