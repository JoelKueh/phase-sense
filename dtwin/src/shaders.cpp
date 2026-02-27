
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
// extern const char frag_src_blur[] = R"(
// #version 430 core

// in vec2 uv;
// out vec4 FragColor;
// uniform sampler2D tex;

// const float WEIGHTS[25] = float[](
//     1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
//     1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
//     1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
//     1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0,
//     1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0, 1.0/25.0
// );

// const ivec2 OFFSETS[25] = ivec2[](
//     vec2(-2, 2), vec2(-1, 2), vec2( 0, 2), vec2( 1, 2), vec2( 2, 2),  
//     vec2(-2, 1), vec2(-1, 1), vec2( 0, 1), vec2( 1, 1), vec2( 2, 1),  
//     vec2(-2, 0), vec2(-1, 0), vec2( 0, 0), vec2( 1, 0), vec2( 2, 0),  
//     vec2(-2,-1), vec2(-1,-1), vec2( 0,-1), vec2( 1,-1), vec2( 2,-1),  
//     vec2(-2,-2), vec2(-1,-2), vec2( 0,-2), vec2( 1,-2), vec2( 2,-2),  
// );

// void main()
// {
//     int i, j;
//     vec3 color;
//     for (i = 0; i < 5; i++) {
//         for (j = 0; j < 5; j++) {
//             color += WEIGHTS[i*5 + j] * 
//         }
//     }
//     FragColor=vec4(, 1.0);
// } 

// Credit to:
// https://github.com/mattdesl/lwjgl-basics/wiki/ShaderLesson5
extern const char frag_src_blur[] = R"(
#version 430 core
out vec4 FragColor;
in vec2 uv;

uniform sampler2D u_texture;
uniform float resolution;
uniform float radius;
uniform vec2 dir;

void main() {
	//this will be our RGBA sum
	vec4 sum = vec4(0.0);
	
	//our original texcoord for this fragment
	vec2 tc = uv;
	
	//the amount to blur, i.e. how far off center to sample from 
	//1.0 -> blur by one pixel
	//2.0 -> blur by two pixels, etc.
	float blur = radius/resolution; 
    
	//the direction of our blur
	//(1.0, 0.0) -> x-axis blur
	//(0.0, 1.0) -> y-axis blur
	float hstep = dir.x;
	float vstep = dir.y;
    
	//apply blurring, using a 9-tap filter with predefined gaussian weights
    
	sum += texture(u_texture, vec2(tc.x - 4.0*blur*hstep, tc.y - 4.0*blur*vstep)) * 0.0162162162;
	sum += texture(u_texture, vec2(tc.x - 3.0*blur*hstep, tc.y - 3.0*blur*vstep)) * 0.0540540541;
	sum += texture(u_texture, vec2(tc.x - 2.0*blur*hstep, tc.y - 2.0*blur*vstep)) * 0.1216216216;
	sum += texture(u_texture, vec2(tc.x - 1.0*blur*hstep, tc.y - 1.0*blur*vstep)) * 0.1945945946;
	
	sum += texture(u_texture, vec2(tc.x, tc.y)) * 0.2270270270;
	
	sum += texture(u_texture, vec2(tc.x + 1.0*blur*hstep, tc.y + 1.0*blur*vstep)) * 0.1945945946;
	sum += texture(u_texture, vec2(tc.x + 2.0*blur*hstep, tc.y + 2.0*blur*vstep)) * 0.1216216216;
	sum += texture(u_texture, vec2(tc.x + 3.0*blur*hstep, tc.y + 3.0*blur*vstep)) * 0.0540540541;
	sum += texture(u_texture, vec2(tc.x + 4.0*blur*hstep, tc.y + 4.0*blur*vstep)) * 0.0162162162;

	//discard alpha for our simple demo, multiply by vertex color and return
	FragColor = vec4(sum.rgb, 1.0);
}
)";
