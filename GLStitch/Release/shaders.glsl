const char* vertexShaderSource = R"(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec2 myCoords;

out vec4 vertexColor;
out vec2 texCoords;
uniform mat4 u_MVP;

void main()
{
	gl_Position = u_MVP * vec4(aPos, 1.0f);
	vertexColor = aColor;
	texCoords = myCoords;
}
)";
const char* fragmentShaderSource = R"(
#version 460 core
out vec4 FragColor;

in vec4 vertexColor;
in vec2 texCoords;

uniform sampler2D myTex;
uniform float scale_theta;
uniform float scale_radius;
uniform float offset_theta;
uniform float offset_radius;

void main()
{
	float center_x = 1920 / 2;
	float center_y = 1080 / 2;
	float f = 1500.0f;
	vec2 textureSize = vec2(1920, 1080);
	vec2 normTexCoord = texCoords * textureSize;
    
    // Calculate the angle theta
    float theta = (normTexCoord.x - center_x) / f;
    
    // Compute the cylindrical coordinates
    float x_cyl = f * tan(theta) + center_x;
    float y_cyl = (normTexCoord.y - center_y) / cos(theta) + center_y;

    // Normalize the cylindrical coordinates back to the range [0, 1]
    vec2 warpedTexCoord = vec2(x_cyl / textureSize.x, y_cyl / textureSize.y);
    
    // Sample the texture with the new coordinates
    FragColor = texture(myTex, warpedTexCoord);
}
)";