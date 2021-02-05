/* Copyright (C) 2020 José María Cruz Lorite
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include "util/lodepng.h"

// Set uniform handy macro
#define uniform(program, count, value, uniform, type)\
    glUniform ## type(glGetUniformLocation(program, uniform), value)
#define uniformv(program, count, value, uniform, type)\
    glUniform ## type ## v(glGetUniformLocation(program, uniform), count, value)

// The GLFW window
GLFWwindow* window = nullptr;

// ComputeShader
const char* shaderSource = R"(
#version 450

layout (binding = 0, rgba8) uniform writeonly image2D img;
layout (binding = 1, rgba8) uniform readonly  image1D palette;

uniform vec2 size;         // Image width - 1, height - 1
uniform int  paletteSize;  // Palette size - 1
   
uniform vec2 cmin;         // Complex plane bottom left point
uniform vec2 cdelta;       // Complex plane visualization range

uniform int   maxDepth;    // Maximun number of iterations - 1
uniform float scaleForce;  // Controls how color palette is applied

// Work group size
layout (local_size_x = 32, local_size_y = 32) in;
void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy); // This pixel (x, y)
  vec2 c = cdelta * pixel / size + cmin;         // Interpolate c

  int  depth = 0;
  vec2 z     = c;
  vec2 z2    = c * c;
  for(; depth <= maxDepth && z2.x + z2.y <= 4.0f; ++depth) {
    z  = vec2(z2.x - z2.y, 2 * z.x * z.y) + c;
    z2 = z * z;
  }

  if (depth > maxDepth)
    // Black color if c is on mandelbrot set
    imageStore(img, pixel, vec4(0.0f, 0.0f, 0.0f, 1.0f));
  else {
    float i  = float(depth) / maxDepth;
    float ie = min(1.0f, log(i * scaleForce + 1.0f) / log(scaleForce));

    int color = int(round((1.0f - ie) * paletteSize));
    imageStore(img, pixel, imageLoad(palette, color));
  }
}
)";

// Initialize GLFW
void initGLFWOffScreen() {
    if (glfwInit() != GL_TRUE) {
        std::cerr << "GLFW initialization failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Application & window init
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // Create the window
    window = glfwCreateWindow(1, 1, "", 0, 0);
    glfwMakeContextCurrent(window);
}

// Destroy GLFW
void destroyGLFWOffScreen() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

// Print device info
void printDeviceInfo() {
    std::cout << glGetString(GL_VENDOR)   << ", " <<
                 glGetString(GL_RENDERER) << ", " <<
                 glGetString(GL_VERSION)  << std::endl;
    GLint value[3];
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &value[0]); 
    std::cout << "Max work-group invocations: " << value[0] << std::endl;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &value[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &value[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &value[2]);
    std::cout << "Max work-group count:       " << value[0] << " x " << value[1] << " x " << value[2] << std::endl;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &value[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &value[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &value[2]);
    std::cout << "Max work-group size:        " << value[0] << " x " << value[1] << " x " << value[2] << std::endl;
}

// Check gl error
GLenum __checkOpenGLError(const char *file, int line) {
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR) {
        std::string error;
        switch (errorCode) {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define checkOpenGLError() __checkOpenGLError(__FILE__, __LINE__)

// Texture enum
enum Texture : uint {
    TEXTURE_IMG = 0,
    TEXTURE_PALETTE = 1
};

int main() {
    /******** PARAMETERS *********/
    GLint   localGroupSize = 32;
    GLint   maxDepth       = 1024;
    GLint   width          = 1024 * 8;
    GLint   height         = 1024 * 8;
    GLfloat scaleForce     = 20.0f;
    /*****************************/

    // Allocate space for the image on host
    std::vector<GLubyte> img(width * height * 4);

    // Load color palette
    unsigned int w_palette = 0;
    unsigned int h_palette = 0;
    std::vector<GLubyte> palette;
    lodepng::decode(palette, w_palette, h_palette, "palette.png");

    // Complex plane min max
    float min_real = -2.0;
    float max_real = 1.0f;
    float min_imag = ((max_real - min_real) * height / width) / -2.0f;
    float max_imag = -1.0f * min_imag;

    GLfloat size[2]  = {(GLfloat)width - 1, (GLfloat)height - 1};
    GLfloat min[2]   = {min_real, min_imag};
    GLfloat delta[2] = {max_real - min_real, max_imag - min_imag};

    // Init glfw
    initGLFWOffScreen();

    // Glad init
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Error  loading glad." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Print device info
    printDeviceInfo();
    std::cout << std::endl;

    // Get start time
    auto start = std::chrono::high_resolution_clock::now();

    // Create shader program
    GLuint shaderObj = glCreateShader(GL_COMPUTE_SHADER);
    GLuint shaderProgram = glCreateProgram();

    // For simplicity no compilation errores are checked
    // Compile shader and create program
    glShaderSource(shaderObj, 1, &shaderSource, NULL);
    glCompileShader(shaderObj);
    glAttachShader(shaderProgram, shaderObj);
    glLinkProgram(shaderProgram);

    // Create textures
    GLuint textures[2];
    glGenTextures(2, textures);

    // Initialize image texture
    glBindTexture(GL_TEXTURE_2D, textures[TEXTURE_IMG]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

    // Initialize palette texture
    glBindTexture(GL_TEXTURE_1D, textures[TEXTURE_PALETTE]);
    glTexStorage1D(GL_TEXTURE_1D, 1, GL_RGBA8, h_palette);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, h_palette, GL_RGBA, GL_UNSIGNED_BYTE, palette.data());
    
    // Use program shader
    glUseProgram(shaderProgram);

    // Set outout textures
    glBindImageTexture(0, textures[TEXTURE_IMG], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glBindImageTexture(1, textures[TEXTURE_PALETTE], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);

    // Set parameters
    uniformv(shaderProgram, 1, size,          "size",        2f);
    uniform( shaderProgram, 1, h_palette - 1, "paletteSize", 1i);
    uniformv(shaderProgram, 1, min,           "cmin",        2f);
    uniformv(shaderProgram, 1, delta,         "cdelta",      2f);
    uniform( shaderProgram, 1, scaleForce,    "scaleForce",  1f);
    uniform( shaderProgram, 1, maxDepth - 1,  "maxDepth",    1i);

    glDispatchCompute(width / localGroupSize, height / localGroupSize, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    // Move texture from device to host
    glBindTexture(GL_TEXTURE_2D, textures[TEXTURE_IMG]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data());

    // Free Resources
    glDeleteShader(shaderObj);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(2, textures);

    // Get end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f <<
        " seconds" << std::endl;

    // Destroy glfw
    destroyGLFWOffScreen();

    // To file
    lodepng::encode("mandelbrot.png", img.data(), width, height);

    exit(EXIT_SUCCESS);
}
