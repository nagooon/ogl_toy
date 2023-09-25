import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *

pygame.init()
pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)

vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
void main(){
    gl_Position = vec4(position, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 color;
void main(){
    color = vec4(1.0, 0.0, 0.0, 1.0); // Red color
}
"""

vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, vertex_shader_source)
glCompileShader(vertex_shader)

fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, fragment_shader_source)
glCompileShader(fragment_shader)

shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)
glUseProgram(shader_program)

vertices = [-0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.0, 0.5, 0.0]

vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, (GLfloat * len(vertices))(*vertices), GL_STATIC_DRAW)

# Specify the vertex attribute layout
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)

while True:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)
    pygame.display.flip()
    pygame.time.wait(10)