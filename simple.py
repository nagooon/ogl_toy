import smplx as smpl
import torch
import os
import numpy as np
import numpy as np
from sklearn.preprocessing import normalize

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

g_xMousePtStart = 0
g_yMousePtStart = 0


def mouse(button, state, x, y):
    global g_xMousePtStart, g_yMousePtStart
    g_xMousePtStart = x
    g_yMousePtStart = y

def motion(x, y):
    global g_xRotate, g_yRotate, g_xMousePtStart, g_yMousePtStart
    g_xRotate += 0.01*(x - g_xMousePtStart)
    g_yRotate += 0.01*(y - g_yMousePtStart)

def keyboard(key, x, y):
    if isinstance(key, bytes):
        key = key.decode()
    global g_zTrans
    if key == 'z':
        print("Z pressed!")
        g_zTrans += 0.1
    if key == 'x':
        print("X pressed!")
        g_zTrans -= 0.1

def reshape(width,height):
    glViewport(0, 0, width, height)

def renderscene():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    global g_yRotate, g_xRotate, g_zRotate, g_xTrans, g_yTrans, g_zTrans
    glShadeModel(GL_SMOOTH)
    glEnable (GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable (GL_LINE_SMOOTH)
    glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_MULTISAMPLE)
    glLoadIdentity()
    
    # gluLookAt(0,0,0, 0, 0, 1, 0, -1, 0)
    # glMatrixMode(GL_PROJECTION)
    # gluPerspective(65, float(1920)/float(1080), 0.01, 9000)
    
    # glMatrixMode(GL_MODELVIEW)
    # glTranslatef(0, 0, 600)
    glRotatef(-g_yRotate, 1.0, 0.0, 0.0)
    glRotatef(-g_xRotate, 0.0, 1.0, 0.0)
    glRotatef(g_zRotate, 0.0, 0.0, 1.0)
    
    # glColor3f(1, 1, 1)
    glPolygonMode(GL_FRONT, GL_FILL)
    # glPolygonMode(GL_BACK, GL_FILL)
    glLineWidth(2.0)
    glColor3f(1, 1, 1)
    # glBegin(GL_TRIANGLES)
    # glVertex3f(0,0,0)
    # glVertex3f(0.5, 0, 0)
    # glVertex3f(0.5, 0.5, 0)
    # glEnd()
    glutSolidTeapot(0.5)
    glutSwapBuffers()
    return

g_xRotate = 0
g_yRotate = 0
g_zRotate = 0

g_xTrans = 0
g_yTrans = 0
g_zTrans = 0

g_ambientLight = (0.35, 0.35, 0.35, 1.0)
g_diffuseLight = (0.75, 0.75, 0.75, 0.7)
g_specular = (0.2, 0.2, 0.2, 1.0)
g_specref = (0.5, 0.5, 0.5, 1.0)

glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
glutInitWindowPosition(100,100)
glutInitWindowSize(1920,1920)

glClearColor(1.0, 1.0, 1.0, 1.0)

glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LESS)
glShadeModel(GL_SMOOTH)


g_winID = glutCreateWindow("3D View")


# glutKeyboardFunc()
# glutMouseFunc()
# glutMotionFunc()


glEnable(GL_LIGHTING)

glLightfv(GL_LIGHT0, GL_AMBIENT, g_ambientLight)
glLightfv(GL_LIGHT0, GL_DIFFUSE, g_diffuseLight)
glLightfv(GL_LIGHT0, GL_SPECULAR, g_specular)
glEnable(GL_LIGHT0)
glEnable(GL_CULL_FACE)
# glFrontFace(GL_CCW)
# glCullFace(GL_FRONT)

glEnable(GL_COLOR_MATERIAL)
glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
glMaterialfv(GL_FRONT, GL_SPECULAR, g_specref)
glMateriali(GL_FRONT, GL_SHININESS, 128)

glutReshapeFunc(reshape)
glutDisplayFunc(renderscene)
glutMouseFunc(mouse)
glutMotionFunc(motion)
glutKeyboardFunc(keyboard)
glutIdleFunc(renderscene)
glutMainLoop()

