import smplx as smpl
import torch
import os
import numpy as np
import numpy as np
from sklearn.preprocessing import normalize

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
def smpl_mesh(parameters):
	model_params = dict(model_path="/DATA/Projects/speech_new/TalkSHOW/visualise/",
						model_type='smplx',
						create_global_orient=True,
						create_body_pose=True,
						create_betas=True,
						num_betas=300,
						create_left_hand_pose=True,
						create_right_hand_pose=True,
						use_pca=False,
						flat_hand_mean=False,
						create_expression=True,
						num_expression_coeffs=100,
						num_pca_comps=12,
						create_jaw_pose=True,
						create_leye_pose=True,
						create_reye_pose=True,
						create_transl=False,
						# gender='ne',
						dtype=torch.float64 )

	smplx_model = smpl.create(**model_params).to("cuda")

	betas = torch.zeros([1, 300], dtype=torch.float64).to('cuda')
	output = smplx_model(betas = betas.cuda(),
						expression=parameters[165:265].unsqueeze_(dim=0).cuda(),
                                 jaw_pose=parameters[0:3].unsqueeze_(dim=0).cuda(),
                                 leye_pose=parameters[3:6].unsqueeze_(dim=0).cuda(),
                                 reye_pose=parameters[6:9].unsqueeze_(dim=0).cuda(),
                                 global_orient=parameters[9:12].unsqueeze_(dim=0).cuda(),
                                 body_pose=parameters[12:75].unsqueeze_(dim=0).cuda(),
                                 left_hand_pose=parameters[75:120].unsqueeze_(dim=0).cuda(),
                                 right_hand_pose=parameters[120:165].unsqueeze_(dim=0).cuda(),
                                 return_verts=True)
	vertices = output.vertices.detach().cpu().numpy().squeeze()
	return vertices

def ComputeNormal(vertices, trifaces):
    vertices = np.expand_dims(vertices, axis = 0)
    if vertices.shape[0] > 5000:
        print('ComputeNormal: Warning: too big to compute {0}'.format(vertices.shape) )
        return

    #compute vertex Normals for all frames
    U = vertices[:,trifaces[:,1],:] - vertices[:,trifaces[:,0],:]  #frames x faceNum x 3
    V = vertices[:,trifaces[:,2],:] - vertices[:,trifaces[:,1],:]  #frames x faceNum x 3
    originalShape = U.shape  #remember: frames x faceNum x 3

    U = np.reshape(U, [-1,3])
    V = np.reshape(V, [-1,3])
    faceNormals = np.cross(U,V) 
    faceNormals = normalize(faceNormals)

    faceNormals = np.reshape(faceNormals, originalShape)

    if False:        #Slow version
        vertex_normals = np.zeros(vertices.shape) #(frames x 11510) x 3
        for fIdx, vIdx in enumerate(trifaces[:,0]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,1]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,2]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
    else:   #Faster version
        # Computing vertex normals, much faster (and obscure) replacement
        index = np.vstack((np.ravel(trifaces), np.repeat(np.arange(len(trifaces)), 3))).T
        index_sorted = index[index[:,0].argsort()]
        vertex_normals = np.add.reduceat(faceNormals[:,index_sorted[:, 1],:][0],
            np.concatenate(([0], np.cumsum(np.unique(index_sorted[:, 0],
            return_counts=True)[1])[:-1])))[None, :]
        vertex_normals = vertex_normals.astype(np.float64)

    originalShape = vertex_normals.shape
    vertex_normals = np.reshape(vertex_normals, [-1,3])
    vertex_normals = normalize(vertex_normals)
    vertex_normals = np.reshape(vertex_normals,originalShape)

    return vertex_normals



smpl_v = smpl_mesh(torch.zeros(265))

smpl_v[:,1] = smpl_v[:,1] + 0.4 * np.ones(10475)

path = "/DATA/Projects/speech_new/TalkSHOW/visualise/smplx/SMPLX_NEUTRAL.npz"
model_data = np.load(path, allow_pickle=True)
data_struct = Struct(**model_data)
smpl_f = data_struct.f

smpl_n = ComputeNormal(smpl_v, smpl_f).squeeze()

g_xMousePtStart = 0
g_yMousePtStart = 0

g_colors = [ (0,0,255), (255,0,0), (0, 255, 127), (170, 170, 0), (0, 0, 128), (153, 50, 204), (60, 20, 220),
    (0, 128, 0), (180, 130, 70), (147, 20, 255), (128, 128, 240), (154, 250, 0), (128, 0, 0),
    (30, 105, 210), (0, 165, 255), (170, 178, 32), (238, 104, 123)]

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
    glShadeModel(GL_SMOOTH)
    global g_yRotate, g_xRotate, g_zRotate, g_xTrans, g_yTrans, g_zTrans
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
    glTranslatef(g_xTrans, 0.0, 0.0)
    glTranslatef(0.0, g_yTrans, 0.0)
    glTranslatef(0.0, 0.0, g_zTrans)
    
    # glColor3f(1, 1, 1)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)
    glLineWidth(2.0)
    for idx in smpl_f:
        glBegin(GL_TRIANGLES)
        color1 = smpl_n[idx[0]]*0.5 + 0.5
        color2 = smpl_n[idx[1]]*0.5 + 0.5
        color3 = smpl_n[idx[2]]*0.5 + 0.5
        glColor3f(color1[0], color1[1], color1[2])
        glNormal3f(smpl_n[idx[0]][0], smpl_n[idx[0]][1], smpl_n[idx[0]][2])
        glVertex3f(smpl_v[idx[0]][0], smpl_v[idx[0]][1], smpl_v[idx[0]][2])
        glColor3f(color2[0], color2[1], color2[2])
        glNormal3f(smpl_n[idx[1]][0], smpl_n[idx[1]][1], smpl_n[idx[1]][2])
        glVertex3f(smpl_v[idx[1]][0], smpl_v[idx[1]][1], smpl_v[idx[1]][2])
        glColor3f(color3[0], color3[1], color3[2])
        glNormal3f(smpl_n[idx[2]][0], smpl_n[idx[2]][1], smpl_n[idx[2]][2])
        glVertex3f(smpl_v[idx[2]][0], smpl_v[idx[2]][1], smpl_v[idx[2]][2])
        glEnd()
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

# glLightfv(GL_LIGHT0, GL_AMBIENT, g_ambientLight)
# glLightfv(GL_LIGHT0, GL_DIFFUSE, g_diffuseLight)
# glLightfv(GL_LIGHT0, GL_SPECULAR, g_specular)
glEnable(GL_LIGHT0)
glEnable(GL_CULL_FACE)

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

