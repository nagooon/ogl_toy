import torch
import os
import numpy as np
import numpy as np
import smplx as smpl

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

smpl_v = smpl_mesh(torch.zeros(265))

smpl_v[:,1] = smpl_v[:,1] + 0.4 * np.ones(10475)

path = "/DATA/Projects/speech_new/TalkSHOW/visualise/smplx/SMPLX_NEUTRAL.npz"
model_data = np.load(path, allow_pickle=True)
data_struct = Struct(**model_data)
smpl_f = data_struct.f

def reshape(width,height):
    glViewport(0, 0, width, height)

angle = 0

def renderscene():
    global angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(angle,0.0,1.0,0.0)
    glBegin(GL_TRIANGLES)
    for idx in smpl_f:
        v1 = idx[0]
        v2 = idx[1]
        v3 = idx[2]
        glVertex3f(smpl_v[v1][0], smpl_v[v1][1], smpl_v[v1][2])
        glVertex3f(smpl_v[v2][0], smpl_v[v2][1], smpl_v[v2][2])
        glVertex3f(smpl_v[v3][0], smpl_v[v3][1], smpl_v[v3][2])
    glEnd()
    glutSwapBuffers()
    angle+=0.01
    return


glutInit()
glutInitDisplayMode(GLUT_DEPTH|GLUT_DOUBLE|GLUT_RGBA)
glutInitWindowPosition(100, 100)
glutInitWindowSize(1920, 1920)
glutCreateWindow("Testing")
glutDisplayFunc(renderscene)
glutIdleFunc(renderscene)
glutReshapeFunc(reshape)
glEnable(GL_DEPTH_TEST)
glutMainLoop()