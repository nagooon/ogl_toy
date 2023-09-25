import pyrender
import trimesh
import torch
import smplx as smpl
import numpy as np
from PIL import Image


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


path = "/DATA/Projects/speech_new/TalkSHOW/visualise/smplx/SMPLX_NEUTRAL.npz"
model_data = np.load(path, allow_pickle=True)
data_struct = Struct(**model_data)
smpl_f = data_struct.f

t_mesh = trimesh.Trimesh(vertices= smpl_mesh(torch.zeros(265)), faces= smpl_f)
render_mesh = pyrender.Mesh.from_trimesh(t_mesh,
                                                 smooth=True,
                                                 material=pyrender.MetallicRoughnessMaterial(
                                                     metallicFactor=0.05,
                                                     roughnessFactor=0.7,
                                                     alphaMode='OPAQUE',
                                                     baseColorFactor=(0, 0, 0.5, 1.0)
                                                 ))
scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

scene.add(render_mesh, pose=np.eye(4))

xmag = 0.5
ymag = 0.5
camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],  # 0.25
                            [0, 0, 1, 0],  # 0.2
                            [0, 0, 0, 1]])
light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=2)
light_pose = np.eye(4)
light_pose[:3, 3] = [0, -1, 1]
scene.add(light, pose=light_pose)

light_pose[:3, 3] = [0, 1, 1]
scene.add(light, pose=light_pose)

light_pose[:3, 3] = [-1, 1, 2]
scene.add(light, pose=light_pose)

spot_l = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
							innerConeAngle=np.pi / 3, outerConeAngle=np.pi / 2)

light_pose[:3, 3] = [-1, 2, 2]
scene.add(spot_l, pose=light_pose)

light_pose[:3, 3] = [1, 2, 2]
scene.add(spot_l, pose=light_pose)

r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
flags = pyrender.RenderFlags.SKIP_CULL_FACES
color, _ = r.render(scene, flags)

image = color[..., ::-1]

Image.fromarray(image).save("temp.png")