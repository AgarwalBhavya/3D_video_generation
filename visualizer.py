import trimesh
import pyrender

def visualize_mesh(mesh):
    scene = pyrender.Scene()
    mesh_pyr = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyr)
    pyrender.Viewer(scene, use_raymond_lighting=True)
