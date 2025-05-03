import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
from utils.visualizer import visualize_mesh

def generate_from_image(image_path):
    img = Image.open(image_path)
    img_nobg = remove(img)  # remove background
    img_array = np.array(img_nobg)

    # Depth estimation (you can use MiDaS model)
    depth = np.ones((img_array.shape[0], img_array.shape[1])) * 0.5  # Placeholder

    # Convert to mesh (mock cube mesh)
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.export("output/generated_from_image.stl")

    visualize_mesh(mesh)
