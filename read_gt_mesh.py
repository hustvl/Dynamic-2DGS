import os
import numpy as np
from collections import defaultdict

def load_obj(obj_file):
    """
    Load vertices and faces from an OBJ file.
    
    Args:
        obj_file (str): Path to the OBJ file.
        
    Returns:
        verts (np.ndarray): Array of vertex coordinates.
        faces (np.ndarray): Array of face indices.
    """
    verts = []
    faces = []
    
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex line
                vert = [float(x) for x in line.split()[1:4]]
                verts.append(vert)
            elif line.startswith('f '):
                # Face line
                face_indices = line.split()[1:4]
                face = []
                for index in face_indices:
                    vertex_index = int(index.split('/')[0]) - 1
                    face.append(vertex_index)
                faces.append(face)
    
    verts = np.array(verts)
    faces = np.array(faces)
    
    return verts, faces

def load_obj_mtl(obj_file, mtl_file):
    # Load the OBJ file
    vertices = []
    faces = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x) - 1 for x in line.split()[1:4]])

    # Load the MTL file
    materials = defaultdict(lambda: {'Kd': [1, 1, 1]})
    current_material = None
    with open(mtl_file, 'r') as f:
        for line in f:
            if line.startswith('newmtl '):
                current_material = line.split()[1]
            elif line.startswith('Kd '):
                materials[current_material]['Kd'] = [float(x) for x in line.split()[1:4]]

    # Combine vertices and materials
    vertex_colors = []
    for face in faces:
        material = materials[current_material]
        vertex_colors.extend([material['Kd']] * 3)

    vertices = np.array(vertices)
    faces = np.array(faces)
    vertex_colors = np.array(vertex_colors)

    return vertices, faces, vertex_colors

# Example usage
obj_file = '/data3/zhangshuai/DG-Mesh/data/dg-mesh/bird/mesh_gt/bluebird_animated0.obj'
mtl_file = '/data3/zhangshuai/DG-Mesh/data/dg-mesh/bird/mesh_gt/bluebird_animated0.mtl'
vertices, faces = load_obj(obj_file)
print(vertices)