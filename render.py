import cv2
import numpy as np
import math
import os
from objectloader import *
from time import sleep

ERROR_DIST = 150
MIN_MATCHES = 120
DEFAULT_COLOR = (0, 0, 255)
TARGET = ""
models = ['target_H20.jpg', 'NaCl_target.jpg', 'FePt.png']
materials = ['H2O.mtl', 'H2O.mtl', 'H2O.mtl']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Azul, Verde, Vermelho, Amarelo


def render(frame, obj, projection):
    h, w = frame.shape[:2]
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3

    # Pré-calcular pontos escalados e centralizados
    scaled_points = np.dot(vertices, scale_matrix)
    centered_points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in scaled_points])

    # Cache de materiais
    '''material_cache = {}
    if obj.mtl:
        for material_name, material_props in obj.mtl.contents.items():
            diffuse_color = material_props.get('Kd', obj.color)
            material_cache[material_name] = tuple([int(c * 255) for c in diffuse_color])
    '''
    i = 0
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([centered_points[vertex - 1] for vertex in face_vertices])

        # Realizar a projeção 3D-para-2D
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)  # Converter pontos para inteiros para desenhar

        # Obter o material para a face atual, se disponível
        #material = face[3] if len(face) == 4 else None

        '''if color and material and material in material_cache:
            color = material_cache[material]
        else:
        
            color = DEFAULT_COLOR
        '''
        #color = DEFAULT_COLOR
        color = obj.face_color[i]
        i+=1
        #color = color2
        # Desenhar a face no frame
        cv2.fillConvexPoly(frame, imgpts, color)

    return frame

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)