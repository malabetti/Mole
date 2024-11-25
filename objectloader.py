import os

class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.faces = []
        self.face_color = []
        self.current_color = (0, 255, 0)
        #self.mtl = None

        dir_name = os.path.dirname(filename)
        with open(filename, "r") as f:
            for line in f:
                values = line.split()
                if not values:
                    continue
                if values[0] == 'v':
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == 'f':
                    face = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]))
                    self.faces.append((face, values[1:]))
                    self.face_color.append(self.current_color)
                elif values[0] == 'o':
                    if 'Hydrogen' in values[1]:
                        self.current_color = (255, 255, 255)
                    elif 'Oxygen' in values[1]:
                        self.current_color = (0, 0, 255)
                    elif 'Iron' in values[1]:
                        self.current_color = (128, 128, 128)
                    elif 'Platinum' in values[1]:
                        self.current_color = (205, 127, 50)
                    elif 'Sodium' in values[1]:
                        self.current_color = (107, 63, 160)
                    elif 'Chlorine' in values[1]:
                        self.current_color = (0, 128, 0)
                    elif 'Fluorine' in values[1]:
                        self.current_color = (0, 255, 0)
                    elif 'Calcium' in values[1]:
                        self.current_color = (255, 0, 0)
                    elif 'Phosphorus' in values[1]:
                        self.current_color = (246, 120, 40)
                '''elif values[0] == 'mtllib':
                    # Attempt to load the material file, but handle the case where it's missing
                    mtl_file = os.path.join(dir_name, values[1])
                    if os.path.exists(mtl_file):
                        self.mtl = MTL(mtl_file)
                    else:
                        print(f"Warning: Material file {mtl_file} not found.")'''
