import os

class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.faces = []
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
                '''elif values[0] == 'mtllib':
                    # Attempt to load the material file, but handle the case where it's missing
                    mtl_file = os.path.join(dir_name, values[1])
                    if os.path.exists(mtl_file):
                        self.mtl = MTL(mtl_file)
                    else:
                        print(f"Warning: Material file {mtl_file} not found.")'''
