import numpy as np


class GroundTruth:

    def __init__(self, key_frame_rate, path_camera_track="data/NewTsukubaStereoDataset/groundtruth/camera_track.txt"):
        self.data = self.load_camera_track(path_camera_track)
        self.key_frame_rate = key_frame_rate

    def load_camera_track(self, path):
        with open(path) as f:
            lines = f.readlines()
            data = [line.strip().split(',') for line in lines]
            data = [[float(val.strip()) for val in row] for row in data]
        return data

    def __getitem__(self, idx):
        """
        returns:
        * current pose (position, rotation) in world coordinate
        * translation (X, Y, Z)
        * absolute scale of translation from previsious frame
        """
        item = self.data[idx]
        x, y, z, a, b, c = item[0:6]

        if idx != 0:
            item_prev = self.data[idx - self.key_frame_rate]
            x_p, y_p, z_p, a_p, b_p, c_p = item_prev[0:6]
            trans = np.array([x - x_p, y - y_p, z - z_p])
            scale_abs = np.linalg.norm(trans)
            rotation = np.array([a - a_p, b - b_p, c - c_p])
        else:
            trans = np.array([0., 0., 0.])
            scale_abs = 0.0
            rotation = np.array([0.0, 0.0, 0.0])
        return {
            "position_abs": [x, y, z],
            "rotatoin_abs": [a, b, c],
            "rotation": rotation,
            "trans": trans,
            "trans_abs_scale": scale_abs,
        }

    def __len__(self):
        return len(self.data)
