import numpy as np
import cv2


class Map():
    def __init__(self, trajectory_gt=None):
        self.trajectory = np.array([[0., 0., 0.]], dtype=np.float32)
        self.trajectory_gt = trajectory_gt
        self.landmarks_bev = None
        self.poses = [np.eye(4)]
        self.x_range = (-500, 500)
        self.z_range = (-500, 500)
        self.viz_width = self.x_range[1] - self.x_range[0]
        self.viz_height = self.z_range[1] - self.z_range[0]
        self.viz_img_map = None
        self.viz_xz_center = np.array([self.viz_width / 2, self.viz_height / 2])
        self.viz_pos_last = 0
        self.vis_trajectory_color = np.array([0, 255, 0])
        self.vis_trajectory_color_gt = np.array([0, 0, 255])
        self.vis_landmarks_color = np.array([255, 100, 55])

        # 真値を可視化したところ、初期姿勢のZは負の方向に向いているようだったのでそれに合わせる。
        self.poses[0][2, 2] = -1

    def append(self, pose_curr, points_4d):
        # add trajectory
        position_curr = pose_curr[:, 3][:-1]
        self.poses.append(pose_curr)
        self.trajectory = np.append(self.trajectory, position_curr[np.newaxis], axis=0)

        # add landmarks
        points_bev = points_4d[:, [0, 2]] # x, z
        points_bev = points_bev.astype(np.int32)
        if self.landmarks_bev is None:
            self.landmarks_bev = points_bev
        else:
            # debugging
            self.landmarks_bev = points_bev
            # self.landmarks_bev = np.concatenate([self.landmarks_bev, points_bev], axis=0)
        self.landmarks_bev = np.unique(self.landmarks_bev, axis=0)

    def update_vis(self):
        self.viz_img_map = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)
        self.update_vis_trajectory()
        # trajectoryは１画素の点を３画素に大きくして見やすくする
        self.viz_img_map = cv2.dilate(self.viz_img_map, np.ones((3, 3)), iterations=1)
        self.update_vis_landmarks()

    def update_vis_trajectory(self):
        # TopView表示（X,Zのみ表示）、画像の中央を原点（０，０）として表示するために座標をシフトする
        trajectory_xz = self.trajectory[:, [0, 2]]
        trajectory_xz += self.viz_xz_center
        trajectory_xz = trajectory_xz.astype(np.int32)
        trajectory_x = trajectory_xz[:, 0]
        trajectory_z = trajectory_xz[:, 1]

        # 画像外の点を除外しておく（除外しないと例外が起きるため）
        x_valid = np.logical_and(trajectory_x >= 0, trajectory_x < self.viz_width)
        z_valid = np.logical_and(trajectory_z >= 0, trajectory_z < self.viz_height)
        valid = np.logical_and(x_valid, z_valid)

        self.viz_img_map[trajectory_z[valid], trajectory_x[valid]] = self.vis_trajectory_color

        if self.trajectory_gt is not None:
            # 与えられていれば真値も表示する
            trajectory_gt_xz = self.trajectory_gt[:len(self.trajectory)][:, [0, 2]]
            trajectory_gt_xz += self.viz_xz_center
            trajectory_gt_x = trajectory_gt_xz[:, 0].astype(np.int32)
            trajectory_gt_z = trajectory_gt_xz[:, 1].astype(np.int32)
            self.viz_img_map[trajectory_gt_z, trajectory_gt_x] = self.vis_trajectory_color_gt

    def update_vis_landmarks(self):
        # TopView表示（X,Zのみ表示）、画像の中央を原点（０，０）として表示するために座標をシフトする
        landmarks_xz = self.landmarks_bev.copy()
        landmarks_xz += self.viz_xz_center.astype(np.int32)
        landmarks_x = landmarks_xz[:, 0]
        landmarks_z = landmarks_xz[:, 1]

        # 画像外の点を除外しておく（除外しないと例外が起きるため）
        x_valid = np.logical_and(landmarks_x >= 0, landmarks_x < self.viz_width)
        z_valid = np.logical_and(landmarks_z >= 0, landmarks_z < self.viz_height)
        valid = np.logical_and(x_valid, z_valid)

        self.viz_img_map[landmarks_z[valid], landmarks_x[valid]] = self.vis_landmarks_color

    def show(self):
        self.update_vis()
        cv2.imshow("trajectory", self.viz_img_map)
        cv2.waitKey(0)
