import numpy as np
import cv2
from pathlib import Path
from typing import Tuple


"""
OpenCVを使いVisual Odometryを実装する。
評価用のデータセット: New Tsukuba Stereo Dataset
"""


def main():

    # カメラの内部パラメタをREADME.mdに記載された以下の 内容から定義した
    # ------
    # The resolution of the images is 640x480 pixels, the baseline of the stereo
    # camera is 10cm and the focal length of the camera is 615 pixels.
    K = np.array([
        [615., 0., 319.5],
        [0., 615., 239.5],
        [0.,   0.,   1.]
    ], dtype=np.float32)

    # ルーカス・カナデ法によるオプティカルフローのためのパラメタ
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    np.set_printoptions(precision=1, suppress=True)

    # 処理するフレーム数を指定する。値は１から始まる整数。
    # 大きくなるほどに計算が高速になるが、オプティカルフローによる特徴点追跡が失敗する可能性が高まるトレードオフが発生する点に注意。
    frame_reduction_rate = 1

    # カメラの姿勢検出を行う前に何回トラッキングを行うかを指定する
    # トラッキングの回数が少ない場合は姿勢の変化量が少ないために推定された姿勢が不安定なる
    # 一方で多いと追跡点の多くが画像からはずれてしまうことになる。
    num_tracking_before_pose_estimation = 5
    key_frame_rate = frame_reduction_rate * num_tracking_before_pose_estimation

    # 真値（ground truth）を扱うクラス
    # ここでは推定された値と真値を見比べて精度を比較するために使われている。
    # また、現在はスケールを合わせるためにも使っているが、これはインチキなのでいずれ正す予定
    groundtruth = GroundTruth(key_frame_rate)

    debug_dir = Path("debug/")
    debug_dir.mkdir(exist_ok=True)

    # 単眼設定のために左画像のみを読み込む
    left_images = sorted(Path("data/NewTsukubaStereoDataset/illumination/fluorescent/").glob("L_*.png"))

    # キーポイント検出器クラス
    kpdet = KeyPointDetector()

    # 最初の画像を読み込む
    img = load_image(left_images[0])
    kp, _ = kpdet.detect(img)

    img_prev, kp_prev = img, kp
    kp_keyf = kp

    # 姿勢変化の時系列情報（軌跡）を初期化する。
    traj_gt = np.array([groundtruth[i]["position_abs"] for i in range(0, len(groundtruth), key_frame_rate)], dtype=np.float32)
    trajectory = Trajectory(trajectory_gt=traj_gt)

    idx_data = 0
    tracking_count = 0
    for img_path in left_images[frame_reduction_rate:][::frame_reduction_rate]:
        idx_data += frame_reduction_rate

        img_curr = load_image(img_path)

        kp_curr, status, error = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, kp_prev, None, **lk_params)
        tracking_count += 1

        # オプティカルフローによる特徴点追跡を指定された回数行うまでループ
        if tracking_count < num_tracking_before_pose_estimation:
            img_prev = img_curr
            kp_prev = kp_curr  # get rid of points failed to track
            continue

        tracking_count = 0

        # ここから姿勢推定をおこなう
        # 追跡がうまく行っている点のみを使う
        kp_curr_good = kp_curr[status == 1][:, np.newaxis]
        kp_keyf_good = kp_keyf[status == 1][:, np.newaxis]
        # 基本行列を計算する
        E, mask = cv2.findEssentialMat(kp_curr_good, kp_keyf_good, cameraMatrix=K, method=cv2.RANSAC, prob=0.99, threshold=1)
        # 基本行列から回転と並進を取り出す
        points, R, t, _ = cv2.recoverPose(E, kp_curr_good, kp_keyf_good, cameraMatrix=K, mask=mask)

        # 並進（ｔ）のノルムは常に1となり、(a)隣接するフレームとのスケール比、(b)実際のスケールとの対応がわからない
        # 現在はインチキで真値を用いて(a)(b) の両方を解決しているが、本来は(a)は解決可能な問題なので後日取り組む
        # FIXME: 隣接フレームとのスケール比率を三角測量を用いて計算するように変更する
        trans_abs_scale = groundtruth[idx_data]["trans_abs_scale"]
        t_scaled = t * trans_abs_scale

        # 並進と回転を一つの行列（４ｘ４）にする
        RT = np.eye(4)
        RT[:3, :4] = np.concatenate([R, t_scaled], axis=1)

        # 軌跡情報を更新して表示する
        trajectory.append(RT)
        trajectory.show()
        # 特徴点追跡の状況を表示する（別窓）
        visualize_keypoint_tracking(img_curr, kp_curr_good, kp_keyf_good)
        # ターミナルにも表示する
        print("position (inference, groudtruth)", trajectory.trajectory[-1], groundtruth[idx_data]["position_abs"])

        # 次のループのための入れ替え
        kp_curr, _ = kpdet.detect(img_curr)
        img_prev = img_curr
        kp_prev = kp_curr
        kp_keyf = kp_prev


def load_image(img_path: Path):
    return cv2.imread(str(img_path), 0)


def visualize_keypoint_tracking(img_curr, kp_curr_good, kp_keyf_good):
    img_debug = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)
    color = np.random.randint(0, 255, (kp_curr_good.shape[0], 3))  # create some random colors# create some random colors
    mask_draw = np.zeros((img_curr.shape[0], img_curr.shape[1], 3)).astype(np.uint8)
    for i, (curr, keyf) in enumerate(zip(kp_curr_good, kp_keyf_good)):
        a, b = curr.astype(np.int).ravel()
        c, d = keyf.astype(np.int).ravel()
        mask_draw = cv2.line(mask_draw, (a, b), (c, d), color[i].tolist(), 2)
        img_debug = cv2.circle(img_debug, (a, b), 5, color[i].tolist(), -1)
    img_debug = cv2.add(img_debug, mask_draw)
    cv2.imshow("optical flow", img_debug)
    cv2.waitKey(-1)


class KeyPointDetector:

    def __init__(self):
        self.orb = cv2.ORB_create()

    def detect(self, img: np.ndarray):
        kp, des = self.orb.detectAndCompute(img, None)
        return self.kp_in_umat(kp), des

    @classmethod
    def kp_in_umat(cls, kps: Tuple[cv2.KeyPoint]):
        return np.array([kp.pt for kp in list(kps)], np.float32)[:, np.newaxis]


class Trajectory():
    def __init__(self, trajectory_gt=None):
        self.trajectory = np.array([[0., 0., 0.]], dtype=np.float32)
        self.trajectory_gt = trajectory_gt
        self.poses = [np.eye(4)]
        self.x_range = (-1000, 1000)
        self.z_range = (-1000, 1000)
        self.viz_width = self.x_range[1] - self.x_range[0]
        self.viz_height = self.z_range[1] - self.z_range[0]
        self.viz_img_traj = None
        self.viz_xz_center = np.array([self.viz_width / 2, self.viz_height / 2])
        self.viz_pos_last = 0
        self.vis_point_color = np.array([0, 255, 0])
        self.vis_point_color_gt = np.array([0, 0, 255])

        # 真値を可視化したところ、初期姿勢のZは負の方向に向いているようだったのでそれに合わせる。
        self.poses[0][2, 2] = -1

    def append(self, pose):
        pose_last = self.poses[-1]
        pose_curr = pose_last @ pose
        position_curr = pose_curr[:, 3][:-1]
        self.poses.append(pose_curr)
        self.trajectory = np.append(self.trajectory, position_curr[np.newaxis], axis=0)

    def update_vis(self):
        self.viz_img_traj = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)

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

        self.viz_img_traj[trajectory_z[valid], trajectory_x[valid]] = self.vis_point_color

        if self.trajectory_gt is not None:
            # 与えられていれば真値も表示する
            trajectory_gt_xz = self.trajectory_gt[:len(self.trajectory)][:, [0, 2]]
            trajectory_gt_xz += self.viz_xz_center
            trajectory_gt_x = trajectory_gt_xz[:, 0].astype(np.int32)
            trajectory_gt_z = trajectory_gt_xz[:, 1].astype(np.int32)
            self.viz_img_traj[trajectory_gt_z, trajectory_gt_x] = self.vis_point_color_gt

        # １画素の点を３画素に大きくして見やすくする
        self.viz_img_traj = cv2.dilate(self.viz_img_traj, np.ones((3, 3)), iterations=1)

    def show(self):
        self.update_vis()
        cv2.imshow("trajectory", self.viz_img_traj)
        cv2.waitKey(0)


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


if __name__ == "__main__":
    main()
