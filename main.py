import numpy as np
import cv2
from pathlib import Path
from typing import Tuple

from map import Map
from groundtruth import GroundTruth
from utils import visualize_keypoint_tracking



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

    np.set_printoptions(precision=2, suppress=True)

    # 処理するフレーム数を指定する。値は１から始まる整数。
    # 大きくなるほどに計算が高速になるが、オプティカルフローによる特徴点追跡が失敗する可能性が高まるトレードオフが発生する点に注意。
    frame_reduction_rate = 1

    # カメラの姿勢検出を行う前に何回トラッキングを行うかを指定する
    # トラッキングの回数が少ない場合は姿勢の変化量が少ないために推定された姿勢が不安定なる
    # 一方で多いと追跡点の多くが画像からはずれてしまうことになる。
    num_tracking_before_pose_estimation = 10
    key_frame_rate = frame_reduction_rate * num_tracking_before_pose_estimation

    # 設定したフレーム数までは正解値をつかってスケールさせることでmapを初期化する。
    frames_before_initialized = 50

    # 再投影誤差がこの閾値に収まっているものだけを信頼できる特徴点として扱う。
    reproj_error_threshold = 4.0  # ピクセル

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
    map = Map(trajectory_gt=traj_gt)

    idx_data = 0
    tracking_count = 0

    initialized = False

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

        # ここから姿勢推定を実行する
        kp_curr_good = kp_curr[status == 1][:, np.newaxis]
        kp_keyf_good = kp_keyf[status == 1][:, np.newaxis]
        if "point_4d" in locals() and point_4d is not None:
            point_4d = point_4d[(status[:len(point_4d)] == 1).squeeze(1)]

        if not initialized:  # 初期化中
            # 真値とスケールを合わせるために初期化時には正解をつかって推定された並進をスケールさせる
            trans_abs_scale = groundtruth[idx_data]["trans_abs_scale"]
            RT_curr, point_4d, kp_keyf_good, kp_curr_good = two_view_sfm_for_init(kp_keyf_good, kp_curr_good, K, trans_abs_scale)

            # 姿勢推定ができなかった場合は次のフレームで再実行する
            if RT_curr is None:
                continue

            if idx_data > frames_before_initialized:
                initialized = True

        else:  # 初期化後
            RT_curr, point_4d, kp_keyf_good, kp_curr_good = solvePnP_and_get_new_3dpoints(RT_keyf, kp_keyf_good, kp_curr_good, point_4d, K, reproj_error_threshold)

        map.append(np.linalg.inv(RT_curr), point_4d)
        map.show()

        # 特徴点追跡の状況を表示する（別窓）
        visualize_keypoint_tracking(img_curr, kp_curr_good, kp_keyf_good, idx_data)
        # ターミナルにも表示する
        print("position (inference, groudtruth)", map.trajectory[-1], groundtruth[idx_data]["position_abs"])

        # 次のループのための入れ替え
        kp_curr, point_4d = update_keypoints(kpdet, img_curr, kp_curr_good, point_4d)
        img_prev = img_curr
        kp_prev = kp_curr
        kp_keyf = kp_prev
        RT_keyf = RT_curr


def two_view_sfm_for_init(kp_keyf_good: np.ndarray, kp_curr_good: np.ndarray, K: np.ndarray, trans_abs_scale: float):
    # 基本行列を計算する
    E, mask = cv2.findEssentialMat(kp_keyf_good, kp_curr_good, cameraMatrix=K, method=cv2.RANSAC, prob=0.99, threshold=1)

    # 基本行列から回転と並進を取り出す
    points, R, tvecs, mask = cv2.recoverPose(E, kp_keyf_good, kp_curr_good, cameraMatrix=K, mask=mask)
    tvecs = tvecs * trans_abs_scale

    # 外れ値を取り除く
    kp_curr_good = kp_curr_good[mask.ravel()==1]
    kp_keyf_good = kp_keyf_good[mask.ravel()==1]

    # 十分なキーポイントが得られていない場合は処理を中断する。
    if len(kp_curr_good) < 20:
        print("0:", len(kp_curr_good), len(kp_keyf_good))
        return None, None, None, None

    RT_relative = np.eye(4, 4)
    RT_relative[:3, :4] = np.concatenate([R, tvecs], axis=1)

    # 射影行列を作成する。
    if not "RT_keyf" in locals():
        RT_keyf = np.eye(4, 4)
    RT_curr = RT_relative @ RT_keyf
    Proj_keyf = np.dot(K, RT_keyf[:3])
    Proj_curr = np.dot(K, RT_curr[:3])

    # triangulate points
    point_4d = cv2.triangulatePoints(Proj_keyf, Proj_curr, kp_keyf_good, kp_curr_good)
    point_4d = point_4d / np.tile(point_4d[-1, :], (4, 1))
    point_4d = point_4d[:, :].T

    print("reproj error: ", np.mean(compute_reproj_error(Proj_curr, point_4d, kp_curr_good.squeeze(1))))

    return RT_curr, point_4d, kp_keyf_good, kp_curr_good


def solvePnP_and_get_new_3dpoints(RT_keyf: np.ndarray, kp_keyf_good: np.ndarray, kp_curr_good: np.ndarray, point_4d: np.ndarray, K: np.ndarray, reproj_error_threshold: float):
    kp_curr_good_3d_known = kp_curr_good[:len(point_4d)]
    kp_curr_good_3d_unknown = kp_curr_good[len(point_4d):]
    kp_keyf_good_3d_known = kp_keyf_good[:len(point_4d)]
    kp_keyf_good_3d_unknown = kp_keyf_good[len(point_4d):]

    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(point_4d[:, :3], kp_curr_good_3d_known.squeeze(1), cameraMatrix=K, distCoeffs=None)

    kp_curr_good_3d_known = kp_curr_good_3d_known[inliers].squeeze(1)
    kp_keyf_good_3d_known = kp_keyf_good_3d_known[inliers].squeeze(1)
    point_4d = point_4d[inliers].squeeze(1)

    kp_curr_good = np.concatenate([kp_curr_good_3d_known, kp_curr_good_3d_unknown], axis=0)
    kp_keyf_good = np.concatenate([kp_keyf_good_3d_known, kp_keyf_good_3d_unknown], axis=0)

    R, Jacob = cv2.Rodrigues(rvecs)

    # 軌跡情報を更新して表示する
    # 並進と回転を一つの行列（４ｘ４）にする
    # TODO: 上で作っている行列と重複があるので統一する
    RT_curr = np.eye(4)
    RT_curr[:3, :4] = np.concatenate([R, tvecs], axis=1)

    RT_relative = RT_curr @ np.linalg.inv(RT_keyf)

    # 射影行列を作成する。
    Proj_keyf = np.dot(K,  RT_keyf[:3])
    Proj_curr = np.dot(K,  RT_curr[:3])

    # 現在のフレームにpoint_4dを投影して誤差を確認する。
    print(rvecs, tvecs, "\n", "reproj error: ", np.mean(compute_reproj_error(Proj_curr, point_4d, kp_curr_good_3d_known.squeeze(1))))

    # triangulate points
    point_4d_new = cv2.triangulatePoints(Proj_keyf, Proj_curr, kp_keyf_good, kp_curr_good)
    point_4d_new = point_4d_new / np.tile(point_4d_new[-1, :], (4, 1))
    point_4d_new = point_4d_new[:, :].T

    reproj_error = compute_reproj_error(Proj_curr, point_4d_new, kp_curr_good.squeeze(1))
    print("reproj error: ", np.mean((reproj_error)))

    # TODO: point4d_newのなかで再投影誤差がすくないものだけを採用する。
    point_4d = point_4d_new[reproj_error <= reproj_error_threshold]
    kp_curr_good = kp_curr_good[reproj_error <= reproj_error_threshold]
    kp_keyf_good = kp_keyf_good[reproj_error <= reproj_error_threshold]

    return RT_curr, point_4d, kp_keyf_good, kp_curr_good


def compute_reproj_error(Proj, point_4d, point_image):
    point_image_reproj = Proj @ point_4d.T  # (4, N)
    point_image_reproj = point_image_reproj.T
    point_image_reproj = point_image_reproj / point_image_reproj[:, [2]]
    error = np.linalg.norm(point_image - point_image_reproj[:, :2], axis=1)
    return error


def update_keypoints(kpdet, img_curr, kp_curr, points_4d, min_keypoins=400):
    """キーポイントを検出して既存のキーポイントと重複していないものを処理対象に加える。画面外に移動したキーポイントは除去する。

    Args:
        kpdet (_type_): キーポイント検出器
        img_curr (_type_): 現在の画像
        kp_curr (_type_): 既存のキーポイント
    """
    assert(len(kp_curr) == len(points_4d)), f"{len(kp_curr)}, {len(points_4d)}"

    x_in_frame = np.logical_and(
        kp_curr[:, :, 0] >= 0,
        kp_curr[:, :, 0] < img_curr.shape[1]
    )
    y_in_frame = np.logical_and(
        kp_curr[:, :, 1] >= 0,
        kp_curr[:, :, 1] < img_curr.shape[0]
    )
    xy_both_in_frame = np.logical_and(x_in_frame, y_in_frame)
    xy_both_in_frame = xy_both_in_frame.squeeze(axis=1)
    kp_curr = kp_curr[xy_both_in_frame]
    points_4d = points_4d[xy_both_in_frame]

    if len(kp_curr) > min_keypoins:
        return kp_curr, points_4d

    kp_new, _ = kpdet.detect(img_curr)

    # init maps
    map_shape = img_curr.shape[:2]
    map_inds_new = np.ones(map_shape, dtype=np.int32) * -1
    map_curr_bitmap = np.zeros(map_shape)
    map_new_bitmap = np.zeros(map_shape)

    kp_new_inds = np.array(list(range(len(kp_new))), dtype=np.int32)

    kp_curr_idx_x = kp_curr[:, :, 0].astype(np.int32).squeeze(-1)
    kp_curr_idx_y = kp_curr[:, :, 1].astype(np.int32).squeeze(-1)
    kp_new_idx_x = kp_new[:, :, 0].astype(np.int32).squeeze(-1)
    kp_new_idx_y = kp_new[:, :, 1].astype(np.int32).squeeze(-1)

    map_inds_new[kp_new_idx_y, kp_new_idx_x] = kp_new_inds

    map_curr_bitmap[kp_curr_idx_y, kp_curr_idx_x] = 2
    map_new_bitmap[kp_new_idx_y, kp_new_idx_x] = 1

    map_curr_bitmap = cv2.dilate(map_curr_bitmap, kernel=np.ones((9, 9)))
    map_curr_new_bitmap = np.stack([map_curr_bitmap, map_new_bitmap], axis=2)
    map_curr_new_bitmap = np.max(map_curr_new_bitmap, axis=2)

    kp_new_append_inds = map_inds_new[map_curr_new_bitmap == 1]
    kp_new_append = kp_new[kp_new_append_inds]
    kp_all = np.concatenate([kp_curr, kp_new_append], axis=0)

    return kp_all, points_4d


def load_image(img_path: Path):
    return cv2.imread(str(img_path), 0)


class KeyPointDetector:

    def __init__(self):
        self.orb = cv2.ORB_create()

    def detect(self, img: np.ndarray):
        kp, des = self.orb.detectAndCompute(img, None)
        return self.kp_in_umat(kp), des

    @classmethod
    def kp_in_umat(cls, kps: Tuple[cv2.KeyPoint]):
        return np.array([kp.pt for kp in list(kps)], np.float32)[:, np.newaxis]


if __name__ == "__main__":
    main()
