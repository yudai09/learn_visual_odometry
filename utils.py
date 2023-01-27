import cv2
import numpy as np


def visualize_keypoint_tracking(img_curr, kp_curr_good, kp_keyf_good, idx_data):
    img_debug = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)
    color = np.random.randint(0, 255, (kp_curr_good.shape[0], 3))  # create some random colors# create some random colors
    mask_draw = np.zeros((img_curr.shape[0], img_curr.shape[1], 3)).astype(np.uint8)
    for i, (curr, keyf) in enumerate(zip(kp_curr_good, kp_keyf_good)):
        a, b = curr.astype(np.int32).ravel()
        c, d = keyf.astype(np.int32).ravel()
        mask_draw = cv2.line(mask_draw, (a, b), (c, d), color[i].tolist(), 2)
        img_debug = cv2.circle(img_debug, (a, b), 5, color[i].tolist(), -1)
    img_debug = cv2.add(img_debug, mask_draw)
    cv2.putText(img_debug,
            text=f"frame no. {idx_data}",
            org=(30, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4)
    cv2.imshow("optical flow", img_debug)
    cv2.waitKey(-1)
