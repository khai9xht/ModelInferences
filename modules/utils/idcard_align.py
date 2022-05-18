from typing import List, Union
import numpy as np
import math
import cv2

def crop_bbox(img: np.ndarray, bbox: List[float], 
        pad_h_ratio=0.0, pad_w_ratio=0.0, copy=False) -> np.ndarray:
    """
    bbox structure: [xmin, ymin, xmax, ymax]
    """
    img_h, img_w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    pad_h = int((ymax - ymin) * pad_h_ratio)
    pad_w = int((xmax - xmin) * pad_w_ratio)
    xmin = xmin - pad_w
    xmax = xmax + pad_w 
    ymin = ymin - pad_h
    ymax = ymax + pad_h 
    xmin = int(xmin) if xmin > 0 else 0
    ymin = int(ymin) if ymin > 0 else 0
    xmax = int(xmax) if xmax < img_w else img_w
    ymax = int(ymax) if ymax < img_h else img_h
    if copy:
        cropped = img[ymin:ymax, xmin:xmax].copy()
    else:
        cropped = img[ymin:ymax, xmin:xmax]
    return cropped


def detect_skew_angle_from_bboxes(bboxes: List[np.ndarray], min_size: float=None) -> float:
    """Returns mean skew angle of bbboxes"""
    if len(bboxes) == 0:
        return 0.0

    x_axis = np.array([1, 0])
    angle_list = []
    for box in bboxes:
        if min_size is not None:
            h, w = bbox_size(box)
            size = max(h, w)
            if size < min_size:
                continue

        direction = get_direction_vector(box)
        angle = get_radian_angle(x_axis, direction)
        angle_list.append(angle)

    angle_list = sorted(angle_list)
    n = len(angle_list)
    quarter_len = len(angle_list) // 4
    start = quarter_len
    end = n - quarter_len
    angle_list = angle_list[start:end]

    mean_angle = np.mean(angle_list)

    return math.degrees(mean_angle)


def rotate_bound(image: np.ndarray, angle: float) -> Union[np.ndarray, np.ndarray]:
    """Rotates image with the angle (in degrees), auto resize"""
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M


def transform_bboxes(bboxes, M):
    n = len(bboxes)
    if n == 0:
        return bboxes
    stacked = np.vstack(bboxes)
    stacked = np.c_[stacked, np.ones(n * 4)]
    transformed = np.dot(stacked, M.T)

    if transformed.shape[-1] == 3:  # perspective transformation
        transformed = transformed[:, :2] / transformed[:, 2][:, None]

    bboxes = np.vsplit(transformed, n)
    bboxes = standardize_bboxes(bboxes)
    return bboxes


def transform_bbox(bbox, M):
    if bbox is None:
        return None

    bbox = transform_bboxes([bbox], M)[0]
    return bbox


def standardize_bboxes(bboxes):
    final_bboxes = []
    for box in bboxes:
        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        final_bboxes.append(box)
    return final_bboxes


def get_direction_vector(bbox):
    """Returns the orientation vector of the bbox"""
    tl, tr, br, bl = bbox
    v1 = tr - tl
    v2 = bl - tl
    if length(v1) > length(v2):
        return v1
    else:
        return v2


def length(vector):
    """Returns the length of the vector"""
    return np.sqrt(vector.dot(vector))


def get_radian_angle(v1, v2):
    """Returns the angle in radians between vectors `v1` and `v2`"""
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    theta_0 = np.arctan2(sinang, cosang)
    m = np.array([v1, v2])
    det = np.linalg.det(m)
    theta = theta_0 if det > 0 else -theta_0
    return theta

def check_orientation(img_width, emblem_bbox):
    """Checks if the id card has the correct orientation"""
    if emblem_bbox is None:
        return True

    if emblem_bbox[0][0] < img_width // 2:
        return True

    return False


def dewarp(img, bboxes):
    """Dewarps the id card such that, all text lines are deskewed"""
    if len(bboxes) < 9:
        return img, np.eye(3)
    h_list = []
    w_list = []
    for bbox in bboxes:
        h, w = bbox_size(bbox)
        h_list.append(h)
        w_list.append(w)
    mean_w = np.mean(w_list)
    min_width = img.shape[1] * 0.25
    mean_w = min_width if min_width > mean_w else mean_w
    median_h = np.median(h_list)
    filtered_bboxes = []
    for bbox in bboxes:
        # filter expired_date line
        x = bbox[0][0]
        if x < img.shape[0] / 10:
            continue

        h, w = bbox_size(bbox)
        if w > mean_w and h < median_h * 1.4:
            filtered_bboxes.append(bbox)

    bboxes = sorted(filtered_bboxes, key=lambda bbox: bbox[0][1])
    if len(bboxes) < 2:
        return img, np.eye(3)

    top_bbox = bboxes[0]
    bot_bbox = bboxes[-1]
    h, w = img.shape[:2]
    M = get_warp_matrix_from_2_bboxes(h, w, top_bbox, bot_bbox)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped, M

def bbox_size(bbox):
    w = bbox[0] - bbox[1]
    h = bbox[0] - bbox[3]
    w = np.linalg.norm(w)
    h = np.linalg.norm(h)
    return int(h), int(w)

def get_warp_matrix_from_2_bboxes(h, w, top_bbox, bot_bbox):
    tl, tr = top_bbox[0], top_bbox[1]
    bl, br = bot_bbox[3], bot_bbox[2]

    top_xx = np.array([tl[0], tr[0]])
    top_yy = np.array([tl[1], tr[1]])
    bot_xx = np.array([bl[0], br[0]])
    bot_yy = np.array([bl[1], br[1]])

    top_z = np.polyfit(top_xx, top_yy, 1)
    bot_z = np.polyfit(bot_xx, bot_yy, 1)

    top_line = np.poly1d(top_z)
    bot_line = np.poly1d(bot_z)

    src_tl = [0, top_line(0)]
    src_tr = [w, top_line(w)]
    src_br = [w, bot_line(w)]
    src_bl = [0, bot_line[0]]
    dst_tl = src_tl
    dst_tr = [src_tr[0], src_tl[1]]
    dst_br = [src_br[0], src_bl[1]]
    dst_bl = src_bl

    src = [src_tl, src_tr, src_br, src_bl]
    dst = [dst_tl, dst_tr, dst_br, dst_bl]

    src = np.array(src, dtype="float32")
    dst = np.array(dst, dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)

    return M