import math


def int_x1y1x2y2(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.ceil(x2)
    y2 = math.ceil(y2)
    return [x1, y1, x2, y2]


def xywh_to_x1y1x2y2(xywh):
    x, y, w, h = xywh
    x1y1x2y2 = [x, y, x + w, y + h]
    return int_x1y1x2y2(x1y1x2y2)


def normalize_x1y1x2y2(x1y1x2y2, width, height, round_precision=4):
    x1, y1, x2, y2 = x1y1x2y2
    norm_x1 = x1 / width
    norm_y1 = y1 / height
    norm_x2 = x2 / width
    norm_y2 = y2 / height
    norm_x1, norm_x2, norm_y1, norm_y2 = list(
        map(lambda x: round(x, round_precision), [norm_x1, norm_x2, norm_y1, norm_y2])
    )
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def denormalize_x1y1x2y2(normalized_x1y1x2y2, width, height):
    norm_x1, norm_y1, norm_x2, norm_y2 = normalized_x1y1x2y2
    x1 = norm_x1 * width
    y1 = norm_y1 * height
    x2 = norm_x2 * width
    y2 = norm_y2 * height
    x1y1x2y2 = [x1, y1, x2, y2]
    return int_x1y1x2y2(x1y1x2y2)


def x1y1x2y2_with_spacing(x1y1x2y2, spacing=2):
    x1, y1, x2, y2 = x1y1x2y2
    x1 -= spacing
    y1 -= spacing
    x2 += spacing
    y2 += spacing
    x1y1x2y2 = [x1, y1, x2, y2]
    return int_x1y1x2y2(x1y1x2y2)
