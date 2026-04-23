from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from settings import Settings


def count_pips_via_distance_transform(pip_mask: MatLike, settings: Settings):
    dist = cv2.distanceTransform(pip_mask, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, settings.dist_thresh_abs, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg.astype("uint8"),
        connectivity=8,
    )

    centers = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 2:
            centers.append((int(centroids[i][0]), int(centroids[i][1])))

    return len(centers), centers


@dataclass
class Output:
    num_dice: int
    num_pips: int


def resize(image: MatLike, settings: Settings) -> tuple[int, int, MatLike, MatLike]:
    width = int(image.shape[1] * settings.scale_percent / 100)
    height = int(image.shape[0] * settings.scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return width, height, image, gray


def dice_body_detection(
    gray: MatLike, settings: Settings
) -> tuple[MatLike, list[MatLike], int]:
    blurred_dice = cv2.GaussianBlur(gray, (7, 7), 0)
    _, dice_thresh = cv2.threshold(
        blurred_dice, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dice_contours, _ = cv2.findContours(
        dice_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours: list[MatLike] = []
    num_dice = 0
    dice_mask = np.zeros_like(gray)

    for cnt in dice_contours:
        area = cv2.contourArea(cnt)
        if area > settings.min_die_area:
            valid_contours.append(cnt)
            cv2.drawContours(dice_mask, [cnt], -1, 255, -1)

            dice_in_this_blob = max(1, round(area / settings.expected_die_area))
            num_dice += dice_in_this_blob

    return dice_mask, valid_contours, num_dice


def create_pip_mask(gray: MatLike, dice_mask: MatLike, settings: Settings) -> MatLike:
    blurred_pips = cv2.GaussianBlur(gray, (5, 5), 0)
    dark_regions = cv2.adaptiveThreshold(
        blurred_pips,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        settings.adaptive_block,
        settings.adaptive_c,
    )

    # Restrict to dice area only
    pip_mask = cv2.bitwise_and(dark_regions, dark_regions, mask=dice_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, open_kernel)

    # DECREASED kernel from 5x5 to 3x3 so we don't accidentally weld pips together!
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_CLOSE, close_kernel)

    return pip_mask


def draw_result(
    image: MatLike,
    pip_centers: list[tuple[int, int]],
    valid_contours: list[MatLike],
    total_pips: int,
    height: int,
    num_dice: int,
    path: Path,
    settings: Settings,
):
    for cx, cy in pip_centers:
        cv2.circle(image, (cx, cy), int(settings.pip_radius_est * 1.3), (0, 0, 255), 2)

    for cnt in valid_contours:
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image,
        f"Total Points: {total_pips}",
        (15, height - 20),
        font,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"Number of Dice: {num_dice}",
        (15, height - 50),
        font,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imwrite(settings.output.joinpath(path.name), image)


def analyze_and_save_dice(path: Path, settings: Settings) -> Output:
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    width, height, image, gray = resize(image, settings)
    dice_mask, valid_contours, num_dice = dice_body_detection(gray, settings)
    pip_mask = create_pip_mask(gray, dice_mask, settings)
    total_pips, pip_centers = count_pips_via_distance_transform(pip_mask, settings)

    draw_result(
        image, pip_centers, valid_contours, total_pips, height, num_dice, path, settings
    )

    return Output(num_dice, total_pips)
