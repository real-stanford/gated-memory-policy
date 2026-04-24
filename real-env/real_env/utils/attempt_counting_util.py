"""Attempt detection for drag cube task via blue robot arm centroid tracking in ROI."""

import time
from typing import Optional, Tuple, List
import cv2
import numpy as np

import cv2
import numpy as np


def hex_to_bgr(hex_str: str) -> np.ndarray:
    """Convert hex color string to BGR array."""
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR array to HSV."""
    hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    return hsv.reshape(3)


def clamp_hsv(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp HSV values to valid OpenCV ranges."""
    lo, hi = lo.astype(int), hi.astype(int)
    lo[0], hi[0] = np.clip(lo[0], 0, 179), np.clip(hi[0], 0, 179)
    lo[1:], hi[1:] = np.clip(lo[1:], 0, 255), np.clip(hi[1:], 0, 255)
    return lo.astype(np.uint8), hi.astype(np.uint8)


def hsv_range_from_hex_pair(
    hex_lo: str, hex_hi: str, pad_h: int, pad_s: int, pad_v: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute HSV range from two hex colors with padding."""
    hsv1 = bgr_to_hsv(hex_to_bgr(hex_lo)).astype(int)
    hsv2 = bgr_to_hsv(hex_to_bgr(hex_hi)).astype(int)
    lo = np.minimum(hsv1, hsv2) - np.array([pad_h, pad_s, pad_v])
    hi = np.maximum(hsv1, hsv2) + np.array([pad_h, pad_s, pad_v])
    return clamp_hsv(lo, hi)


def hsv_range_from_hex_samples(
    hex_samples: List[str], pad_h: int, pad_s: int, pad_v: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute HSV range from multiple hex color samples with padding."""
    hsvs = np.stack(
        [bgr_to_hsv(hex_to_bgr(h)).astype(int) for h in hex_samples], axis=0
    )
    lo = hsvs.min(axis=0) - np.array([pad_h, pad_s, pad_v])
    hi = hsvs.max(axis=0) + np.array([pad_h, pad_s, pad_v])
    return clamp_hsv(lo, hi)


def largest_contour(mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    """Find largest contour in mask meeting minimum area threshold."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


# ur5 joint color
BLUE_ARM_HEX_LO = "#52a0d8"
BLUE_ARM_HEX_HI = "#63baf6"
# BLUE_ARM_HEX_LO = "#337FC0"
# BLUE_ARM_HEX_HI = "#2586D4"
PAD_H, PAD_S, PAD_V = 5, 30, 30
BLUE_ARM_MIN_AREA = 500

# Drag Cube color
CUBE_HEX_LO = "#337FC0"
CUBE_HEX_HI = "#2586D4"
CUBE_MIN_AREA = 300

DEFAULT_ROI_CENTER = (330, 440)  # Drag Cube Setting
DEFAULT_ROI_HALF = 300

LOWER_BLUE, UPPER_BLUE = hsv_range_from_hex_pair(
    BLUE_ARM_HEX_LO, BLUE_ARM_HEX_HI, PAD_H, PAD_S, PAD_V
)

LOWER_CUBE, UPPER_CUBE = hsv_range_from_hex_pair(
    CUBE_HEX_LO, CUBE_HEX_HI, PAD_H, PAD_S, PAD_V
)


def detect_blue_centroid_in_roi(
    frame: np.ndarray,
    roi_center: Tuple[int, int],
    roi_half: int,
) -> bool:
    """
    Detect if blue robot arm centroid is within ROI.

    Args:
        frame: RGB image (H, W, 3) from camera observations
        roi_center: (x, y) center of ROI in pixels
        roi_half: half-size of square ROI in pixels

    Returns:
        True if blue arm centroid detected in ROI, False otherwise
    """
    # Convert RGB to BGR (camera frames are RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contour = largest_contour(mask, BLUE_ARM_MIN_AREA)
    if contour is None:
        return False

    # Get centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Check if centroid in ROI
    x0 = roi_center[0] - roi_half
    x1 = roi_center[0] + roi_half
    y0 = roi_center[1] - roi_half
    y1 = roi_center[1] + roi_half

    return x0 <= cx < x1 and y0 <= cy < y1


def create_attempt_detector(
    roi_center: Tuple[int, int] = DEFAULT_ROI_CENTER,
    roi_half: int = DEFAULT_ROI_HALF,
    block_duration_s: float = 8.0,
    use_video_time: bool = False,
):
    """
    Create attempt detector with blocking window to prevent duplicate counts.

    Args:
        roi_center: (x, y) center of ROI for tip detection
        roi_half: half-size of square ROI in pixels
        block_duration_s: seconds to block after detection to avoid duplicates
        use_video_time: if True, use externally provided time; if False, use wall-clock time

    Returns:
        Callable detector function: frame, current_time=None -> bool
        The detector also has a `last_detection_info` attribute with (in_roi, centroid, contour)

    Example:
        >>> detector = create_attempt_detector(block_duration_s=8.0)
        >>> if detector(frame):
        ...     attempt_count += 1
        >>> # Access last detection details
        >>> in_roi, centroid, contour = detector.last_detection_info

        >>> # With video time
        >>> detector = create_attempt_detector(block_duration_s=8.0, use_video_time=True)
        >>> if detector(frame, current_time=video_timestamp):
        ...     attempt_count += 1
    """
    last_detection_time: Optional[float] = None

    def detect(frame: np.ndarray, current_time: Optional[float] = None) -> bool:
        """Returns True if new attempt detected (blue centroid in ROI and not in blocking window)."""
        nonlocal last_detection_time

        if use_video_time:
            if current_time is None:
                raise ValueError(
                    "current_time must be provided when use_video_time=True"
                )
            time_now = current_time
        else:
            time_now = time.monotonic()

        # Detect centroid and contour (cache results for visualization)
        (
            arm_in_roi,
            arm_centroid,
            arm_contour,
            cube_centroid,
            cube_contour,
        ) = _detect_centroid_and_contour(frame, roi_center, roi_half)
        detect.last_detection_info = (
            arm_in_roi,
            arm_centroid,
            arm_contour,
            cube_centroid,
            cube_contour,
        )

        # Check blocking window
        if last_detection_time is not None:
            if (time_now - last_detection_time) < block_duration_s:
                return False

        # Detect blue arm centroid in ROI
        if arm_in_roi:
            last_detection_time = time_now
            return True

        return False

    # Initialize attribute
    detect.last_detection_info = (False, None, None, None, None)
    return detect


def _draw_text_with_outline(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
):
    """Draw text with black outline for visibility."""
    outline_thickness = thickness + 2
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        outline_thickness,
    )
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def _detect_centroid_and_contour(
    frame: np.ndarray,
    roi_center: Tuple[int, int],
    roi_half: int,
) -> Tuple[
    bool,
    Optional[Tuple[int, int]],
    Optional[np.ndarray],
    Optional[Tuple[int, int]],
    Optional[np.ndarray],
]:
    """
    Detect blue arm and cube centroids and contours, check if arm is in ROI.

    Returns:
        (arm_in_roi, arm_centroid, arm_contour, cube_centroid, cube_contour) tuple
    """
    # Convert RGB to BGR if needed (camera frames are RGB)
    if frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Detect blue arm
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    arm_contour = largest_contour(mask_blue, BLUE_ARM_MIN_AREA)
    arm_centroid = None
    arm_in_roi = False

    if arm_contour is not None:
        M = cv2.moments(arm_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            arm_centroid = (cx, cy)

            # Check if arm centroid in ROI
            x0 = roi_center[0] - roi_half
            x1 = roi_center[0] + roi_half
            y0 = roi_center[1] - roi_half
            y1 = roi_center[1] + roi_half
            arm_in_roi = x0 <= cx < x1 and y0 <= cy < y1

    # Detect cube
    mask_cube = cv2.inRange(hsv, LOWER_CUBE, UPPER_CUBE)
    mask_cube = cv2.morphologyEx(mask_cube, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_cube = cv2.morphologyEx(mask_cube, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cube_contour = largest_contour(mask_cube, CUBE_MIN_AREA)
    cube_centroid = None

    if cube_contour is not None:
        M = cv2.moments(cube_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cube_centroid = (cx, cy)

    return arm_in_roi, arm_centroid, arm_contour, cube_centroid, cube_contour


def show_realtime_attempt_detection(
    frame_rgb: np.ndarray,
    attempt_count: int,
    detection_occurred: bool,
    roi_center: Tuple[int, int],
    roi_half: int,
    detection_info: Optional[
        Tuple[
            bool,
            Optional[Tuple[int, int]],
            Optional[np.ndarray],
            Optional[Tuple[int, int]],
            Optional[np.ndarray],
        ]
    ] = None,
    window_name: str = "Attempt Detection",
):
    """
    Show real-time visualization of attempt detection with ROI, arm and cube centroid tracking, and count.

    Args:
        frame_rgb: RGB image from camera
        attempt_count: Current attempt counter
        detection_occurred: Whether an attempt was just detected this frame
        roi_center: (x, y) center of ROI
        roi_half: half-size of square ROI in pixels
        detection_info: Optional cached (arm_in_roi, arm_centroid, arm_contour, cube_centroid, cube_contour) tuple
        window_name: Name for the OpenCV window
    """
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # Use cached detection info if provided, otherwise detect
    if detection_info is not None:
        (
            arm_in_roi,
            arm_centroid,
            arm_contour,
            cube_centroid,
            cube_contour,
        ) = detection_info
    else:
        (
            arm_in_roi,
            arm_centroid,
            arm_contour,
            cube_centroid,
            cube_contour,
        ) = _detect_centroid_and_contour(frame, roi_center, roi_half)

    # Create visualization
    vis = frame.copy()
    roi_x0 = roi_center[0] - roi_half
    roi_y0 = roi_center[1] - roi_half
    roi_x1 = roi_center[0] + roi_half
    roi_y1 = roi_center[1] + roi_half

    # Draw ROI box (yellow)
    cv2.rectangle(vis, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 255), 3)

    # Draw arm contour (green)
    if arm_contour is not None:
        cv2.drawContours(vis, [arm_contour], -1, (0, 255, 0), 2)

    # Draw cube contour (cyan)
    if cube_contour is not None:
        cv2.drawContours(vis, [cube_contour], -1, (255, 255, 0), 2)

    # Draw ROI status
    status = "ARM IN ROI" if arm_in_roi else "OUT"
    color = (0, 255, 0) if arm_in_roi else (0, 0, 255)
    height, width = vis.shape[:2]
    # Flash red border on detection
    if detection_occurred:
        cv2.rectangle(vis, (0, 0), (width - 1, height - 1), (0, 0, 255), 10)

    # Resize vis to 1/4 size
    vis = cv2.resize(vis, (width // 4, height // 4))

    # Draw centroids after resizing for better visibility
    if arm_centroid is not None:
        scaled_arm = (arm_centroid[0] // 4, arm_centroid[1] // 4)
        cv2.circle(vis, scaled_arm, 2, (0, 0, 255), -1)
        cv2.putText(
            vis,
            "ARM",
            (scaled_arm[0] + 4, scaled_arm[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )
    if cube_centroid is not None:
        scaled_cube = (cube_centroid[0] // 4, cube_centroid[1] // 4)
        cv2.circle(vis, scaled_cube, 2, (255, 0, 255), -1)
        cv2.putText(
            vis,
            "CUBE",
            (scaled_cube[0] + 4, scaled_cube[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 255),
            1,
        )

    _draw_text_with_outline(vis, status, (20, 40), 1.2, color)

    # Draw attempt count
    _draw_text_with_outline(
        vis, f"Attempts: {attempt_count}", (20, 80), 1.0, (0, 255, 0)
    )

    cv2.imshow(window_name, vis)
    cv2.waitKey(1)
