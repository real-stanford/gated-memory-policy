import itertools
from typing import Any, Sequence
import cv2
from cv2.typing import MatLike
import numpy as np
import numpy.typing as npt
import glob
import os
import multiprocessing as mp
from robot_utils.video_utils import save_np_array_as_video

cv2.setNumThreads(1)


def detect_circles(
    contours: Sequence[MatLike],
    area_min: float,
    area_max: float,
    circularity_min: float,
    x_min: float,
    exclude_center: tuple[float, float] | None = None,
    exclude_radius: float | None = None,
) -> list[dict[str, Any]]:
    # Check whether the convex hull is close to a circle
    detected_objects: list[dict[str, Any]] = []
    for contour in contours:
        convex_hull = cv2.convexHull(contour)
        # Calculate the area of the convex hull
        area = cv2.contourArea(convex_hull)
        # Calculate the perimeter of the convex hull
        perimeter = cv2.arcLength(convex_hull, True)
        if perimeter == 0:
            continue
        # Calculate the circularity of the convex hull
        radius = np.sqrt(area / np.pi)
        circularity = 4 * np.pi * area / (perimeter**2)
        # If the circularity is close to 1, then the convex hull is close to a circle

        if area < area_min or area > area_max:
            continue
        # print(
        #     f"circularity: {circularity:.3f}, area: {area:.1f}, perimeter: {perimeter:.3f}"
        # )

        if circularity < circularity_min:
            continue

        # Filter out small contours that are likely noise
        moment = cv2.moments(convex_hull)
        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
            if cx < x_min:
                # Filter out the contours that are too close to the left camera edge, not on the table
                # print(f"Skipping object at {cx}, {cy} because it is too close to the left edge.")
                continue

            detected_objects.append(
                {
                    "contour": convex_hull,
                    "area": area,
                    "center": (cx, cy),
                    "circularity": circularity,
                    "perimeter": perimeter,
                    "radius": radius,
                }
            )

    if (
        exclude_center is not None
        and exclude_radius is not None
        # and len(detected_objects) > 1
    ):
        remaining_objects = []
        for object in detected_objects:
            if (
                np.linalg.norm(np.array(exclude_center) - np.array(object["center"]))
                > exclude_radius
            ):
                remaining_objects.append(object)

        detected_objects = remaining_objects
    # Sort the detected objects by area in descending order
    detected_objects.sort(key=lambda x: x["circularity"], reverse=True)

    return detected_objects


def get_cup_info(
    image: MatLike,
) -> dict[str, Any] | None:
    # --- Parameters to Tune ---
    # Gaussian blur kernel size
    BLUR_KERNEL_SIZE = (7, 7)
    # Canny edge detection thresholds
    CANNY_THRESHOLD_1 = 20
    CANNY_THRESHOLD_2 = 70
    # Minimum contour area to be considered an object (filters out noise)
    # MIN_CONTOUR_AREA = 4000
    CUP_AREA_MIN = 5000
    CUP_AREA_MAX = 8000

    # To filter out the center of the saucer
    SAUCER_AREA_MIN = 13000
    SAUCER_AREA_MAX = 20000
    SAUCER_CENTER_DISTANCE_MIN = 10

    HALF_CUP_PERIMETER_MIN = 100
    HALF_CUP_PERIMETER_MAX = 500

    CIRCULARITY_MIN = 0.95

    X_MIN = 100

    # --- Main Script ---

    # Create a copy to draw on later
    annotated_img = image.copy()

    # 2. Pre-process the image
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

    # 3. Detect Edges
    # Canny edge detector finds sharp intensity changes
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # 4. Find Contours
    # Find the outlines of the white objects in the edged image
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_saucers = detect_circles(
        contours,
        SAUCER_AREA_MIN,
        SAUCER_AREA_MAX,
        CIRCULARITY_MIN,
        X_MIN,
    )
    if len(detected_saucers) > 0:
        saucer_center: tuple[float, float] | None = detected_saucers[0]["center"]
    else:
        saucer_center = None

    # Find the convex hull of the contours
    detected_cups: list[dict[str, Any]] = detect_circles(
        contours,
        CUP_AREA_MIN,
        CUP_AREA_MAX,
        CIRCULARITY_MIN,
        X_MIN,
        saucer_center,
        SAUCER_CENTER_DISTANCE_MIN,
    )

    # Ensure we have at least two objects to identify
    if len(detected_cups) >= 1 and (
        saucer_center is None
        or np.linalg.norm(
            np.array(saucer_center) - np.array(detected_cups[0]["center"])
        )
        > SAUCER_CENTER_DISTANCE_MIN
    ):
        return detected_cups[0]

    else:
        # If not found, will try using random combination of multiple contours
        potential_edge_contours: list[MatLike] = [
            contour
            for contour in contours
            if cv2.arcLength(contour, False) > HALF_CUP_PERIMETER_MIN
            and cv2.arcLength(contour, False) < HALF_CUP_PERIMETER_MAX
        ]

        combinations = itertools.combinations(potential_edge_contours, 2)

        combined_contours: Sequence[MatLike] = [
            np.vstack(combination) for combination in combinations
        ]

        detected_combined_cups: list[dict[str, Any]] = detect_circles(
            combined_contours,
            CUP_AREA_MIN,
            CUP_AREA_MAX,
            CIRCULARITY_MIN,
            X_MIN,
            saucer_center,
            SAUCER_CENTER_DISTANCE_MIN,
        )

        if len(detected_combined_cups) >= 1:
            return detected_combined_cups[0]
        elif len(detected_cups) >= 1:
            return detected_cups[0]
        else:
            print(
                # f"Could not identify two distinct objects. Found {len(detected_objects)}."
                f"Could not identify cup."
            )
            return None


def annotate_video(
    source_video_path: str,
    output_video_path: str,
    keep_first_cup_center: bool,
    success_radius_ratio: int = 2,
    use_last_pos_if_no_cup: bool = True,
) -> None:
    cap = cv2.VideoCapture(
        source_video_path,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_HW_ACCELERATION,
            cv2.VIDEO_ACCELERATION_ANY,  # Let OpenCV/FFmpeg choose the best available HW acceleration
        ],
    )
    annotated_video_frames: list[MatLike] = []
    first_cup_center: tuple[int, int] | None = None
    first_cup_radius: float | None = None

    last_cup_info: dict[str, Any] | None = None
    last_cup_color: tuple[int, int, int] = (0, 0, 0)

    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # print(f"Processing frame {frame_count}")
        cup_info = get_cup_info(frame)

        if cup_info is not None:
            last_cup_info = cup_info
            if first_cup_center is None:
                first_cup_center = cup_info["center"]
                first_cup_radius = cup_info["radius"]
            color = (0, 255, 0)  # Green
            if (
                keep_first_cup_center
                and first_cup_center is not None
                and first_cup_radius is not None
                and np.linalg.norm(
                    np.array(first_cup_center) - np.array(cup_info["center"])
                )
                > success_radius_ratio * first_cup_radius
            ):
                # print(first_cup_center, cup_info["center"])
                # print(cup_info["radius"], np.linalg.norm(np.array(first_cup_center) - np.array(cup_info["center"])))
                color = (0, 0, 255)  # Red

            last_cup_color = color

            # cv2.drawContours(frame, [cup_info["contour"]], -1, color, 3)
            cv2.circle(frame, cup_info["center"], 7, color, -1)
        elif use_last_pos_if_no_cup and last_cup_info is not None:
            cv2.circle(frame, last_cup_info["center"], 7, last_cup_color, -1)

        if (
            keep_first_cup_center
            and first_cup_center is not None
            and first_cup_radius is not None
        ):
            # Blue
            cv2.circle(frame, first_cup_center, 7, (255, 0, 0), -1)
            if success_radius_ratio > 0:
                border_radius = int(success_radius_ratio * first_cup_radius)
                cv2.circle(frame, first_cup_center, border_radius, (255, 0, 0), 2)

        annotated_video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Annotated Image", annotated_img)
        # cv2.waitKey(1)
    cap.release()
    save_np_array_as_video(annotated_video_frames, output_video_path)


def annotate_entire_run(
    run_dir: str, keep_first_cup_center: bool = True, process_num: int = 8
) -> None:
    episode_dirs = glob.glob(os.path.join(run_dir, "episode_*"))
    source_video_paths: list[str] = []
    output_video_paths: list[str] = []

    # episode_dirs = [f"{run_dir}/episode_00007"]

    for episode_dir in episode_dirs:
        source_video_path = f"{episode_dir}/third_person_camera_0.zarr/main.mp4"
        print(source_video_path)
        output_video_path = source_video_path.replace(".mp4", "_annotated.mp4")
        source_video_paths.append(source_video_path)
        output_video_paths.append(output_video_path)

    keep_first_cup_center_list = [keep_first_cup_center] * len(source_video_paths)
    with mp.Pool(process_num) as pool:
        pool.starmap(
            annotate_video,
            zip(source_video_paths, output_video_paths, keep_first_cup_center_list),
        )
