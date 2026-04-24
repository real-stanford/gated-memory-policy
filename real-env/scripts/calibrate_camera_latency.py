"""Camera latency measurement using QR code timestamps."""

from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process
from functools import partial
from collections import deque
from contextlib import contextmanager
from typing import Optional, Union
import cv2
import pypeertalk

from real_env.utils.qr_code_util import dynamic_qr_timecode, read_qr_code
from real_env.peripherals.iphone import iPhone, IPHONE_INPUT_STREAMS_CONFIG


class LatencyWindow:
    """Tracks latency measurements in a sliding window."""
    
    def __init__(self, size: int):
        self.size = size
        self.values = deque(maxlen=size)
    
    def add(self, latency: float):
        self.values.append(latency)
    
    @property
    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    def summary(self) -> str:
        status = "FULL" if self.count == self.size else "PARTIAL"
        return f"Measurements: {self.count} | Average: {self.average:.3f}s | Status: {status}"


def measure_qr_latency(frame, timestamp: datetime) -> Optional[float]:
    """Calculate latency from QR code in frame."""
    qr_data = read_qr_code(frame)
    if not qr_data:
        return None
    
    try:
        qr_time = datetime.fromisoformat(qr_data)
        # Positive latency = camera is behind QR generation
        return (timestamp - qr_time).total_seconds()
    except ValueError:
        return None


def setup_camera(camera_id: int) -> cv2.VideoCapture:
    """Initialize camera with optimal settings."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering for real-time capture
    return cap


def setup_iphone_camera() -> iPhone:
    """Initialize iPhone camera."""
    try:
        devices = pypeertalk.get_connected_devices()
        if not devices:
            raise RuntimeError("No iPhone devices connected via USB")
        
        iphone = iPhone(
            name="iphone_calibration",
            server_endpoint="tcp://*:0",  # sys assign available port
            iphone_device_info=devices[0],
            camera_configs=IPHONE_INPUT_STREAMS_CONFIG,
            camera_latency_s=0.0, 
            display_resolution_WH=None, 
            logger_endpoint="tcp://localhost:55555",
        )
        return iphone
    except Exception as e:
        raise RuntimeError(f"Failed to initialize iPhone camera: {e}")


@contextmanager
def qr_generator(refresh_rate: float):
    """Manage QR code generation process."""
    process = Process(
        target=partial(dynamic_qr_timecode, refresh_rate_hz=refresh_rate),
        daemon=False
    )
    
    try:
        process.start()
        yield
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def run_measurement(camera_source: Union[int, str], eval_latency: bool, window_size: int):
    """Main measurement loop."""
    # Setup camera based on source type
    cap = None
    iphone = None
    
    if isinstance(camera_source, int):
        # webcam
        cap = setup_camera(camera_source)
        camera_type = "webcam"
        camera_name = f"Camera {camera_source}"
    elif camera_source == "iphone":
        # iPhone 
        iphone = setup_iphone_camera()
        camera_type = "iphone"
        camera_name = "iPhone main camera"
    else:
        raise ValueError(f"Unsupported camera source: {camera_source}")
    
    window = LatencyWindow(window_size) if window_size > 0 else None
    
    print(f"{camera_name} ready. Press 'q' to quit.")
    if window:
        print(f"Window averaging: {window_size} samples")
    else:
        print("Window averaging: disabled")
    
    try:
        while True:
            frame = None
            timestamp = None
            
            if camera_type == "webcam" and cap is not None:
                success, frame = cap.read()
                timestamp = datetime.now()  # Capture timestamp immediately after frame read
                
                if not success:
                    continue
                    
            elif camera_type == "iphone" and iphone is not None:
                try:
                    frames_dict, _ = iphone.get_images_bgr_hwc_and_timestamp()
                    timestamp = datetime.now()
                    
                    if "main" not in frames_dict:
                        print("Warning: 'main' camera not found in iPhone frames")
                        continue
                        
                    frame = frames_dict["main"]
                    if frame is None:
                        continue
                        
                except Exception as e:
                    print(f"Error getting iPhone frame: {e}")
                    continue
            
            if frame is None or timestamp is None:
                continue
            
            # Measure latency if enabled
            if eval_latency:
                latency = measure_qr_latency(frame, timestamp)
                if latency is not None:
                    if window:
                        window.add(latency)
                        print(f"Latency: {latency:.3f}s | Avg: {window.average:.3f}s ({window.count}/{window.size})")
                    else:
                        print(f"Latency: {latency:.3f}s")
            
            # Display and check for quit (uncommented for debugging)
            # cv2.imshow(f"{camera_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Extract ASCII from key code
                break
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        if window and window.count > 0:
            print(f"Final stats: {window.summary()}")
    
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = ArgumentParser(description="Camera latency measurement tool")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera index (ignored if --use_iphone is set)")
    parser.add_argument("--use_iphone", action="store_true", help="Use iPhone camera instead of regular camera")
    parser.add_argument("--eval_latency", action="store_true", help="Enable latency measurement")
    parser.add_argument("--qr_refresh_rate", type=float, default=60.0, help="QR refresh rate (Hz)")
    parser.add_argument("--sliding_window", type=int, default=0, help="Window size for averaging")
    
    args = parser.parse_args()
    
    camera_source = "iphone" if args.use_iphone else args.camera_id
    
    if args.eval_latency:
        # Run QR generator in separate process for latency measurement
        with qr_generator(args.qr_refresh_rate):
            run_measurement(camera_source, args.eval_latency, args.sliding_window)
    else:
        # Camera preview mode without latency measurement
        run_measurement(camera_source, args.eval_latency, args.sliding_window)


if __name__ == "__main__":
    main() 