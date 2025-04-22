import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    print('pyrealsense2 is not installed. Please install it by running: pip install pyrealsense2')
from applebot.utils.common_utils import read_yaml


def rs_intrinsics_to_opencv_intrinsics(intr):
    D = np.array(intr.coeffs)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    return K, D


def get_intrinsics(pipeline_profile, stream=rs.stream.color):
    stream_profile = pipeline_profile.get_stream(stream) # Fetch stream profile for depth stream
    intr = stream_profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    return rs_intrinsics_to_opencv_intrinsics(intr)


class CaptureRS:
    def __init__(self, serial_number=None, intrinsics=None, auto_close=False):
        self.init_serial_number = serial_number

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
        # 1280 x 720 is the highest realsense dep can get

        # Start streaming
        pipeline_profile = self.pipeline.start(config)

        # And get the device info
        print(f'Connected to {self.init_serial_number}')

        # get the camera intrinsics
        # print('Using default intrinsics from camera.')
        if intrinsics is None:
            self.intrinsics = get_intrinsics(pipeline_profile)
        else:
            self.intrinsics = intrinsics

        if auto_close:
            def close_rs():
                print(f'Closing RealSense camera: {self.init_serial_number}')
                self.close()
            import atexit
            atexit.register(close_rs)

    def capture(self, dep_only=False):
        # Wait for a coherent pair of frames: depth and color
        while True:
            frameset = self.pipeline.wait_for_frames(timeout_ms=5000)

            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)

            # Update color and depth frames:
            aligned_depth_frame = frameset.get_depth_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(frameset.get_color_frame().get_data()).copy()
            if dep_only:
                # depth_frame = frameset.get_depth_frame()
                # if not depth_frame: continue
                # depth_image = np.asanyarray(depth_frame.get_data())
                return depth_image
            return color_image, depth_image

    def skip_frames(self, n):
        for i in range(n):
            _ = self.pipeline.wait_for_frames()

    def close(self):
        self.pipeline.stop()


def get_camera_serial_from_name(camera_name):
    camera_name_to_serial = read_yaml('./config.yml').camera_name_to_serial
    return camera_name_to_serial[camera_name]


def get_realsense_capturer(camera_name: str, auto_close: bool = True, skip_frames: int = 0) -> CaptureRS:
    camera_serial = get_camera_serial_from_name(camera_name)
    capture = CaptureRS(serial_number=camera_serial, intrinsics=None, auto_close=auto_close)
    if skip_frames > 0:
        capture.skip_frames(skip_frames)
    return capture


def get_realsense_capturer_dict(cameras: list[str], auto_close: bool = True, skip_frames: int = 0) -> dict[str, CaptureRS]:
    captures = {}
    for camera_name in cameras:
        captures[camera_name] = get_realsense_capturer(camera_name, auto_close=auto_close, skip_frames=skip_frames)
    return captures
