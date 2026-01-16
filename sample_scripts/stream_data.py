from dynamo.realsense_device_manager import DeviceManager
from dynamo import calibration, stream
import pyrealsense2 as rs

resolutionWidth = 848
resolutionHeight = 480
frameRate = 30

rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.depth, resolutionWidth, resolutionHeight, rs.format.z16, frameRate)
rsConfig.enable_stream(rs.stream.infrared, 1, resolutionWidth, resolutionHeight, rs.format.y8, frameRate)


# transformationMatrices = calibration.load('newCalibration.cal')

calibration_file = "../data/hardcoded_calibration.cal"
try:
    device_transformations = calibration.load(calibration_file) # loads the file

    # Assuming device_manager is already set up
    # Now pass the loaded transformations to stream.start
    # stream.start(device_manager, device_transformations, save_directory, stream_time)

    print(f"Loaded hardcoded transformations for cameras: {list(device_transformations.keys())}")
    # Proceed with using stream.start...

except FileNotFoundError:
    print(f"Error: Calibration file '{calibration_file}' not found.")
except Exception as e:
    print(f"Error loading calibration file: {e}")





deviceManager = DeviceManager(rs.context(), rsConfig)
deviceManager.load_settings_json('../capture_settings/clampedCaptureSettings.json')
deviceManager.enable_all_devices()

# folder to save the data in
data_location = "../data/"

# length of time to stream the data for
#TODO this only works for time >= 1.0, less than this breaks the code
time = 5.0

stream.start(deviceManager, device_transformations, data_location, time)


