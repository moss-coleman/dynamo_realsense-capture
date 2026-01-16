from dynamo.realsense_device_manager import DeviceManager
import pyrealsense2 as rs

resolutionWidth = 848
resolutionHeight = 480
frameRate = 90

rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.depth, resolutionWidth, resolutionHeight, rs.format.z16, frameRate)

deviceManager = DeviceManager(rs.context(), rsConfig)

deviceManager.enable_device_from_file('../sample_files/20190418_142057.bag')

frame = deviceManager.poll_frames()
