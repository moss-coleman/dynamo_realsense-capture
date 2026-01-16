from dynamo.realsense_device_manager import DeviceManager
from dynamo import calibration
# from dynamo import calibration_robot
import pyrealsense2 as rs


# 1280, 720 
# 848, 480
resolutionWidth = 1280
resolutionHeight = 720 
frameRate = 15 

rsConfig = rs.config()
rsConfig.enable_stream(rs.stream.depth, resolutionWidth, resolutionHeight, rs.format.z16, frameRate)
rsConfig.enable_stream(rs.stream.color, resolutionWidth, resolutionHeight, rs.format.bgr8, frameRate)

deviceManager = DeviceManager(rs.context(), rsConfig)

fileName = './chessboardCalibration.cal' #file to save transformation matrices

#--- Large paper Chessboard on cardboard --- #
chessboardWidth = 3 #number of corners in width
chessboardHeight = 4 #number of corners in height
chessboardSquareSize = 0.07175#chessboard square size in meters

#--- Small charuco board --- #
# chessboardWidth = 7 #number of corners in width
# chessboardHeight = 7 #number of corners in height
# chessboardSquareSize = 0.015 #chessboard square size in meters

#--- Justins board --- #
# chessboardWidth = 7 #number of corners in width
# chessboardHeight = 4 #number of corners in height
# chessboardSquareSize = 0.033 #chessboard square size in meters


cameraList = list(deviceManager._available_devices)
print("cameraList", cameraList)

# transformationMatrices = calibration_robot.new(fileName, deviceManager, chessboardWidth, chessboardHeight, chessboardSquareSize)
transformationMatrices = calibration.new(fileName, deviceManager, chessboardWidth, chessboardHeight, chessboardSquareSize)
# transformationMatrices = calibration.newIterative(fileName, deviceManager, cameraList, chessboardWidth, chessboardHeight, chessboardSquareSize)
# def newIterative(fileName,deviceManager, cameraList, chessboardHeight, chessboardWidth, chessboardSquareSize):
