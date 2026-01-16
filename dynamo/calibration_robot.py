__doc__ = \
"""
Calibrate multiple Intel RealSense D4XX cameras to a single global coordinate system using a defined checkerboard or Charuco board.
Now part of the dynamo/calibration.py file. This file may be outdated or intended for robot-specific calibration routines.
"""

##########################################################################################################################################
##                             License: Apache 2.0. See LICENSE and LICENSE.librealsense files in root directory.		                ##
##########################################################################################################################################
## This code was inspired from the librealsense box_dimensioner_multicam example, and may contain certain lines of code from this file: ##
## (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/calibration_kabsch.py).##
##########################################################################################################################################


from turtle import color
import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco # <--- Added Import
import numpy as np
import time
import pickle
import yaml

# Check if running in main package or as standalone script for module import
try:
    from .realsense_device_manager import DeviceManager
    from .calculate_rmsd import *
except ImportError:
    # Fallback for running script directly
    # This assumes the script is run from the parent directory of 'dynamo'
    # Or that 'dynamo' is in the Python path
    print("Running as standalone script, attempting different import path.")
    try:
        from realsense_device_manager import DeviceManager
        from calculate_rmsd import *
    except ImportError:
        raise ImportError("Could not import necessary dynamo modules. Ensure the script is run correctly relative to the dynamo package or the package is installed.")


def invTrans(matrix):
    """
    Returns inverse of a transformation matrix

    Parameters
    ----------
    matrix : (4,4) array
        Input transformation matrix

    Returns
    -------
    invMatrix : (4,4) array
        Inverse of input transformation matrix
    """
    invRot = np.eye(4)
    invRot[0:3,0:3] = matrix[0:3,0:3].T
    invTrans = np.eye(4)
    invTrans[0:3,3] = -matrix[0:3,3]
    invMatrix = np.matmul(invRot,invTrans)
    return invMatrix

def load(fileName):
    """
    Calibration parameters for previously connected cameras are loaded from a pickle file format.
    Calibration parameters for each camera include a 4x4 transformation matrix and rmsd error of calibration

    Parameters
    ----------
    fileName : str
        Filename of stored calibration parameters.

    Returns
    -------
    devicesTransformation : dict
        Keys of camera's serial number holding dictionary of calibration parameters per camera

    Example
    -------
    load('savedCalibration.cal')
    """
    with open(fileName,'rb') as f:
        devicesTransformation = pickle.load(f)
    return devicesTransformation

def newChessboard(fileName,deviceManager, chessboardHeight, chessboardWidth, chessboardSquareSize):
    """ 
    New calibration parameters for each connected camera and are created and saved in a pickle file format.
    
    Cameras must be all be viewing the calibration checkerboard. 

    Calibration parameters for each camera include a 4x4 transformation matrix and rmsd error of calibration 

    Parameters
    ----------
    fileName : str
        Filename to store calibration parameters.
    
    deviceManager : DeviceManager object
        realsense_device_manager object which manages connections to all cameras
    
    chessboardHeight : int
        Number of chessboard intersections defining height of target chessboard

    chessboardWidth : int
        Number of chessboard intersections defining width of target chessboard
    
    chessboardSquareSize : float
        Dimension of side of chessboard (m)

    Returns
    -------
    devicesTransformation : dict
        dictionary with keys of camera's serial number holding dictionary of calibration parameters per camera

    Example
    -----
        new('savedCalibration.cal')
    """

    deviceManager.enable_all_devices()
               
    time.sleep(1) #let autoexposure on cameras stabilize over one second 
    cameraSet = deviceManager._enabled_devices
    chessboardLocations = detectChessboard(deviceManager, cameraSet, chessboardHeight, chessboardWidth, chessboardSquareSize) #return locations of chessboards from reference frame of each camera
    devicesTransformations = poseTransformation(chessboardLocations, chessboardHeight, chessboardWidth, chessboardSquareSize) #return dictionary of 
    
    try:
        with open(fileName,'wb') as f:
            pickle.dump(devicesTransformations, f)
        print(f"Chessboard calibration successful. Transformations saved to {fileName}")
        
        # Also save in YAML format
        board_parameters = {
            'height': chessboardHeight,
            'width': chessboardWidth,
            'squareSize': chessboardSquareSize
        }
        save_calibration_yaml(devicesTransformations, fileName, board_parameters)
        
    except Exception as e:
        print(f"Error saving calibration file {fileName}: {e}")
        return {}
    
    return devicesTransformations

def transformation_matrix_to_yaml_format(transform_matrix, camera_serial, board_parameters, error=0.0, method_name='charuco', orientation='rotation_matrix'):
    """
    Convert a 4x4 transformation matrix to YAML format compatible with the existing calibration config.
    
    Parameters
    ----------
    transform_matrix : np.ndarray
        4x4 transformation matrix
    camera_serial : str
        Camera serial number
    board_parameters : dict
        Board parameters used for calibration
    error : float
        Calibration error (RMSD)
    method_name : str
        Name of the calibration method used
        
    Returns
    -------
    dict
        Dictionary in the format expected by the YAML calibration config
    """
    # Extract rotation and translation
    rotation_matrix = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]

    if orientation == 'rotation_matrix':
        # save the transformation matrix in yaml format
        camera_entry = {
            'description': f'Calibrated using {method_name} method (error: {error:.6f})',
            'transformation_matrix': transform_matrix.tolist(),
        }

    elif orientation == 'euler_angles':
        # Convert rotation matrix to Euler angles (in degrees)
        # Using ZYX convention (yaw, pitch, roll)
        euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        # Convert to degrees and reorder to match expected format
        euler_deg = [euler_angles[2], euler_angles[1], euler_angles[0]]  # ZYX -> XYZ
        
        # Create camera entry
        camera_entry = {
            'description': f'Calibrated using {method_name} method (error: {error:.6f})',
            'orientation_euler_deg': euler_deg,
            'position': translation.tolist()
        }
    
    return camera_entry

def save_calibration_yaml(devicesTransformations, fileName, board_parameters, error=0.0):
    """
    Save calibration data in YAML format compatible with existing calibration config files.
    
    Parameters
    ----------
    devicesTransformations : dict
        Dictionary of camera transformations
    fileName : str
        Base filename for the .cal file
    board_parameters : dict
        Board parameters used for calibration
    error : float
        Calibration error (RMSD)
    """
    try:
        # Create YAML filename by replacing .cal extension
        yaml_filename = fileName.replace('.cal', '.yaml')
        
        # Determine board type and prepare calibration settings
        if 'squaresX' in board_parameters and 'squaresY' in board_parameters:
            # Charuco board
            board_type = 'charuco_board'
            board_settings = {
                'marker_length': board_parameters.get('markerLength', 0.011),
                'square_length': board_parameters.get('squareLength', 0.015),
                'squares_x': board_parameters.get('squaresX', 8),
                'squares_y': board_parameters.get('squaresY', 8)
            }
            method_name = 'charuco'
        elif 'height' in board_parameters and 'width' in board_parameters:
            # Chessboard
            board_type = 'chessboard'
            board_settings = {
                'height': board_parameters.get('height', 6),
                'width': board_parameters.get('width', 9),
                'square_size': board_parameters.get('squareSize', 0.025)
            }
            method_name = 'chessboard'
        else:
            # Unknown board type, use generic settings
            board_type = 'calibration_board'
            board_settings = {}
            method_name = 'unknown'
        
        calibration_settings = {
            board_type: board_settings,
            'enable_temporal_filtering': False,
            'images_per_camera': 1  # Single capture for direct method
        }
        
        # Prepare cameras section
        cameras = {}
        for camera_serial, transform_data in devicesTransformations.items():
            # Handle different data structures
            if isinstance(transform_data, np.ndarray) and transform_data.shape == (4, 4):
                # Direct transformation matrix
                transform_matrix = transform_data
                cameras[camera_serial] = transformation_matrix_to_yaml_format(
                    transform_matrix, camera_serial, board_parameters, error, method_name
                )
            elif isinstance(transform_data, list) and len(transform_data) >= 1:
                # List format [transform_matrix, error]
                transform_matrix = transform_data[0]
                if isinstance(transform_matrix, np.ndarray) and transform_matrix.shape == (4, 4):
                    cameras[camera_serial] = transformation_matrix_to_yaml_format(
                        transform_matrix, camera_serial, board_parameters, 
                        transform_data[1] if len(transform_data) > 1 else error, 
                        method_name
                    )
        
        # Create complete YAML structure
        yaml_data = {
            'calibration_settings': calibration_settings,
            'cameras': cameras
        }
        
        # Save YAML file
        with open(yaml_filename, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Calibration data also saved in YAML format: {yaml_filename}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save YAML file: {e}")

def newCharuco(fileName, deviceManager, board_parameters, save_files=True): 
    """
    New calibration parameters for each connected camera are created and saved in a pickle file format.

    Cameras must all be viewing the calibration board (chessboard or Charuco).

    Calibration parameters for each camera include a 4x4 transformation matrix and rmsd error of calibration.

    Parameters
    ----------
    fileName : str
        Filename to store calibration parameters.

    deviceManager : DeviceManager object
        realsense_device_manager object which manages connections to all cameras

    board_type : str
        Type of calibration board used ('chessboard' or 'charuco').

    board_parameters : dict
        Dictionary containing parameters for the specified board type.
        For 'chessboard': {'height': int, 'width': int, 'squareSize': float}
        For 'charuco': {'squaresX': int, 'squaresY': int, 'squareLength': float, 'markerLength': float, 'dictionary': aruco_dictionary}
                         (e.g., 'dictionary': aruco.DICT_6X6_250)

    Returns
    -------
    devicesTransformation : dict
        dictionary with keys of camera's serial number holding dictionary of calibration parameters per camera

    Example
    -----
    # Chessboard example
    chessboard_params = {'height': 6, 'width': 9, 'squareSize': 0.025}
    new('cal_chessboard.cal', deviceManager, 'chessboard', chessboard_params)

    # Charuco example
    charuco_params = {'squaresX': 5, 'squaresY': 7, 'squareLength': 0.04, 'markerLength': 0.02, 'dictionary': aruco.DICT_6X6_250}
    new('cal_charuco.cal', deviceManager, 'charuco', charuco_params)
    """
    deviceManager.enable_all_devices()

    time.sleep(1) # let autoexposure on cameras stabilize over one second
    cameraSet = list(deviceManager._enabled_devices.keys()) # Get list of enabled camera serials

    print(f"Starting calibration for cameras: {cameraSet}")

    boardLocations = detectCharucoBoard(deviceManager, cameraSet, board_parameters) # return locations of board corners from reference frame of each camera
    devicesTransformations = boardLocations # return dictionary of transformations
    if save_files:
        try:
            with open(fileName,'wb') as f:
                pickle.dump(devicesTransformations, f)
            print(f"Calibration successful. Transformations saved to {fileName}")
            
            # Also save in YAML format
            save_calibration_yaml(devicesTransformations, fileName, board_parameters)
            
        except Exception as e:
            print(f"Error saving calibration file {fileName}: {e}")
            # deviceManager.disable_all_devices() # Consider disabling here
            return {} # Return empty dict if saving fails


    # Consider whether to disable devices here or leave it to the calling script
    # deviceManager.disable_all_devices()
    return devicesTransformations


def newIterative(fileName, deviceManager, cameraList, board_type, board_parameters):
    """
    New calibration parameters for each connected camera are created and saved iteratively.

    Function will iterate through camera list and will search for the calibration board
    between each consecutive set of two cameras in cameraList.
    The user must move the board between the sets of cameras as the function works through the list.

    Calibration parameters for each camera include a 4x4 transformation matrix and rmsd error of calibration.

    Parameters
    ----------
    fileName : str
        Filename to store calibration parameters.

    deviceManager : DeviceManager object
        realsense_device_manager object which manages connections to all cameras

    cameraList : list
        list of serial numbers to calibrate cameras in order

    board_type : str
        Type of calibration board used ('chessboard' or 'charuco').

    board_parameters : dict
        Dictionary containing parameters for the specified board type (see 'new' function docstring for details).


    Returns
    -------
    deviceTransformations : dict
        dictionary with keys of camera's serial number holding dictionary of calibration parameters per camera

    Example
    -----
        # Chessboard example
        chessboard_params = {'height': 6, 'width': 9, 'squareSize': 0.025}
        cam_list = ['serial1', 'serial2', 'serial3']
        newIterative('cal_iterative_chess.cal', deviceManager, cam_list, 'chessboard', chessboard_params)

        # Charuco example
        charuco_params = {'squaresX': 5, 'squaresY': 7, 'squareLength': 0.04, 'markerLength': 0.02, 'dictionary': aruco.DICT_6X6_250}
        newIterative('cal_iterative_charuco.cal', deviceManager, cam_list, 'charuco', charuco_params)
    """

    deviceManager.enable_all_devices() # Ensure devices are enabled at the start

    deviceTransformations = {}
    # Initialize transformations with identity matrix and zero RMSD for all cameras in the list
    for cam_serial in cameraList:
         deviceTransformations[cam_serial] = [np.eye(4), 0.0]


    relativeTransformations = {}

    time.sleep(1) #let autoexposure on cameras stabilize over one second

    # TODO: create a list of pairs of cameras to calibrate, as not all cameras are suitable and can be calibrated together

    for i in range(len(cameraList) - 1):
        cam1_serial = cameraList[i]
        cam2_serial = cameraList[i+1]
        cset = [cam1_serial, cam2_serial]

        fstring = f"Place the calibration board so it's visible by cameras {cam1_serial} and {cam2_serial}. Press ENTER to start calibration for this pair."
        input(fstring)

        setTransformations = {}
        # Detect board for the current pair
        if board_type == 'chessboard':
            boardLocations = detectChessboard(deviceManager, cset, board_parameters['height'], board_parameters['width'], board_parameters['squareSize'])
        elif board_type == 'charuco':
            boardLocations = detectCharucoBoard(deviceManager, cset, board_parameters)

        # Check if both cameras detected the board
        if len(boardLocations) == 2 and cam1_serial in boardLocations and cam2_serial in boardLocations:
            # Calculate transformations relative to the board
            if board_type == 'chessboard':
                pairTransformations = poseTransformationChessboard(boardLocations, board_parameters)
            elif board_type == 'charuco':
                pairTransformations = poseTransformationCharuco(boardLocations)

            # Ensure transformations were calculated successfully for both
            if cam1_serial in pairTransformations and cam2_serial in pairTransformations:
                # Calculate transformation from cam2 to cam1 (T_cam2_to_cam1)
                T_cam1_to_board = pairTransformations[cam1_serial][0] # This is T_cam1_to_obj
                T_cam2_to_board = pairTransformations[cam2_serial][0] # This is T_cam2_to_obj

                # T_cam2_to_cam1 = T_board_to_cam1 * T_cam2_to_board = inv(T_cam1_to_board) * T_cam2_to_board
                T_board_to_cam1 = invTrans(T_cam1_to_board)
                T_cam2_to_cam1 = np.matmul(T_board_to_cam1, T_cam2_to_board) # Transformation from cam2 frame to cam1 frame

                # Store the relative transformation T_cam(i+1)_to_cam(i)
                relativeTransformations[(cam2_serial, cam1_serial)] = T_cam2_to_cam1
                fstring = f"Cameras {cam1_serial} and {cam2_serial} calibrated relative to each other. Press ENTER to continue."
                input(fstring)
            else:
                fstring = f"Error: Could not calculate pose transformation for one or both cameras ({cam1_serial}, {cam2_serial}) even though board was detected. Skipping this pair. Press ENTER to continue."
                input(fstring)
                relativeTransformations[(cam2_serial, cam1_serial)] = np.eye(4) # Store identity if pose failed

        else:
             detected_cams = list(boardLocations.keys())
             fstring = f"Error: Could not detect the board in both cameras ({cam1_serial}, {cam2_serial}). Detected in: {detected_cams}. Skipping this pair. Press ENTER to continue."
             input(fstring)
             # Store identity if detection failed for the pair to avoid breaking the chain
             relativeTransformations[(cam2_serial, cam1_serial)] = np.eye(4)


    # Chain the transformations: T_world = T_N * T_N-1 * ... * T_1 * T_0 (where T_0 is identity)
    # Assume cameraList[0] is the reference camera (world frame)
    # T_cam_i_to_world = T_cam_1_to_world * T_cam_2_to_cam1 * ... * T_cam_i_to_cam_i-1
    # Note: Stored relativeTransformations[(cam_k, cam_k-1)] is T_cam_k_to_cam_k-1

    # Set the first camera's transformation to identity (relative to world = itself)
    deviceTransformations[cameraList[0]][0] = np.eye(4)

    for i in range(len(cameraList) - 1):
        cam_curr_serial = cameraList[i]
        cam_next_serial = cameraList[i+1]

        # Get the transformation from the next camera frame to the current camera frame
        T_next_to_curr = relativeTransformations.get((cam_next_serial, cam_curr_serial))

        if T_next_to_curr is not None:
            # Get the transformation from the current camera frame to the world frame
            T_curr_to_world = deviceTransformations[cam_curr_serial][0]

            # Calculate the transformation from the next camera frame to the world frame
            # T_next_to_world = T_curr_to_world * T_next_to_curr
            T_next_to_world = np.matmul(T_curr_to_world, T_next_to_curr)
            deviceTransformations[cam_next_serial][0] = T_next_to_world
            # RMSD is not directly calculated in this chained approach, keep as 0 or use pair-wise RMSD if needed
            # deviceTransformations[cam_next_serial][1] = pairTransformations.get(cam_next_serial, [None, 0.0])[1] # Maybe store pairwise RMSD?
        else:
             print(f"Warning: Missing relative transformation between {cam_next_serial} and {cam_curr_serial}. Setting {cam_next_serial}'s transform based on {cam_curr_serial}'s current transform.")
             # This might happen if an identity matrix was stored due to previous failure
             deviceTransformations[cam_next_serial][0] = deviceTransformations[cam_curr_serial][0] # Or handle error differently


    print("Final calculated transformations relative to {}:".format(cameraList[0]))
    for cam_serial in cameraList:
        print(f"  Camera {cam_serial}:")
        print(deviceTransformations[cam_serial][0])
        # print(f"  RMSD (pairwise, if available): {deviceTransformations[cam_serial][1]}")

    try:
        with open(fileName,'wb') as f:
            pickle.dump(deviceTransformations, f)
        print(f"Iterative calibration successful. Transformations saved to {fileName}")
        
        # Also save in YAML format
        save_calibration_yaml(deviceTransformations, fileName, board_parameters)
        
    except Exception as e:
        print(f"Error saving iterative calibration file {fileName}: {e}")
        # deviceManager.disable_all_devices() # Consider disabling here
        return {} # Return empty dict if saving fails

    # deviceManager.disable_all_devices() # Consider disabling devices at the end
    return deviceTransformations


# def detectChessboard(deviceManager, cameraSet, chessboardHeight, chessboardWidth, chessboardSquareSize):
#     """ 
#     Chessboard locations are computed for each connected RealSense camera

#     Parameters
#     ----------
#     deviceManager : DeviceManager object
#         realsense_device_manager object which manages connections to all cameras

#     cameraSet : list
#         list of camera serial numbers to detect the chessboard
    
#     chessboardHeight: int
#         Number of chessboard intersections defining height of target chessboard

#     chessboardWidth: int
#         Number of chessboard intersections defining width of target chessboard
    
#     chessboardSquareSize: float
#         Dimension of side of chessboard (m)

#     Returns
#     -------
#     chessboardLocations : dict
#         dictionary with keys of camera's serial number holding detected corners, local 2D and 3D coordinates of detected corners, and their valid depth points

#     Example
#     -----
#         detectChessboard(deviceManager, chessboardHeight, chessboardWidth, chessboardSquareSize)
#     """
#     chessboardDeviceCount = 0
#     devicesChessboardLocations = {}

#     while len(devicesChessboardLocations) < len(cameraSet): #iterate through detecting chessboard until all available devices see chessboard
#         cameraFrames = deviceManager.poll_frames()
#         devicesIntrinsics = deviceManager.get_device_intrinsics(cameraFrames)
        
#         for device, frames in cameraFrames.items(): #this will iterate through each device's serial number (which are used as keys in the frames object)
#             if not device in devicesChessboardLocations and device in cameraSet: #if the camera has not already detected the chessboard

#                 align = rs.align(rs.stream.depth) #align the color sensor to the depth sensor using the factory extrinsics
#                 alignedFrames = align.process(frames)
                
#                 colorImage = np.asanyarray(alignedFrames.get_color_frame().get_data()) 
#                 bwImage = cv2.cvtColor(colorImage,cv2.COLOR_BGR2GRAY) #convert color image to B&W image to use in openCV functions

#                 depthFrame = alignedFrames.get_depth_frame()
#                 depthIntrinsics = devicesIntrinsics[device][rs.stream.depth]  #obtain the depth sensor intrinsic properties

#                 chessboardFound, corners = cv2.findChessboardCorners(bwImage, (chessboardWidth, chessboardHeight)) #use openCV function to detect chessboard corners

#                 if chessboardFound: #if the camera sees the chessboard
#                     print(device," sees the chessboard!")
#                     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #subpix criteria
#                     points2D = cv2.cornerSubPix(bwImage, corners, (11,11), (-1,-1), criteria) #further refine corners of chessboard using sub pixeling

#                     cv2.drawChessboardCorners(bwImage, (chessboardWidth, chessboardHeight), points2D, chessboardFound) #draw chessboard corners in white over b&w image for verification
#                     cv2.imshow(device,bwImage) #show the detected chessboard corners on the image
#                     cv2.waitKey(50)

#                     points2D = np.transpose(corners, (2,0,1)) 
#                     points3D = np.zeros((3, len(points2D[0]))) #preallocate array to turn 2D chessboard corner points from camera into 3D points 
#                     validPoints = [False] * len(points2D[0])

#                     for index in range(len(points2D[0])): #iterate over every corner to find 3D location of chessboard corner
#                         corner = points2D[:,index].flatten()
#                         depth = depthFrame.as_depth_frame().get_distance(round(corner[0]), round(corner[1])) #this gets the depth at the pixel for each corner
#                         if depth != 0 and depth is not None: #if the corner point has a valid depth value from the depth sensor
#                             validPoints[index] = True #sets points which have a depth value as valid

#                             #formualtion for finding 3D location of 2d point:
#                             points3D[0, index] = (corner[0]-depthIntrinsics.ppx)/depthIntrinsics.fx*depth 
#                             points3D[1, index] = (corner[1]-depthIntrinsics.ppy)/depthIntrinsics.fy*depth
#                             points3D[2, index] = depth

#                     devicesChessboardLocations[device] = corners, points2D, points3D, validPoints #save array for each camera of detected corners, their 2D locations, their 3D locations, and if they have a valid depth value
#                     chessboardDeviceCount += 1

#                 if not chessboardFound: #if the camera doesn't see the chessboard
#                     devicesChessboardLocations = {}
#                     #chessboardDeviceCount = 0
#                     cv2.imshow(device,bwImage)
#                     cv2.waitKey(50)
#                     print(device," cannot detect the chessboard!")

#         time.sleep(1)
#         cv2.destroyAllWindows()
#     return devicesChessboardLocations


def detectCharucoBoard(deviceManager, cameraSet, board_parameters):
    """
    Detects charuco calibration board corners/markers for each specified camera.
    inputs:
        deviceManager: DeviceManager object
        cameraSet: list of camera serial numbers to detect the board in.
        board_parameters: dictionary containing parameters for the specified board type (see 'new' function docstring for details).
    outputs:
        devicesBoardLocations: dictionary with keys of camera's serial number holding the pose
        of the board in the camera frame.
    """
    boardDeviceCount = 0
    # sample_count = 0
    sample_taken = False
    # target_sample_count = 2 
    sample_taken = False
    devicesBoardLocations = {}
    aruco_dict = aruco.getPredefinedDictionary(board_parameters['dictionary'])

    charuco_board = aruco.CharucoBoard(
            (board_parameters['squaresX'], board_parameters['squaresY']),
            board_parameters['squareLength'],
            board_parameters['markerLength'],
            aruco_dict)
    charuco_board.setLegacyPattern(True)
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
    aruco_params = aruco.DetectorParameters()
    if hasattr(aruco_params, 'cornerRefinementMethod'):
        aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    # Create camera index mapping and window setup
    camera_indices = {}
    for i, device_serial in enumerate(cameraSet):
        camera_indices[device_serial] = i

    # Create and position windows for each camera
    for device_serial in cameraSet:
        camera_index = camera_indices[device_serial]
        window_name = f"Camera {camera_index + 1} ({device_serial})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x_offset = (camera_index % 2) * 800  # 2 columns
        y_offset = (camera_index // 2) * 600  # 2 rows
        cv2.moveWindow(window_name, x_offset, y_offset)
        cv2.resizeWindow(window_name, 800, 600)  # Set reasonable window size

    # Dictionary to store pose samples for each camera: {serial: [(rvec, tvec), ...]}
    pose_samples = {serial: [] for serial in cameraSet}
    corners_samples = {serial: [] for serial in cameraSet}

    # while sample_count < target_sample_count:
    while sample_taken == False:
        cameraFrames = deviceManager.poll_frames()
        devicesIntrinsics = deviceManager.get_device_intrinsics(cameraFrames)
        
        # Store the current pose for each camera in this iteration
        current_poses = {}
        current_corners = {}

        for device_serial in cameraSet:
            if device_serial in cameraFrames and cameraFrames[device_serial]:
                frames = cameraFrames[device_serial]
                
                align = rs.align(rs.stream.depth)
                alignedFrames = align.process(frames)

                # alignedFrames = frames

                # color_frame = frames.get_color_frame()
                colorImage = np.asanyarray(alignedFrames.get_color_frame().get_data())
                bwImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)

                depthFrame = alignedFrames.get_depth_frame()
                # depthIntrinsics = devicesIntrinsics[device_serial][rs.stream.depth]

                color_intrinsic_matrix = devicesIntrinsics[device_serial][rs.stream.color]
                # Convert to numpy array format for opencv
                color_intrinsic_matrix = np.array([
                    [color_intrinsic_matrix.fx, 0, color_intrinsic_matrix.ppx], 
                    [0, color_intrinsic_matrix.fy, color_intrinsic_matrix.ppy], 
                    [0, 0, 1]
                ])
                depth_intrinsic_params = devicesIntrinsics[device_serial][rs.stream.depth]
                depth_intrinsic_matrix = np.array([
                    [depth_intrinsic_params.fx, 0, depth_intrinsic_params.ppx], 
                    [0, depth_intrinsic_params.fy, depth_intrinsic_params.ppy], 
                    [0, 0, 1]
                ])
                
                dist_coeffs = np.zeros((4, 1))
                
                # Detect the board in the image
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(bwImage)
                # print("Initial size of charuco_corners:", len(charuco_corners))
                    
                # Debug: Show ChArUco detection details
                if charuco_ids is not None and len(charuco_ids) > 0:

                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #subpix criteria
                    charuco_corners = cv2.cornerSubPix(bwImage, charuco_corners, (11,11), (-1,-1), criteria) #further refine corners of chessboard using sub pixeling
                    # charuco_corners, charuco_ids = cv2.interpolateCornersCharuco(marker_corners, marker_ids, bwImage, charuco_board, charuco_corners, charuco_ids) 
                    # Refine the charuco corners
                    # refined_corners = charuco_detector.refineDetectedCorners(charuco_corners, charuco_ids, bwImage)
                    # charuco_corners = refined_corners[0]
                    # charuco_ids = refined_corners[1]

                    obj_points, img_points = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                    # print("charuco_corners", len(charuco_corners))
                    # print("obj_points", len(obj_points))
                    # print("img_points", len(img_points))
                    
                    # Use robust pose estimation with multiple methods
                    # rvec, tvec = robust_pose_estimation(obj_points, img_points, color_intrinsic_matrix, dist_coeffs)
                    rvec, tvec = robust_pose_estimation(obj_points, img_points, depth_intrinsic_matrix, dist_coeffs)
                    
                    if rvec is not None and tvec is not None:
                        # Draw pose axes on the image
                        cv2.drawFrameAxes(colorImage, depth_intrinsic_matrix, dist_coeffs, rvec, tvec, 0.1)
                        
                        # Draw ChArUco corners
                        cv2.aruco.drawDetectedCornersCharuco(colorImage, charuco_corners, charuco_ids)

                        cv2.aruco.drawDetectedMarkers(colorImage, marker_corners, marker_ids)

                        # Store the pose for this camera for this frame
                        current_poses[device_serial] = (rvec, tvec)

                        # points2D = np.transpose(charuco_corners, (2,0,1)) 
                        # points3D = np.zeros((3, len(points2D[0]))) #preallocate array to turn 2D chessboard corner points from camera into 3D points 
                        # validPoints = [False] * len(points2D[0])

                        # for index in range(len(points2D[0])): #iterate over every corner to find 3D location of chessboard corner
                        #     corner = points2D[:,index].flatten()
                        #     depth = depthFrame.as_depth_frame().get_distance(round(corner[0]), round(corner[1])) #this gets the depth at the pixel for each corner
                        #     if depth != 0 and depth is not None: #if the corner point has a valid depth value from the depth sensor
                        #         validPoints[index] = True #sets points which have a depth value as valid

                        #         #formualtion for finding 3D location of 2d point:
                        #         points3D[0, index] = (corner[0]-depth_intrinsic_params.ppx)/depth_intrinsic_params.fx*depth 
                        #         points3D[1, index] = (corner[1]-depth_intrinsic_params.ppy)/depth_intrinsic_params.fy*depth
                        #         points3D[2, index] = depth

                        # current_corners[device_serial] = charuco_corners, points2D, points3D, validPoints #save array for each camera of detected corners, their 2D locations, their 3D locations, and if they have a valid depth value

                
                # Add camera identification text on the image
                camera_index = camera_indices[device_serial]
                cv2.putText(colorImage, f"Camera {camera_index + 1} ({device_serial})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the image in the correct window
                window_name = f"Camera {camera_index + 1} ({device_serial})"
                cv2.imshow(window_name, colorImage)

        # Wait for key press and update all windows
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        elif key == ord('c'):
            # Record the current pose for each camera if available
            for device_serial in cameraSet:
                if device_serial in current_poses:
                    pose_samples[device_serial].append(current_poses[device_serial])

                # if device_serial in current_corners:
                #     corners_samples[device_serial].append(current_corners[device_serial])
            # sample_count += 1
            # print(f"Sample {sample_count}/{target_sample_count} recorded for all cameras with detected pose.")
            print("Sample taken for all cameras with detected pose.")
            sample_taken = True

        time.sleep(0.1)
    
    # Convert the poses to a SE(3) matrix
    pose_matrices = poses_to_matrices(pose_samples)
    pose_matrices = transform_to_charuco_frame(pose_matrices)

    # Convert the corners to a SE(3) matrix
    # convert the corners to a format expected by poseTransformationCharuco
    # corners_samples_formatted = {}
    # for serial, corners in corners_samples.items():
    #     corners_samples_formatted[serial] = corners_2D_to_3D(corners['charuco_corners'], corners['charuco_ids'], corners['marker_corners'], corners['marker_ids'])
    # pose_matrices_from_corners = poseTransformationCharuco(corners_samples, board_parameters)
    # print("pose_matrices_from_corners", pose_matrices_from_corners)

    devicesBoardLocations = pose_matrices
    # devicesBoardLocations = pose_matrices_from_corners

    cv2.destroyAllWindows()
    return devicesBoardLocations

def transform_to_charuco_frame(pose_matrices):
    """
    Transform the poses to the charuco frame, where the marker position is the origin.
    """
    for serial, pose_matrix in pose_matrices.items():
        pose_matrices[serial] = invTrans(pose_matrix)
    return pose_matrices

def poses_to_matrices(pose_dict):
    """
    Convert a dictionary of camera poses to a dictionary of SE(3) matrices.
    pose_dict: dict where key is camera serial number, value is (rvec, tvec) or list of (rvec, tvec)
    Returns: dict where key is camera serial number, value is the transformation matrix T
    """
    matrices = {}
    for serial, pose in pose_dict.items():
        # If multiple poses per camera, use the first one (or average if needed)
        if isinstance(pose, list):
            # Optionally, average the poses or just use the first
            rvec, tvec = pose[0]
        else:
            rvec, tvec = pose
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        matrices[serial] = T
    return matrices

def robust_pose_estimation(obj_points, img_points, camera_matrix, dist_coeffs):
    """Robust pose estimation with multiple PnP methods and pose filtering."""
    if len(obj_points) < 4:
        return None, None
    
    # # Ensure inputs are proper numpy arrays with correct types
    # obj_points = np.array(obj_points, dtype=np.float32)
    # img_points = np.array(img_points, dtype=np.float32)
    # camera_matrix = np.array(camera_matrix, dtype=np.float32)
    # dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
    
    # Ensure distortion coefficients are in the right shape
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(-1, 1)
    
    # Try multiple PnP methods
    methods = [
        cv2.SOLVEPNP_ITERATIVE,
        cv2.SOLVEPNP_EPNP,
        cv2.SOLVEPNP_P3P,
        cv2.SOLVEPNP_AP3P
    ]
    
    best_rvec = None
    best_tvec = None
    best_error = float('inf')
    
    for method in methods:
        try:
            if method in [cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P] and len(obj_points) >= 4:
                # For P3P methods, use only 4 points
                ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_points[:4], img_points[:4], 
                    camera_matrix, dist_coeffs,
                    flags=method
                )
            else:
                ret, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, 
                    camera_matrix, dist_coeffs,
                    flags=method
                )
                
            if ret:
                # Calculate reprojection error
                # Ensure obj_points is in the correct format for cv2.projectPoints
                obj_points_for_projection = obj_points.copy()
                if obj_points_for_projection.ndim == 2:
                    obj_points_for_projection = obj_points_for_projection.reshape(-1, 1, 3)
                projected_points, _ = cv2.projectPoints(
                    obj_points_for_projection, rvec, tvec, 
                    camera_matrix, dist_coeffs
                )
                # Ensure both arrays have the same shape for error calculation
                projected_points_2d = projected_points.reshape(-1, 2)
                if img_points.shape != projected_points_2d.shape:
                    # Reshape img_points to match projected_points if needed
                    img_points_reshaped = img_points.reshape(-1, 2)
                else:
                    img_points_reshaped = img_points
                error = np.mean(np.linalg.norm(img_points_reshaped - projected_points_2d, axis=1))
                
                if error < best_error:
                    best_error = error
                    best_rvec = rvec.copy()
                    best_tvec = tvec.copy()
                    
        except Exception as e:
            print(f"Method {method} failed: {e}")
            continue
    
    return best_rvec, best_tvec

def poseTransformationChessboard(chessboardLocations, chessboardHeight, chessboardWidth, chessboardSquareSize):
    """ 
    Transformation matrices and error are computed for each connected RealSense camera

    Parameters
    ----------
    chessboardLocations : dict
        dictionary with keys of camera's serial number holding detected corners, local 2D and 3D coordinates of detected corners, and their valid depth points

    chessboardHeight: int
        Number of chessboard intersections defining height of target chessboard

    chessboardWidth: int
        Number of chessboard intersections defining width of target chessboard
    
    chessboardSquareSize: float
        Dimension of side of chessboard (m)

    Returns
    -------
    devicesTransformation: dict
        dictionary with keys of camera's serial number with each camera's 4x4 transformation matrix and root-mean squared error

    Example
    -----
        poseTransformation(chessboardLocations, chessboardHeight, chessboardWidth, chessboardSquareSize)
    """
    devicesTransformation = {}
    for (serial, [corners, points2D, points3D, validPoints]) in chessboardLocations.items(): #for every camera which has detected the chessboard

        if len(points2D[0])<5: #check if there are at least 5 points to be able to compute transformation matrix
            print(serial, " does not have enough points to have a valid depth for calculating the transformatoin")

        else:
            chessboardPoints = np.zeros((chessboardWidth*chessboardHeight,3), np.float32) #container for 3d coordinates of chessboard corners in global coordinates of chessboard
            chessboardPoints[:,:2] = np.mgrid[0:chessboardWidth, 0:chessboardHeight].T.reshape(-1,2)
            chessboardPoints = chessboardPoints.transpose() * chessboardSquareSize
            validchessboardPoints = chessboardPoints[:,validPoints].transpose()
            validobservedchessboardPoints = points3D[:, validPoints].transpose() #take chessboard points which have been detected by depth sensor

            chessboardPointsCentered = validchessboardPoints - centroid(validchessboardPoints) #center global chessboard points so that reference frame is same for every camera
            observedchessboardCentered = validobservedchessboardPoints - centroid(validobservedchessboardPoints) 

            rotationMatrix = kabsch(chessboardPointsCentered, observedchessboardCentered) #calculate rotation between local coordiante system and global coordinate system
            rmsdValue = kabsch_rmsd(chessboardPointsCentered, observedchessboardCentered) #calculate error of rotation matrix

            translationVector = centroid(validobservedchessboardPoints) - np.matmul(centroid(validchessboardPoints), rotationMatrix) #calculate translation between local and global coordinate system
            trans = -np.matmul(rotationMatrix, translationVector.transpose())

            poseMat = np.zeros((4,4)) #build 4x4 transformation matrix
            poseMat[:3,:3] = rotationMatrix
            poseMat[:3,3] = trans.flatten()
            poseMat[3,3] = 1
            
            devicesTransformation[serial] = [poseMat, rmsdValue]
    return devicesTransformation


def poseTransformationCharuco(boardLocations, board_parameters):
    """
    Calculates transformation matrices and RMSD error for each camera based on detected board locations.

    Parameters
    ----------
    boardLocations : dict
        Dictionary from detectBoard containing detected corners/markers and coordinates.
        Format for chessboard: {serial: (points2D_valid, points3D_valid)}
        Format for charuco: {serial: (points2D_valid, charucoIds_valid, points3D_valid)}

    board_type : str
        Type of calibration board used ('chessboard' or 'charuco').

    board_parameters : dict
        Dictionary containing parameters for the specified board type (see 'new' function docstring for details).


    Returns
    -------
    devicesTransformation: dict
        dictionary with keys of camera's serial number with each camera's
        4x4 transformation matrix (camera frame to object/world frame) and
        root-mean squared error: {serial: [poseMat_cam_to_obj, rmsdValue]}

    """
    devicesTransformation = {}

    aruco_dict = aruco.getPredefinedDictionary(board_parameters['dictionary'])
    charuco_board = aruco.CharucoBoard(
        (board_parameters['squaresX'], board_parameters['squaresY']),
        board_parameters['squareLength'],
        board_parameters['markerLength'],
        aruco_dict)

    charucoboardWidth = board_parameters['squaresX'] -1
    charucoboardHeight = board_parameters['squaresY'] -1
    charucoboardSquareLength = board_parameters['squareLength']
    charucoboardMarkerLength = board_parameters['markerLength']

    # Get the 3D coordinates of all Charuco corners from the board definition
    # Note: getChessboardCorners() returns a (N, 3) array or similar structure
    allCharucoCorners3D = charuco_board.getChessboardCorners() # Get all potential corners
    print("Potential Charuco corners", len(allCharucoCorners3D))
    print("allCharucoCorners3D", allCharucoCorners3D)
    if allCharucoCorners3D is None or len(allCharucoCorners3D) == 0:
            print("Error: Could not get Charuco corner coordinates from board definition.")
            return {}
    else: 
        print("Successfully got Charuco corner coordinates from board definition.")


    # --- Iterate through each camera that detected the board ---
    for (serial, detected_data) in boardLocations.items(): #for every camera which has detected the chessboard
        [corners, points2D, points3D, validPoints] = detected_data[0]
        print("length points2D", len(points2D[0]))
        print("length points3D", len(points3D[0]))
        print("length valid points", len(validPoints))
        if len(points2D[0])<5: #check if there are at least 5 points to be able to compute transformation matrix
            print(serial, " does not have enough points to have a valid depth for calculating the transformatoin")
        else:
            print("enought points, calculating transformation")
            charucoboardPoints = np.zeros((charucoboardWidth*charucoboardHeight,3), np.float32) #container for 3d coordinates of chessboard corners in global coordinates of chessboard
            charucoboardPoints[:,:2] = np.mgrid[0:charucoboardWidth, 0:charucoboardHeight].T.reshape(-1,2)
            charucoboardPoints = charucoboardPoints.transpose() * charucoboardSquareLength

            print("size charucoboardPoints", len(charucoboardPoints[0]))
            print("size validPoints", len(validPoints))
            validcharucoboardPoints = charucoboardPoints[:,validPoints].transpose()
            validobservedcharucoboardPoints = points3D[:, validPoints].transpose() #take chessboard points which have been detected by depth sensor

            charucoboardPointsCentered = validcharucoboardPoints - centroid(validcharucoboardPoints) #center global chessboard points so that reference frame is same for every camera
            observedcharucoboardPointsCentered = validobservedcharucoboardPoints - centroid(validobservedcharucoboardPoints) 

            rotationMatrix = kabsch(charucoboardPointsCentered, observedcharucoboardPointsCentered) #calculate rotation between local coordiante system and global coordinate system
            rmsdValue = kabsch_rmsd(charucoboardPointsCentered, observedcharucoboardPointsCentered) #calculate error of rotation matrix

            translationVector = centroid(validobservedcharucoboardPoints) - np.matmul(centroid(validcharucoboardPoints), rotationMatrix) #calculate translation between local and global coordinate system
            trans = -np.matmul(rotationMatrix, translationVector.transpose())

            poseMat = np.zeros((4,4)) #build 4x4 transformation matrix
            poseMat[:3,:3] = rotationMatrix
            poseMat[:3,3] = trans.flatten()
            poseMat[3,3] = 1
            print("poseMat", poseMat)
            
            devicesTransformation[serial] = [poseMat, rmsdValue]
    return devicesTransformation

    # for serial, detected_data in boardLocations.items():

    #     observedPoints3D_cam = None # Points in camera frame (X,Y,Z)
    #     correspondingObjectPoints_obj = None # Ideal points in object frame (X,Y,Z)

    #     # detectBoard returns (points2D_valid, charucoIds_valid, points3D_valid)
    #     print("detected_data", detected_data)

    #     # current_corners[device_serial] = charuco_corners, points2D, points3D, validPoints #save array for each camera of detected corners, their 2D locations, their 3D locations, and if they have a valid depth value
    #     charuco_corners, points2D, points3D, validPoints = detected_data
    #     observedPoints3D_cam = points3D_valid_cam.transpose() # Shape (N, 3)

    #     # For Charuco, we match using the IDs.
    #     if len(charucoIds_valid) != len(observedPoints3D_cam):
    #             print(f"Error for {serial}: Mismatch between charuco IDs ({len(charucoIds_valid)}) and observed points ({len(observedPoints3D_cam)}). Skipping.")
    #             continue
    #     if allCharucoCorners3D is None: # Should have been initialized earlier
    #             print(f"Error for {serial}: Charuco object points not available. Skipping.")
    #             continue
    #     try:
    #         # Select the object points corresponding to the detected charucoIds_valid
    #             correspondingObjectPoints_obj = allCharucoCorners3D[charucoIds_valid]
    #     except IndexError:
    #             max_id = len(allCharucoCorners3D) - 1
    #             offending_ids = [id_ for id_ in charucoIds_valid if id_ > max_id]
    #             print(f"Error for {serial}: Detected Charuco IDs {offending_ids} are out of bounds for the defined board (max ID: {max_id}). Skipping.")
    #             continue
    #     except Exception as e:
    #             print(f"Error matching Charuco IDs to object points for {serial}: {e}. Skipping.")
    #             continue


    #     # --- Proceed with Kabsch if points are valid ---
    #     if observedPoints3D_cam is None or correspondingObjectPoints_obj is None or len(observedPoints3D_cam) < 4:
    #         print(f"{serial}: Not enough valid corresponding points found ({len(observedPoints3D_cam) if observedPoints3D_cam is not None else 0}). Cannot calculate transformation.")
    #         continue

    #     # --- Kabsch Algorithm ---
    #     # Ensure both sets have the same number of points
    #     if observedPoints3D_cam.shape != correspondingObjectPoints_obj.shape:
    #          print(f"Error for {serial}: Shape mismatch between observed ({observedPoints3D_cam.shape}) and object ({correspondingObjectPoints_obj.shape}) points. Skipping.")
    #          continue

    #     # 1. Center the point clouds (mean position)
    #     objectCentroid_obj = centroid(correspondingObjectPoints_obj)     # Centroid in object frame
    #     observedCentroid_cam = centroid(observedPoints3D_cam)           # Centroid in camera frame

    #     objectCentered_obj = correspondingObjectPoints_obj - objectCentroid_obj
    #     observedCentered_cam = observedPoints3D_cam - observedCentroid_cam

    #     # 2. Calculate rotation matrix (R) that rotates *centered observed* points onto *centered object* points
    #     # R: camera frame -> object frame for centered points
    #     # kabsch(P, Q) finds rotation U such that P*U aligns with Q.
    #     # Here P = observedCentered_cam, Q = objectCentered_obj
    #     try:
    #          rotationMatrix_cam_to_obj = kabsch(observedCentered_cam, objectCentered_obj)
    #     except Exception as e:
    #          print(f"Error during Kabsch SVD for {serial}: {e}. Skipping.")
    #          continue

    #     # 3. Calculate RMSD *after* applying the rotation (Kabsch RMSD includes the rotation)
    #     # Calculates RMSD between Q and P*U where U = kabsch(P, Q)
    #     rmsdValue = kabsch_rmsd(observedCentered_cam, objectCentered_obj, translate=False)

    #     # 4. Calculate translation (t) from object frame origin to camera frame origin, expressed in the *camera* frame.
    #     # object_point_cam = R_obj_to_cam * object_point_obj + t_obj_to_cam_in_cam
    #     # observedCentroid_cam = R_obj_to_cam * objectCentroid_obj + t_obj_to_cam_in_cam
    #     # Need R_obj_to_cam = inv(rotationMatrix_cam_to_obj) = rotationMatrix_cam_to_obj.T
    #     R_obj_to_cam = rotationMatrix_cam_to_obj.T
    #     translation_obj_to_cam_in_cam = observedCentroid_cam - np.dot(objectCentroid_obj, R_obj_to_cam)

    #     # 5. Construct the 4x4 transformation matrix (T_obj_to_cam): transforms points from object frame to camera frame
    #     # poseMat_obj_to_cam = np.eye(4)
    #     # poseMat_obj_to_cam[:3, :3] = R_obj_to_cam
    #     # poseMat_obj_to_cam[:3, 3] = translation_obj_to_cam_in_cam

    #     # We want the inverse: T_cam_to_obj (transforms points from camera frame to object frame)
    #     poseMat_cam_to_obj = np.eye(4)
    #     poseMat_cam_to_obj[:3, :3] = rotationMatrix_cam_to_obj # R (cam -> obj)
    #     # t (cam -> obj in obj frame) = -R_cam_to_obj * t_obj_to_cam_in_cam
    #     translation_cam_to_obj_in_obj = -np.dot(rotationMatrix_cam_to_obj, translation_obj_to_cam_in_cam)
    #     poseMat_cam_to_obj[:3, 3] = translation_cam_to_obj_in_obj


    #     devicesTransformation[serial] = [poseMat_cam_to_obj, rmsdValue]
    #     print(f"  Calculated transformation for {serial} with RMSD: {rmsdValue:.6f}")

    # if not devicesTransformation:
    #     print("Warning: Pose transformation calculation failed for all devices.")

    # return devicesTransformation


if __name__ == "__main__":
    # This block provides example usage when the script is run directly
    # It requires cameras to be connected and the necessary packages installed.

    print("Running calibration script example...")

    # --- Basic Setup ---
    try:
        context = rs.context()
        if len(context.devices) == 0:
            print("No RealSense devices detected. Exiting.")
            exit()

        config = rs.config()
        # Configure streams (use common settings, adjust as needed)
        # Higher resolution might improve detection but reduce FPS
        # width, height, fps = 1280, 720, 30
        width, height, fps = 1920, 1080, 6 
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) # Use BGR8 for OpenCV

        device_manager = DeviceManager(context, config)
        print(f"DeviceManager initialized for {len(context.devices)} devices.")

    except Exception as e:
        print(f"Error during RealSense setup: {e}")
        exit()

    # --- Define Calibration Board Parameters ---

    # Example Chessboard: 9x6 internal corners, 2.5cm squares
    use_chessboard = False # Set to True to run chessboard calibration
    chessboard_params = {'height': 6, 'width': 9, 'squareSize': 0.025}

    # Example Charuco: 7x5 squares, 4cm square, 2cm marker, Dict 6x6_250
    use_charuco = True # Set to True to run Charuco calibration
    
    charuco_params = {
        'squaresX': 8,
        'squaresY': 8,
        'squareLength': 0.015,
        'markerLength': 0.011,
        'dictionary': aruco.DICT_5X5_250 # Pass the enum directly
    }
    # charuco_params = {
    #     'squaresX': 7,
    #     'squaresY': 5,
    #     'squareLength': 0.04,
    #     'markerLength': 0.02,
    #     'dictionary': aruco.DICT_6X6_250 # Pass the enum directly
    # }

    use_iterative = False # Set to True to run iterative calibration (requires >= 2 cameras)


    # --- Run Calibration ---
    all_transformations = {}

    if use_chessboard:
        print("\n--- Starting Chessboard Calibration ---")
        input("Position the chessboard visibly in all cameras and press Enter...")
        try:
            # Use the 'new' function (corrected name)
            transformations_chess = new('calibration_chessboard.cal', device_manager, 'chessboard', chessboard_params)
            if transformations_chess:
                 print("\nChessboard Calibration Results:")
                 for serial, (mat, rmsd) in transformations_chess.items():
                     print(f"  Camera {serial}: RMSD={rmsd:.6f}")
                     # print(mat)
                 all_transformations['chessboard'] = transformations_chess
            else:
                 print("Chessboard calibration failed.")
        except Exception as e:
            print(f"An error occurred during chessboard calibration: {e}")
            import traceback
            traceback.print_exc()
        finally:
             # Disable devices if you are done or moving to the next type
             # device_manager.disable_all_devices()
             pass


    if use_charuco:
        print("\n--- Starting Charuco Calibration ---")
        input("Position the Charuco board visibly in all cameras and press Enter...")
        try:
            # Use the 'new' function (corrected name)
            print("Using the charuco params: ", charuco_params)
            transformations_charuco = new('calibration_charuco.cal', device_manager, 'charuco', charuco_params)
            if transformations_charuco:
                print("\nCharuco Calibration Results:")
                for serial, (mat, rmsd) in transformations_charuco.items():
                    print(f"  Camera {serial}: RMSD={rmsd:.6f}")
                    # print(mat)
                all_transformations['charuco'] = transformations_charuco
            else:
                 print("Charuco calibration failed.")
        except Exception as e:
            print(f"An error occurred during Charuco calibration: {e}")
            import traceback
            traceback.print_exc()
        finally:
             # Disable devices if done
             # device_manager.disable_all_devices()
             pass


    if use_iterative:
        # Requires at least 2 devices
        camera_list = list(device_manager._available_devices) # Get available devices
        if len(camera_list) >= 2:
            print("\n--- Starting Iterative Charuco Calibration ---")
            print(f"Using camera order: {camera_list}")
            # Run iterative with Charuco (or change to chessboard_params if desired)
            try:
                iter_tf = newIterative('cal_iterative_charuco.cal', device_manager, camera_list, 'charuco', charuco_params)
                if iter_tf:
                    print("\nIterative Calibration Results (relative to {}):".format(camera_list[0]))
                    for serial, (mat, rmsd) in iter_tf.items():
                        print(f"  Camera {serial}:")
                        print(mat)
                    all_transformations['iterative_charuco'] = iter_tf
                else:
                     print("Iterative calibration failed.")
            except Exception as e:
                 print(f"An error occurred during iterative calibration: {e}")
                 import traceback
                 traceback.print_exc()
            finally:
                 # Disable devices if done
                 device_manager.disable_all_devices()
                 pass
        else:
            print("\nNeed at least 2 cameras for iterative calibration. Skipping.")


    # --- Cleanup ---
    try:
        print("\nDisabling RealSense devices.")
        device_manager.disable_all_devices() # Ensure devices are stopped
    except Exception as e:
        print(f"Error disabling devices: {e}")

    print("\nCalibration script example finished.")
