__doc__ = 

"""
Calibrate multiple Intel RealSense D4XX cameras to a single global coordinate system using a defined checkerboard

Distributed as a module of DynaMo: https://github.com/anderson-cu-bioastronautics/dynamo_realsense-capture
"""

##########################################################################################################################################
##                             License: Apache 2.0. See LICENSE and LICENSE.librealsense files in root directory.		                ##
##########################################################################################################################################
## This code was inspired from the librealsense box_dimensioner_multicam example, and may contain certain lines of code from this file: ##
## (https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/calibration_kabsch.py).##                                          
##########################################################################################################################################

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import pickle
import cv2.aruco as aruco

from dynamo.realsense_device_manager import DeviceManager
# from .realsense_device_manager import DeviceManager
from .calculate_rmsd import *



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

def new(fileName, deviceManager, board_type, board_parameters):
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

    boardLocations = detectBoard(deviceManager, cameraSet, board_type, board_parameters) # return locations of board corners from reference frame of each camera
    devicesTransformations = poseTransformation(boardLocations, board_type, board_parameters) # return dictionary of transformations
    with open(fileName,'wb') as f:
        pickle.dump(devicesTransformations, f)
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

    deviceManager.enable_all_devices()

    deviceTransformations = {}
    # Initialize transformations with identity matrix and zero RMSD for all cameras in the list
    for cam_serial in cameraList:
         deviceTransformations[cam_serial] = [np.eye(4), 0.0]


    relativeTransformations = {}

    time.sleep(1) #let autoexposure on cameras stabilize over one second

    for i in range(len(cameraList) - 1):
        cam1_serial = cameraList[i]
        cam2_serial = cameraList[i+1]
        cset = [cam1_serial, cam2_serial]

        fstring = f"Place the calibration board so it's visible by cameras {cam1_serial} and {cam2_serial}. Press ENTER to start calibration for this pair."
        input(fstring)

        setTransformations = {}
        # Detect board for the current pair
        boardLocations = detectBoard(deviceManager, cset, board_type, board_parameters)

        # Check if both cameras detected the board
        if len(boardLocations) == 2:
            # Calculate transformations relative to the board
            pairTransformations = poseTransformation(boardLocations, board_type, board_parameters)

            # Calculate transformation from cam1 to cam2
            # T_cam1_to_board = pairTransformations[cam1_serial][0]
            # T_cam2_to_board = pairTransformations[cam2_serial][0]
            # T_board_to_cam2 = invTrans(T_cam2_to_board)
            # T_cam1_to_cam2 = np.matmul(T_board_to_cam2, T_cam1_to_board)

            # Calculate transformation from cam2 to cam1 (more intuitive: T_world = T_cam_N * T_cam_N-1 * ... * T_cam1)
            # Let cam1 be the reference for this pair's calibration (identity)
            T_cam1_to_board = pairTransformations[cam1_serial][0]
            T_cam2_to_board = pairTransformations[cam2_serial][0]
            T_board_to_cam1 = invTrans(T_cam1_to_board)
            T_cam2_to_cam1 = np.matmul(T_board_to_cam1, T_cam2_to_board) # Transformation from cam2 frame to cam1 frame

            # Store the relative transformation T_cam(i+1)_to_cam(i)
            relativeTransformations[(cam2_serial, cam1_serial)] = T_cam2_to_cam1


            fstring = f"Cameras {cam1_serial} and {cam2_serial} calibrated relative to each other. Press ENTER to continue."
            input(fstring)
        else:
             fstring = f"Error: Could not detect the board in both cameras ({cam1_serial}, {cam2_serial}). Skipping this pair. Press ENTER to continue."
             input(fstring)
             # Store identity if detection failed for the pair to avoid breaking the chain
             relativeTransformations[(cam2_serial, cam1_serial)] = np.eye(4)


    # Chain the transformations: T_world = T_N * T_N-1 * ... * T_1 * T_0 (where T_0 is identity)
    # Assume cameraList[0] is the reference camera (world frame)
    # T_cam_i_to_world = T_cam_1_to_world * T_cam_2_to_cam1 * ... * T_cam_i_to_cam_i-1
    # Note: Stored relativeTransformations[(cam_k, cam_k-1)] is T_cam_k_to_cam_k-1

    # Set the first camera's transformation to identity
    deviceTransformations[cameraList[0]][0] = np.eye(4)

    for i in range(len(cameraList) - 1):
        cam_curr_serial = cameraList[i]
        cam_next_serial = cameraList[i+1]

        # Get the transformation from the next camera to the current camera
        T_next_to_curr = relativeTransformations.get((cam_next_serial, cam_curr_serial))

        if T_next_to_curr is not None:
            # Get the transformation from the current camera to the world
            T_curr_to_world = deviceTransformations[cam_curr_serial][0]
            # Calculate the transformation from the next camera to the world
            T_next_to_world = np.matmul(T_curr_to_world, T_next_to_curr)
            deviceTransformations[cam_next_serial][0] = T_next_to_world
            # RMSD is not directly calculated in this chained approach, keep as 0 or use pair-wise RMSD if needed
        else:
             print(f"Warning: Missing relative transformation between {cam_next_serial} and {cam_curr_serial}. Setting {cam_next_serial}'s transform based on {cam_curr_serial}'s current transform.")
             deviceTransformations[cam_next_serial][0] = deviceTransformations[cam_curr_serial][0] # Or handle error differently


    print("Final calculated transformations relative to {}:".format(cameraList[0]))
    for cam_serial in cameraList:
        print(f"  {cam_serial}:")
        print(deviceTransformations[cam_serial][0])

    with open(fileName,'wb') as f:
        pickle.dump(deviceTransformations, f)
    return deviceTransformations


def detectBoard(deviceManager, cameraSet, board_type, board_parameters):
    """
    Detects calibration board (chessboard or Charuco) corners/markers for each specified camera.

    Parameters
    ----------
    deviceManager : DeviceManager object
        realsense_device_manager object which manages connections to all cameras

    cameraSet : list
        list of camera serial numbers to detect the board in.

    board_type : str
        Type of calibration board used ('chessboard' or 'charuco').

    board_parameters : dict
        Dictionary containing parameters for the specified board type (see 'new' function docstring for details).

    Returns
    -------
    devicesBoardLocations : dict
        dictionary with keys of camera's serial number holding detected corners/markers,
        local 2D and 3D coordinates of detected corners, and their valid depth points.
        Format for chessboard: {serial: (corners, points2D, points3D, validPoints)}
        Format for charuco: {serial: (charucoCorners, charucoIds, points3D, validPoints)}

    """
    boardDeviceCount = 0
    devicesBoardLocations = {}

    if board_type == 'charuco':
        aruco_dict = aruco.getPredefinedDictionary(board_parameters['dictionary'])
        charuco_board = aruco.CharucoBoard(
            (board_parameters['squaresX'], board_parameters['squaresY']),
            board_parameters['squareLength'],
            board_parameters['markerLength'],
            aruco_dict)
        aruco_params = aruco.DetectorParameters()
        # Add refinement parameters if needed
        # aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # aruco_params.cornerRefinementWinSize = 5
        # aruco_params.cornerRefinementMaxIterations = 30
        # aruco_params.cornerRefinementMinAccuracy = 0.01

    print(f"Attempting to detect {board_type} board in cameras: {cameraSet}")
    attempts = 0
    max_attempts = 10 # Try multiple times to ensure detection

    while len(devicesBoardLocations) < len(cameraSet) and attempts < max_attempts:
        cameraFrames = deviceManager.poll_frames()
        if not cameraFrames:
            print("Warning: Failed to poll frames.")
            time.sleep(0.1)
            attempts +=1
            continue

        devicesIntrinsics = deviceManager.get_device_intrinsics(cameraFrames)
        current_detected_in_frame = {}

        for device_serial in cameraSet:
            if device_serial in devicesBoardLocations: # Skip if already detected for this camera
                 continue
            if device_serial not in cameraFrames:
                 print(f"Warning: No frame received for camera {device_serial} in this poll.")
                 continue

            frames = cameraFrames[device_serial]
            if not frames:
                print(f"Warning: Empty frameset for camera {device_serial}.")
                continue

            align = rs.align(rs.stream.depth) # align the color sensor to the depth sensor
            try:
                 alignedFrames = align.process(frames)
                 if not alignedFrames: continue
                 color_frame = alignedFrames.get_color_frame()
                 depth_frame = alignedFrames.get_depth_frame()
                 if not color_frame or not depth_frame: continue
            except Exception as e:
                print(f"Error processing frames for {device_serial}: {e}")
                continue

            colorImage = np.asanyarray(color_frame.get_data())
            bwImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY) # convert color image to B&W

            try:
                depthIntrinsics = devicesIntrinsics[device_serial][rs.stream.depth]
            except KeyError:
                 print(f"Warning: Could not get depth intrinsics for {device_serial}")
                 continue


            boardFound = False
            points2D = None
            points3D = None
            validPoints = None
            ids = None # For Charuco

            if board_type == 'chessboard':
                height = board_parameters['height']
                width = board_parameters['width']
                boardFound, corners = cv2.findChessboardCorners(bwImage, (width, height))

                if boardFound:
                    print(f"{device_serial} sees the chessboard!")
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    points2D = cv2.cornerSubPix(bwImage, corners, (11,11), (-1,-1), criteria)

                    # Draw for verification
                    draw_img = cv2.cvtColor(bwImage, cv2.COLOR_GRAY2BGR) # Create color image to draw corners
                    cv2.drawChessboardCorners(draw_img, (width, height), points2D, boardFound)
                    cv2.imshow(device_serial, draw_img)
                    cv2.waitKey(50)

                    points2D_flat = points2D.reshape(-1, 2) # Flatten for easier processing
                    points3D = np.zeros((3, len(points2D_flat)))
                    validPoints = [False] * len(points2D_flat)

                    for index, corner in enumerate(points2D_flat):
                        depth = depth_frame.get_distance(round(corner[0]), round(corner[1]))
                        if depth > 0: # Check for valid depth (non-zero)
                            validPoints[index] = True
                            points3D[0, index] = (corner[0] - depthIntrinsics.ppx) / depthIntrinsics.fx * depth
                            points3D[1, index] = (corner[1] - depthIntrinsics.ppy) / depthIntrinsics.fy * depth
                            points3D[2, index] = depth
                        # else: Point remains [0,0,0] and validPoints[index] remains False

                    # Filter out invalid points before storing
                    valid_indices = [i for i, valid in enumerate(validPoints) if valid]
                    if len(valid_indices) > 4 : # Need at least 5 valid points
                        points2D_valid = points2D_flat[valid_indices] # Use the flat version here
                        points3D_valid = points3D[:, valid_indices]
                        current_detected_in_frame[device_serial] = (points2D_valid, points3D_valid) # Store valid 2D/3D points directly
                        print(f"    Detected {len(points2D_valid)} valid chessboard corners.")
                    else:
                         boardFound = False # Not enough valid points
                         print(f"    Detected chessboard but found only {len(valid_indices)} valid points (need > 4).")


            elif board_type == 'charuco':
                # Detect ArUco markers first
                corners, ids, rejectedImgPoints = aruco.detectMarkers(bwImage, aruco_dict, parameters=aruco_params)

                if ids is not None and len(ids) > 3: # Need enough markers to interpolate Charuco corners
                    # Interpolate Charuco corners
                    retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, bwImage, charuco_board)

                    if retval and charucoCorners is not None and len(charucoCorners) > 3:
                        boardFound = True
                        print(f"{device_serial} sees the Charuco board ({len(charucoCorners)} corners)!")
                        points2D = charucoCorners # These are the refined Charuco corners

                        # Draw for verification
                        draw_img = cv2.cvtColor(bwImage, cv2.COLOR_GRAY2BGR)
                        aruco.drawDetectedMarkers(draw_img, corners, ids)
                        aruco.drawDetectedCornersCharuco(draw_img, charucoCorners, charucoIds, (255, 0, 0))
                        cv2.imshow(device_serial, draw_img)
                        cv2.waitKey(50)

                        points2D_flat = points2D.reshape(-1, 2) # Flatten for easier processing
                        points3D = np.zeros((3, len(points2D_flat)))
                        validPoints = [False] * len(points2D_flat)

                        for index, corner in enumerate(points2D_flat):
                            depth = depth_frame.get_distance(round(corner[0]), round(corner[1]))
                            if depth > 0: # Check for valid depth (non-zero)
                                validPoints[index] = True
                                points3D[0, index] = (corner[0] - depthIntrinsics.ppx) / depthIntrinsics.fx * depth
                                points3D[1, index] = (corner[1] - depthIntrinsics.ppy) / depthIntrinsics.fy * depth
                                points3D[2, index] = depth
                            # else: Point remains [0,0,0] and validPoints[index] remains False

                        # Filter out invalid points before storing
                        valid_indices = [i for i, valid in enumerate(validPoints) if valid]
                        if len(valid_indices) > 3: # Need at least 4 valid Charuco corners
                            points2D_valid = points2D_flat[valid_indices]
                            points3D_valid = points3D[:, valid_indices]
                            charucoIds_valid = charucoIds.flatten()[valid_indices] # Get corresponding IDs

                            current_detected_in_frame[device_serial] = (points2D_valid, charucoIds_valid, points3D_valid) # Store valid 2D points, IDs, and 3D points
                            print(f"    Detected {len(points2D_valid)} valid Charuco corners.")
                        else:
                             boardFound = False # Not enough valid points
                             print(f"    Detected Charuco board but found only {len(valid_indices)} valid points (need > 3).")
                    else:
                        print(f"{device_serial}: Detected markers but failed to interpolate enough Charuco corners.")

                else:
                    print(f"{device_serial}: Could not detect enough ArUco markers.")
                    cv2.imshow(device_serial, bwImage) # Show image even if nothing detected
                    cv2.waitKey(50)


            if not boardFound:
                 # If board not found for this device in this frame, clear previous successful detections for this device to retry
                 if device_serial in devicesBoardLocations:
                      del devicesBoardLocations[device_serial]
                      print(f"Cleared previous detection for {device_serial}, retrying.")
                 # Reset the overall detection dictionary if any camera fails in the current attempt for synchronous detection
                 # This forces all cameras to see the board simultaneously
                 # devicesBoardLocations.clear() # Uncomment this line for strict simultaneous detection
                 # print(f"{device_serial} cannot detect the {board_type} board! Resetting detections for this attempt.")
                 # break # Break from inner loop (devices) and retry polling frames

        # Add successfully detected boards from this frame to the main dictionary
        devicesBoardLocations.update(current_detected_in_frame)

        if len(devicesBoardLocations) == len(cameraSet):
            print("Successfully detected board in all specified cameras.")
            break # Exit while loop

        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}. Detected in {len(devicesBoardLocations)}/{len(cameraSet)} cameras. Retrying...")
        time.sleep(0.5) # Wait before next attempt

    cv2.destroyAllWindows()
    if len(devicesBoardLocations) < len(cameraSet):
        print(f"Error: Failed to detect the {board_type} board in all required cameras ({list(set(cameraSet) - set(devicesBoardLocations.keys()))}) after {max_attempts} attempts.")
        return {} # Return empty if not all detected

    return devicesBoardLocations


def poseTransformation(boardLocations, board_type, board_parameters):
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
        4x4 transformation matrix and root-mean squared error: {serial: [poseMat, rmsdValue]}

    """
    devicesTransformation = {}

    # Prepare object points (ideal 3D coordinates of the board corners)
    objectPoints = None
    if board_type == 'chessboard':
        height = board_parameters['height']
        width = board_parameters['width']
        squareSize = board_parameters['squareSize']
        objectPoints = np.zeros((width * height, 3), np.float32)
        objectPoints[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        objectPoints *= squareSize
        # objectPoints = objectPoints.transpose() # Keep as (N, 3) for Kabsch input

    elif board_type == 'charuco':
        aruco_dict = aruco.getPredefinedDictionary(board_parameters['dictionary'])
        charuco_board = aruco.CharucoBoard(
            (board_parameters['squaresX'], board_parameters['squaresY']),
            board_parameters['squareLength'],
            board_parameters['markerLength'],
            aruco_dict)
        # Get the 3D coordinates of all Charuco corners from the board definition
        # Note: charuco_board.getChessboardCorners() returns a (N, 3) array
        allCharucoCorners3D = charuco_board.getChessboardCorners()
        if allCharucoCorners3D is None or len(allCharucoCorners3D) == 0:
             print("Error: Could not get Charuco corner coordinates from board definition.")
             return {}


    for serial, detected_data in boardLocations.items():

        observedPoints3D = None
        correspondingObjectPoints = None

        if board_type == 'chessboard':
            points2D_valid, points3D_valid = detected_data
            observedPoints3D = points3D_valid.transpose() # Shape (N, 3) for Kabsch
            # For chessboard, the order is fixed, so we can use the precomputed objectPoints
            # We need to ensure we only use object points corresponding to the validly detected points3D_valid
            # This requires knowing the original indices, which detectBoard currently discards.
            # Modification needed in detectBoard: return original indices or map points directly.
            # --- ASSUMPTION: For now, assume detectBoard returns all points in order if successful ---
            # This assumption is WRONG based on current detectBoard filtering. Needs fixing.
            # Temporary workaround: Assume all points were detected if entry exists (Risky!)
            if len(observedPoints3D) != len(objectPoints):
                 print(f"Warning for {serial}: Mismatch between observed ({len(observedPoints3D)}) and expected ({len(objectPoints)}) chessboard points. Skipping calibration for this camera.")
                 continue
            correspondingObjectPoints = objectPoints

        elif board_type == 'charuco':
            points2D_valid, charucoIds_valid, points3D_valid = detected_data
            observedPoints3D = points3D_valid.transpose() # Shape (N, 3) for Kabsch
            # For Charuco, we need to match observed points to object points using the IDs
            if len(charucoIds_valid) != len(observedPoints3D):
                 print(f"Error for {serial}: Mismatch between charuco IDs ({len(charucoIds_valid)}) and observed points ({len(observedPoints3D)}).")
                 continue
            try:
                # Select the object points corresponding to the detected charucoIds_valid
                 correspondingObjectPoints = allCharucoCorners3D[charucoIds_valid]
            except IndexError:
                 print(f"Error for {serial}: Detected Charuco IDs are out of bounds for the defined board. Max ID on board: {len(allCharucoCorners3D)-1}, detected IDs: {charucoIds_valid}")
                 continue

        if observedPoints3D is None or correspondingObjectPoints is None or len(observedPoints3D) < 4:
            print(f"{serial}: Not enough valid corresponding points found ({len(observedPoints3D) if observedPoints3D is not None else 0}). Cannot calculate transformation.")
            continue

        # --- Kabsch Algorithm ---
        # 1. Center the point clouds
        objectCentroid = centroid(correspondingObjectPoints)
        observedCentroid = centroid(observedPoints3D)
        objectCentered = correspondingObjectPoints - objectCentroid
        observedCentered = observedPoints3D - observedCentroid

        # 2. Calculate rotation matrix
        rotationMatrix = kabsch(observedCentered, objectCentered) # Rotate observed onto object this time (aligns with standard solvePnP where object is reference)

        # 3. Calculate RMSD
        rmsdValue = kabsch_rmsd(objectCentered, observedCentered, translate=False) # Use centered points, rotation applied within kabsch_rmsd if needed

        # 4. Calculate translation (T_obj_to_cam = T_obs_centroid - R * T_obj_centroid)
        # translationVector = observedCentroid - np.dot(objectCentroid, rotationMatrix.T) # If R rotates obj to obs
        translationVector = observedCentroid - np.dot(objectCentroid, rotationMatrix) # If R rotates obs to obj

        # 5. Construct the 4x4 transformation matrix (transforms points from object frame to camera frame)
        poseMat = np.eye(4)
        poseMat[:3, :3] = rotationMatrix # R (obs -> obj)
        poseMat[:3, 3] = translationVector # T (obj -> cam)

        # The required matrix usually transforms points from the camera frame to the world (object) frame.
        # Let's compute the inverse transformation. T_cam_to_obj = inv(T_obj_to_cam)
        poseMat_cam_to_obj = invTrans(poseMat)


        devicesTransformation[serial] = [poseMat_cam_to_obj, rmsdValue]
        print(f"  Calculated transformation for {serial} with RMSD: {rmsdValue:.4f}")

    return devicesTransformation

if __name__ == "__main__":
    print("This script needs to be called with appropriate arguments for calibration.")
    print("Example usage within another script:")
    print("""
import pyrealsense2 as rs
from dynamo.realsense_device_manager import DeviceManager
import dynamo.calibration as calib
import cv2.aruco as aruco

# Setup DeviceManager
context = rs.context()
config = rs.config()
# Configure streams (e.g., depth and color)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Use BGR8 for OpenCV compatibility
device_manager = DeviceManager(context, config)

# --- Chessboard Example ---
# chessboard_params = {'height': 6, 'width': 9, 'squareSize': 0.025}
# try:
#     print("Starting Chessboard Calibration...")
#     transformations_chess = calib.new('calibration_chessboard.cal', device_manager, 'chessboard', chessboard_params)
#     print("\\nChessboard Calibration Results:")
#     for serial, (mat, rmsd) in transformations_chess.items():
#         print(f"  Camera {serial}: RMSD={rmsd:.4f}")
#         # print(mat)
# finally:
#      device_manager.disable_all_devices() # Important to stop streams

# --- Charuco Example ---
charuco_params = {'squaresX': 5, 'squaresY': 7, 'squareLength': 0.04, 'markerLength': 0.02, 'dictionary': aruco.DICT_6X6_250}
try:
    print("\\nStarting Charuco Calibration...")
    transformations_charuco = calib.new('calibration_charuco.cal', device_manager, 'charuco', charuco_params)
    print("\\nCharuco Calibration Results:")
    if transformations_charuco:
        for serial, (mat, rmsd) in transformations_charuco.items():
            print(f"  Camera {serial}: RMSD={rmsd:.4f}")
            # print(mat)
    else:
         print("Charuco calibration failed.")
finally:
     device_manager.disable_all_devices() # Important to stop streams

# --- Iterative Example ---
# camera_list = list(device_manager._available_devices) # Get available devices AFTER enabling streams
# if len(camera_list) >= 2:
#      try:
#         print("\\nStarting Iterative Charuco Calibration...")
#         iter_tf = calib.newIterative('cal_iterative_charuco.cal', device_manager, camera_list, 'charuco', charuco_params)
#         print("\\nIterative Calibration Results (relative to {}):".format(camera_list[0]))
#         for serial, (mat, rmsd) in iter_tf.items():
#             print(f"  Camera {serial}:")
#             print(mat)
#      finally:
#         device_manager.disable_all_devices()
# else:
#     print("\\nNeed at least 2 cameras for iterative calibration.")

""")
