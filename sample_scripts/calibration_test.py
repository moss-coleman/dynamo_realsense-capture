import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco # <--- Added Import
# import numpy as np
# import time

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
    width, height, fps = 640, 480, 30
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
# charuco_params = {'squaresX': 8, 'squaresY': 8, 'squareLength': 0.015, 'markerLength': 0.011, 'dictionary': aruco.DICT_5X5_250}
charuco_params = {
    'squaresX': 8,
    'squaresY': 8,
    'squareLength': 0.015,
    'markerLength': 0.011,
    'dictionary': aruco.DICT_5X5_250 # Pass the enum directly
}

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
                 # device_manager.disable_all_devices()
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


# import pyrealsense2 as rs
# from dynamo.realsense_device_manager import DeviceManager
# import dynamo.calibration_robot as calib
# # import dynamo.calibration_robot as calib
# # from ..dynamo.calibration_robot import new 
# # from dynamo import calibration_robot as calib
# import cv2.aruco as aruco
#
# # Setup DeviceManager
# context = rs.context()
# config = rs.config()
# # Configure streams (e.g., depth and color)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Use BGR8 for OpenCV compatibility
# device_manager = DeviceManager(context, config)
#
# # --- Chessboard Example ---
# # chessboard_params = {'height': 6, 'width': 9, 'squareSize': 0.025}
# # try:
# #     print("Starting Chessboard Calibration...")
# #     transformations_chess = new('calibration_chessboard.cal', device_manager, 'chessboard', chessboard_params)
# #     print("\\nChessboard Calibration Results:")
# #     for serial, (mat, rmsd) in transformations_chess.items():
# #         print(f"  Camera {serial}: RMSD={rmsd:.4f}")
# #         # print(mat)
# # finally:
# #      device_manager.disable_all_devices() # Important to stop streams
#
# # --- Charuco Example ---
# charuco_params = {'squaresX': 8, 'squaresY': 8, 'squareLength': 0.015, 'markerLength': 0.011, 'dictionary': aruco.DICT_5X5_250}
# try:
#     print("\\nStarting Charuco Calibration...")
#     transformations_charuco = calib.new('calibration_charuco.cal', device_manager, 'charuco', charuco_params)
#     print("\\nCharuco Calibration Results:")
#     if transformations_charuco:
#         for serial, (mat, rmsd) in transformations_charuco.items():
#             print(f"  Camera {serial}: RMSD={rmsd:.4f}")
#             # print(mat)
#     else:
#          print("Charuco calibration failed.")
# finally:
#      device_manager.disable_all_devices() # Important to stop streams
#
# # --- Iterative Example ---
# # camera_list = list(device_manager._available_devices) # Get available devices AFTER enabling streams
# # if len(camera_list) >= 2:
# #      try:
# #         print("\\nStarting Iterative Charuco Calibration...")
# #         iter_tf = newIterative('cal_iterative_charuco.cal', device_manager, camera_list, 'charuco', charuco_params)
# #         print("\\nIterative Calibration Results (relative to {}):".format(camera_list[0]))
# #         for serial, (mat, rmsd) in iter_tf.items():
# #             print(f"  Camera {serial}:")
# #             print(mat)
# #      finally:
# #         device_manager.disable_all_devices()
# # else:
# #     print("\\nNeed at least 2 cameras for iterative calibration.")
