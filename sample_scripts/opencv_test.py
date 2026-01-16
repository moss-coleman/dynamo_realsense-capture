import cv2
import sys

print(f"Python Executable: {sys.executable}")
print(f"CV2 Version: {cv2.__version__}")

try:
    # Check if the aruco submodule exists
    print(f"Accessing cv2.aruco: {cv2.aruco}")

    # Check specifically for interpolateCornersCharuco
    if hasattr(cv2.aruco, 'interpolateCornersCharuco'):
        print("cv2.aruco.interpolateCornersCharuco function found.")
    else:
        print("cv2.aruco module found, but 'interpolateCornersCharuco' attribute is MISSING.")
        print("This strongly suggests an incomplete or outdated 'opencv-contrib-python' installation.")
        print("Available attributes in cv2.aruco:", dir(cv2.aruco)) # List available functions

    # Also re-check for detectMarkers just in case
    if hasattr(cv2.aruco, 'detectMarkers'):
        print("cv2.aruco.detectMarkers function found.")
    else:
        print("cv2.aruco.detectMarkers function is also MISSING.")


except AttributeError:
    print("AttributeError: cv2.aruco module not found.")
    print("This usually means 'opencv-contrib-python' is not installed or not installed correctly.")
    print("Available attributes in cv2:", dir(cv2))
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# import cv2
# import sys
#
# print(f"Python Executable: {sys.executable}")
# print(f"CV2 Version: {cv2.__version__}")
#
# try:
#     # Check if the aruco submodule exists
#     print(f"Accessing cv2.aruco: {cv2.aruco}")
#
#     # Check if detectMarkers exists within cv2.aruco
#     if hasattr(cv2.aruco, 'detectMarkers'):
#         print("cv2.aruco.detectMarkers function found.")
#     else:
#         print("cv2.aruco module found, but 'detectMarkers' attribute is MISSING.")
#         print("Available attributes in cv2.aruco:", dir(cv2.aruco))
#         print("Available attributes in cv2.aruco.ArucoDetector:", dir(cv2.aruco.ArucoDetector))
#
# except AttributeError:
#     print("AttributeError: cv2.aruco module not found.")
#     print("This usually means 'opencv-contrib-python' is not installed or not installed correctly.")
#     print("Available attributes in cv2:", dir(cv2))
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
