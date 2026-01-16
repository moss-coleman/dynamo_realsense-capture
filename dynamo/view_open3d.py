##########################################################################################################################################
##                                               License: Apache 2.0. See LICENSE file in root directory.                               ##
##########################################################################################################################################
"""
View captured 3D scans from DynaMo using Open3D.
Point clouds are transformed to the common world frame, and the
pose of each camera is visualized as a coordinate frame.

Distributed as a module of DynaMo: https://github.com/anderson-cu-bioastronautics/dynamo_realsense-capture
"""
import pickle
import time
import numpy as np
import cv2
import argparse
import os
import signal
import sys
import open3d as o3d # Use Open3D

# Global flag to handle SIGINT gracefully within the visualizer loop
keep_running = True

def signalHandler(signal, frame):
    """Sets a flag to terminate the loop gracefully."""
    global keep_running
    print("Signal received, shutting down viewer...")
    keep_running = False


def depthFrametoPC(deviceData):
    """
    Function which takes saved depth and color/infrared 2D frames and converts
    them into 3D points and corresponding colors, transforming them into the
    common world coordinate frame using poseMat.

    Parameters
    ----------
    deviceData : dict
        Dictionary with depth frame, infrared frame, depth sensor intrinsics,
        and transformation matrix (poseMat).

    Returns
    -------
    pointsTransformed : (N,3) numpy array
        Array of calculated 3D point coordinates in the world frame.
    colors_rgb : (N,3) numpy array
        Array of corresponding RGB colors (0-255 range). Returns None if no
        color/infrared data is available.
    """
    depth = deviceData.get('depth')
    if depth is None:
        # print("Warning: No depth data found in deviceData.") # Reduce verbosity
        return None, None

    rgb_image = None
    if 'color' in deviceData:
        rgb_image = deviceData['color']
    elif 'infrared' in deviceData:
        infrared = deviceData['infrared']
        rgb_image = cv2.cvtColor(np.asanyarray(infrared), cv2.COLOR_GRAY2RGB)

    cameraIntrinsics = deviceData.get('intrinsics')
    poseMat = deviceData.get('poseMat') # T_cam_to_world

    #TODO Add transform here to account for the realsense camera pose vs. the given
    # orrr maybe apply the same transform in the generate transform file 
    
    # Need both intrinsics and pose matrix to generate points in world frame
    if cameraIntrinsics is None or poseMat is None:
        # print("Warning: Missing intrinsics or pose matrix in deviceData.") # Reduce verbosity
        return None, None

    try:
        height, width = depth.shape
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            cameraIntrinsics['fx'], cameraIntrinsics['fy'],
            cameraIntrinsics['ppx'], cameraIntrinsics['ppy']
        )
        o3d_depth = o3d.geometry.Image(depth.astype(np.uint16))

        o3d_rgb = None
        if rgb_image is not None:
             if rgb_image.dtype != np.uint8:
                 if np.issubdtype(rgb_image.dtype, np.floating):
                      rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
                 else:
                      rgb_image = rgb_image.astype(np.uint8)
             o3d_rgb = o3d.geometry.Image(rgb_image)

        if o3d_rgb is not None:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False
            )
        else:
             rgbd_image = o3d.geometry.Image(depth.astype(np.uint16))

        pcd_camera_frame = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d_intrinsics
        ) if o3d_rgb is not None else o3d.geometry.PointCloud.create_from_depth_image(
             o3d_depth, o3d_intrinsics, depth_scale=1000.0, depth_trunc=4.0
            )

        # !!! APPLY TRANSFORMATION TO WORLD FRAME !!!
        # poseMat should be T_cam_to_world
        pcd_camera_frame.transform(poseMat)

        pointsTransformed = np.asarray(pcd_camera_frame.points)
        colors_rgb = None
        if o3d_rgb is not None:
              if pcd_camera_frame.has_colors():
                   colors_rgb_float = np.asarray(pcd_camera_frame.colors)
                   colors_rgb = (colors_rgb_float * 255).astype(np.uint8)
              else:
                   colors_rgb = np.zeros_like(pointsTransformed, dtype=np.uint8)
        else:
            colors_rgb = np.full_like(pointsTransformed, 128, dtype=np.uint8)

    except Exception as e:
        print(f"Error processing frame to point cloud: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug
        return None, None

    return pointsTransformed, colors_rgb


def getPointCloud(frame):
    """
    Function which allows for the conversion of frame data from multiple cameras
    into a single combined point cloud (points and colors) in the world frame.

    Parameters
    ----------
    frame : dict
        Dictionary with keys as serial numbers of each connected camera,
        containing frames data from each camera (as loaded from pickle).

    Returns
    -------
    all_points : (N,3) numpy array or None
        Combined array of 3D point coordinates (world frame) from all cameras.
    all_colors : (N,3) numpy array or None
        Combined array of corresponding RGB colors (0-255) from all cameras.
    """
    all_points_list = []
    all_colors_list = []
    has_color_data = False

    for device_serial, deviceData in frame.items():
        points, colors = depthFrametoPC(deviceData)
        if points is not None:
            all_points_list.append(points)
            if colors is not None:
                all_colors_list.append(colors)
                if np.any(colors > 0) and not np.all(colors == 128):
                     has_color_data = True
            else:
                placeholder_colors = np.full_like(points, 128, dtype=np.uint8) # Grey
                all_colors_list.append(placeholder_colors)
        # else: # Reduce verbosity
             # print(f"Warning: Failed to process points for device {device_serial}")

    if not all_points_list:
        return None, None

    all_points = np.concatenate(all_points_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)

    return all_points, all_colors

def viewSinglePC(folderDirectory, downsample_voxel_size=0.005, camera_frame_size=0.1):
    global keep_running
    keep_running = True
    signal.signal(signal.SIGINT, signalHandler)

    print("Initializing Open3D Visualizer...")
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis = o3d.visualization.VisualizerWithEditing()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='DynaMo Point Cloud Viewer (World Frame + Camera Poses)')

    # Add world coordinate frame at origin for reference
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(world_frame)

    pcd = o3d.geometry.PointCloud() # For combined point cloud
    camera_frames_geoms = {} # Store camera frame geometries {serial: mesh}
    is_first_frame = True

    # def close_window_callback(vis):
    #     global keep_running
    #     print("Escape key pressed, closing window.")
    #     # keep_running = False #orig
    #     keep_running = True 
    #     return False
    # vis.register_key_callback(256, close_window_callback)


    i = 0
    try:
        file_list = sorted([f for f in os.listdir(folderDirectory) if f.endswith('.pickle')])
    except FileNotFoundError:
        print(f"Error: Folder '{folderDirectory}' not found.")
        return
    num_files = len(file_list)
    if num_files == 0:
        print(f"No .pickle files found in '{folderDirectory}'.")
        return
    print(f"Found {num_files} pickle files in '{folderDirectory}'.")

    # frame_step = 1 if full_fps else 10
    frame_step = 1

    while keep_running and i < num_files:
        start_time = time.time()
        fname = os.path.join(folderDirectory, file_list[i])

        try:
            with open(fname, 'rb') as file:
                frame_data = pickle.load(file) # This is a dict {serial: deviceData}
        except FileNotFoundError:
            print(f"Error: File not found {fname}")
            i += frame_step
            continue
        except Exception as e:
            print(f"Error loading file {fname}: {e}")
            i += frame_step
            continue

        # Get combined point cloud in world frame
        combined_points, combined_colors = getPointCloud(frame_data)

        # --- Update Combined Point Cloud Geometry ---
        update_pcd_geom = False
        if combined_points is not None:
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            if combined_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(combined_colors.astype(np.float64) / 255.0)
            else:
                pcd.colors = o3d.utility.Vector3dVector()

            if downsample_voxel_size and downsample_voxel_size > 0:
                 pcd_downsampled = pcd.voxel_down_sample(downsample_voxel_size)
                 pcd_to_update = pcd_downsampled
            else:
                 pcd_to_update = pcd

            update_pcd_geom = True # Flag that we have valid points to update/add

            if is_first_frame:
                vis.add_geometry(pcd_to_update, reset_bounding_box=True)
            # else: # Update handled below after camera frames


        # --- Update Camera Frame Geometries ---
        active_serials_in_frame = set(frame_data.keys())
        serials_to_remove = set(camera_frames_geoms.keys()) - active_serials_in_frame

        # Remove geometries for cameras not present in this frame (if needed)
        for serial in serials_to_remove:
             if serial in camera_frames_geoms:
                  vis.remove_geometry(camera_frames_geoms[serial], reset_bounding_box=False)
                  del camera_frames_geoms[serial]

        # Add or update geometries for cameras present in this frame
        for serial, deviceData in frame_data.items():
            poseMat = deviceData.get('poseMat') # T_cam_to_world
            if poseMat is None:
                continue # Skip if no pose matrix for this camera

            # Create a new camera frame mesh if it doesn't exist
            if serial not in camera_frames_geoms:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=camera_frame_size, origin=[0, 0, 0]
                )
                cam_frame.transform(poseMat) # Transform to camera's pose in world
                camera_frames_geoms[serial] = cam_frame
                vis.add_geometry(camera_frames_geoms[serial], reset_bounding_box=False)
            else:
                # Update existing camera frame geometry
                # Create a temporary frame, transform it, then copy to existing geometry
                # This avoids issues with repeated transforms if pose changes
                temp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=camera_frame_size, origin=[0, 0, 0]
                )
                temp_frame.transform(poseMat)
                # Update vertices and triangles (more robust than just calling transform again)
                camera_frames_geoms[serial].vertices = temp_frame.vertices
                camera_frames_geoms[serial].triangles = temp_frame.triangles
                camera_frames_geoms[serial].compute_vertex_normals()
                vis.update_geometry(camera_frames_geoms[serial])


        # --- Final Update for Combined Point Cloud ---
        if not is_first_frame and update_pcd_geom:
             vis.update_geometry(pcd_to_update) # Update combined cloud

        if update_pcd_geom: # Mark first frame as done only if we added geometry
            is_first_frame = False

        # --- Poll Events ---
        if not vis.poll_events():
            keep_running = False
            break
        vis.update_renderer()

        # --- Control Playback Speed ---
        elapsed_time = time.time() - start_time
        target_frame_time = 1.0 / 30.0 # Target 30 FPS
        sleep_time = target_frame_time - elapsed_time 
        sleep_time = 0.5 
        if sleep_time > 0:
             time.sleep(sleep_time)

        i += frame_step

    print("Closing viewer window.")
    vis.destroy_window()
    

def viewPointClouds(folderDirectory, full_fps, downsample_voxel_size=0.005, camera_frame_size=0.1):
    """
    Function which allows for the viewing of pointClouds using Open3D.
    Displays combined point clouds (world frame) and visualizes each camera's pose.

    Parameters
    ----------
    folderDirectory : str
        Directory containing .pickle files of saved frames from capture.
    full_fps : bool
        True if you wish to view every frame, False to view every 10th frame.
    downsample_voxel_size : float, optional
        Voxel size for downsampling the point cloud. Set to 0 to disable. Default is 0.005.
    camera_frame_size : float, optional
        Size of the coordinate frame mesh used to visualize camera poses. Default is 0.1.
    """
    global keep_running
    keep_running = True
    signal.signal(signal.SIGINT, signalHandler)

    print("Initializing Open3D Visualizer...")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='DynaMo Point Cloud Viewer (World Frame + Camera Poses)')

    # Add world coordinate frame at origin for reference
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(world_frame)

    pcd = o3d.geometry.PointCloud() # For combined point cloud
    camera_frames_geoms = {} # Store camera frame geometries {serial: mesh}
    is_first_frame = True

    def close_window_callback(vis):
        global keep_running
        print("Escape key pressed, closing window.")
        # keep_running = False #orig
        keep_running = True #orig
        return False
    vis.register_key_callback(256, close_window_callback)


    i = 0
    try:
        file_list = sorted([f for f in os.listdir(folderDirectory) if f.endswith('.pickle')])
    except FileNotFoundError:
        print(f"Error: Folder '{folderDirectory}' not found.")
        return
    num_files = len(file_list)
    if num_files == 0:
        print(f"No .pickle files found in '{folderDirectory}'.")
        return
    print(f"Found {num_files} pickle files in '{folderDirectory}'.")

    frame_step = 1 if full_fps else 10

    while keep_running and i < num_files:
        start_time = time.time()
        fname = os.path.join(folderDirectory, file_list[i])

        try:
            with open(fname, 'rb') as file:
                frame_data = pickle.load(file) # This is a dict {serial: deviceData}
        except FileNotFoundError:
            print(f"Error: File not found {fname}")
            i += frame_step
            continue
        except Exception as e:
            print(f"Error loading file {fname}: {e}")
            i += frame_step
            continue

        # Get combined point cloud in world frame
        combined_points, combined_colors = getPointCloud(frame_data)

        # --- Update Combined Point Cloud Geometry ---
        update_pcd_geom = False
        if combined_points is not None:
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            if combined_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(combined_colors.astype(np.float64) / 255.0)
            else:
                pcd.colors = o3d.utility.Vector3dVector()

            if downsample_voxel_size and downsample_voxel_size > 0:
                 pcd_downsampled = pcd.voxel_down_sample(downsample_voxel_size)
                 pcd_to_update = pcd_downsampled
            else:
                 pcd_to_update = pcd

            update_pcd_geom = True # Flag that we have valid points to update/add

            if is_first_frame:
                vis.add_geometry(pcd_to_update, reset_bounding_box=True)
            # else: # Update handled below after camera frames


        # --- Update Camera Frame Geometries ---
        active_serials_in_frame = set(frame_data.keys())
        serials_to_remove = set(camera_frames_geoms.keys()) - active_serials_in_frame

        # Remove geometries for cameras not present in this frame (if needed)
        for serial in serials_to_remove:
             if serial in camera_frames_geoms:
                  vis.remove_geometry(camera_frames_geoms[serial], reset_bounding_box=False)
                  del camera_frames_geoms[serial]

        # Add or update geometries for cameras present in this frame
        for serial, deviceData in frame_data.items():
            poseMat = deviceData.get('poseMat') # T_cam_to_world
            if poseMat is None:
                continue # Skip if no pose matrix for this camera

            # Create a new camera frame mesh if it doesn't exist
            if serial not in camera_frames_geoms:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=camera_frame_size, origin=[0, 0, 0]
                )
                cam_frame.transform(poseMat) # Transform to camera's pose in world
                camera_frames_geoms[serial] = cam_frame
                vis.add_geometry(camera_frames_geoms[serial], reset_bounding_box=False)
            else:
                # Update existing camera frame geometry
                # Create a temporary frame, transform it, then copy to existing geometry
                # This avoids issues with repeated transforms if pose changes
                temp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=camera_frame_size, origin=[0, 0, 0]
                )
                temp_frame.transform(poseMat)
                # Update vertices and triangles (more robust than just calling transform again)
                camera_frames_geoms[serial].vertices = temp_frame.vertices
                camera_frames_geoms[serial].triangles = temp_frame.triangles
                camera_frames_geoms[serial].compute_vertex_normals()
                vis.update_geometry(camera_frames_geoms[serial])


        # --- Final Update for Combined Point Cloud ---
        if not is_first_frame and update_pcd_geom:
             vis.update_geometry(pcd_to_update) # Update combined cloud

        if update_pcd_geom: # Mark first frame as done only if we added geometry
            is_first_frame = False

        # --- Poll Events ---
        if not vis.poll_events():
            keep_running = False
            break
        vis.update_renderer()

        # --- Control Playback Speed ---
        elapsed_time = time.time() - start_time
        target_frame_time = 1.0 / 30.0 # Target 30 FPS
        sleep_time = target_frame_time - elapsed_time
        if sleep_time > 0:
             time.sleep(sleep_time)

        i += frame_step

    print("Closing viewer window.")
    # vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View DynaMo captured point clouds (world frame) and camera poses using Open3D.")
    parser.add_argument("--folder", help="Folder containing .pickle data files.",
                        type=str, default='data')
    parser.add_argument("--full", help="Playback every frame (default is every 10th frame).",
                        action='store_true')
    parser.add_argument("--voxel_size", help="Voxel size for point cloud downsampling (e.g., 0.005). Set to 0 to disable.",
                        type=float, default=0.005)
    parser.add_argument("--frame_size", help="Size of coordinate frame markers for camera poses (meters).",
                        type=float, default=0.1)

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found '{args.folder}'")
        sys.exit(1)

    viewPointClouds(args.folder, args.full, args.voxel_size, args.frame_size)






##########################################################################################################################################
##                                               License: Apache 2.0. See LICENSE file in root directory.                               ##
##########################################################################################################################################
# """
# View captured 3D scans from DynaMo using Open3D
#
# Distributed as a module of DynaMo: https://github.com/anderson-cu-bioastronautics/dynamo_realsense-capture
# """
# import pickle
# import time
# import numpy as np
# import cv2
# import argparse
# import os
# import signal
# import sys
# import open3d as o3d # Use Open3D
#
# # Global flag to handle SIGINT gracefully within the visualizer loop
# keep_running = True
#
# def signalHandler(signal, frame):
#     """Sets a flag to terminate the loop gracefully."""
#     global keep_running
#     print("Signal received, shutting down viewer...")
#     keep_running = False
#
#
# def depthFrametoPC(deviceData):
#     """
#     Function which takes saved depth and color/infrared 2D frames and converts
#     them into 3D points and corresponding colors for Open3D.
#
#     Parameters
#     ----------
#     deviceData : dict
#         Dictionary with depth frame, infrared frame, depth sensor intrinsics,
#         and transformation matrix (poseMat).
#
#     Returns
#     -------
#     pointsTransformed : (N,3) numpy array
#         Array of calculated 3D point coordinates in the world frame.
#     colors_rgb : (N,3) numpy array
#         Array of corresponding RGB colors (0-255 range). Returns None if no
#         color/infrared data is available.
#     """
#     depth = deviceData.get('depth')
#     if depth is None:
#         print("Warning: No depth data found in deviceData.")
#         return None, None
#
#     rgb_image = None
#     if 'color' in deviceData: # if a color frame was saved in the frame
#         rgb_image = deviceData['color']
#     elif 'infrared' in deviceData: # otherwise use the saved infrared frame
#         infrared = deviceData['infrared']
#         # Convert grayscale infrared to 3-channel RGB for coloring
#         rgb_image = cv2.cvtColor(np.asanyarray(infrared), cv2.COLOR_GRAY2RGB)
#
#     cameraIntrinsics = deviceData.get('intrinsics')
#     poseMat = deviceData.get('poseMat')
#
#     if cameraIntrinsics is None or poseMat is None:
#         print("Warning: Missing intrinsics or pose matrix in deviceData.")
#         return None, None
#
#     try:
#         height, width = depth.shape
#         # Create Open3D intrinsic object
#         o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
#             width, height,
#             cameraIntrinsics['fx'], cameraIntrinsics['fy'],
#             cameraIntrinsics['ppx'], cameraIntrinsics['ppy']
#         )
#
#         # Create Open3D Image objects
#         o3d_depth = o3d.geometry.Image(depth.astype(np.uint16)) # Ensure correct depth type
#
#         o3d_rgb = None
#         if rgb_image is not None:
#              # Ensure image is uint8
#              if rgb_image.dtype != np.uint8:
#                  # Attempt conversion if necessary, e.g., if infrared was float
#                  if np.issubdtype(rgb_image.dtype, np.floating):
#                       rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
#                  else:
#                       rgb_image = rgb_image.astype(np.uint8)
#              o3d_rgb = o3d.geometry.Image(rgb_image)
#
#         # Create RGBD image
#         if o3d_rgb is not None:
#             rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#                 o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False
#             )
#         else:
#             # Create from depth only if no color is available
#             # Note: This results in a point cloud without color information
#              rgbd_image = o3d.geometry.Image(depth.astype(np.uint16)) # Use depth directly
#
#
#         # Create point cloud from RGBD image and intrinsics
#         # This gives points in the camera's coordinate frame
#         pcd_camera_frame = o3d.geometry.PointCloud.create_from_rgbd_image(
#             rgbd_image, o3d_intrinsics
#         ) if o3d_rgb is not None else o3d.geometry.PointCloud.create_from_depth_image(
#              o3d_depth, o3d_intrinsics, depth_scale=1000.0, depth_trunc=4.0
#             )
#
#
#         # Transform points to the world frame using the pose matrix
#         # poseMat is T_cam_to_world
#         pcd_camera_frame.transform(poseMat)
#
#         pointsTransformed = np.asarray(pcd_camera_frame.points)
#         colors_rgb = None
#         if o3d_rgb is not None:
#              # Colors need to be extracted carefully after potential downsampling/filtering
#              # Re-map colors or use points directly if possible (less robust)
#              # For simplicity, let's try getting colors directly from the transformed pcd
#              # Convert Open3D colors (0-1 float) back to 0-255 int if needed later,
#              # but let's try returning the points/colors arrays directly first.
#              # Note: Open3D `create_from_rgbd_image` might not preserve colors perfectly if depth invalid.
#              # A more robust way might be to re-project points back to image plane for color lookup.
#              # Sticking to simpler method for now:
#               if pcd_camera_frame.has_colors():
#                    colors_rgb_float = np.asarray(pcd_camera_frame.colors)
#                    colors_rgb = (colors_rgb_float * 255).astype(np.uint8)
#               else: # Handle case where create_from_depth_image was used
#                    colors_rgb = np.zeros_like(pointsTransformed, dtype=np.uint8) # Default to black or grey?
#                    # colors_rgb[:] = 128 # Grey default
#
#
#     except Exception as e:
#         print(f"Error processing frame to point cloud: {e}")
#         import traceback
#         traceback.print_exc()
#         return None, None
#
#
#     return pointsTransformed, colors_rgb
#
#
# def getPointCloud(frame):
#     """
#     Function which allows for the conversion of frame data from multiple cameras
#     into a single combined point cloud (points and colors).
#
#     Parameters
#     ----------
#     frame : dict
#         Dictionary with keys as serial numbers of each connected camera,
#         containing frames data from each camera (as loaded from pickle).
#
#     Returns
#     -------
#     all_points : (N,3) numpy array or None
#         Combined array of 3D point coordinates from all cameras.
#     all_colors : (N,3) numpy array or None
#         Combined array of corresponding RGB colors (0-255) from all cameras.
#         Returns array of zeros if no color data is available.
#     """
#     all_points_list = []
#     all_colors_list = []
#     has_color_data = False
#
#     # Process each camera's data sequentially (removed multiprocessing for simplicity with Open3D visualizer)
#     for device_serial, deviceData in frame.items():
#         points, colors = depthFrametoPC(deviceData)
#         if points is not None:
#             all_points_list.append(points)
#             if colors is not None:
#                 all_colors_list.append(colors)
#                 # Check if actual color info was returned, not just the placeholder zeros
#                 if np.any(colors > 0):
#                      has_color_data = True
#             else:
#                 # If colors are None, generate placeholder (e.g., grey) of the same size
#                 placeholder_colors = np.full_like(points, 128, dtype=np.uint8) # Grey
#                 all_colors_list.append(placeholder_colors)
#         else:
#              print(f"Warning: Failed to process points for device {device_serial}")
#
#
#     if not all_points_list:
#         return None, None # Return None if no points were generated
#
#     # Concatenate points and colors from all cameras
#     all_points = np.concatenate(all_points_list, axis=0)
#     all_colors = np.concatenate(all_colors_list, axis=0)
#
#     # If no camera provided actual color, the result might be all zeros/grey.
#     # The calling function should handle visualization accordingly.
#
#     return all_points, all_colors
#
#
# def viewPointClouds(folderDirectory, full_fps, downsample_voxel_size=0.005):
#     """
#     Function which allows for the viewing of pointClouds using Open3D.
#
#     Parameters
#     ----------
#     folderDirectory : str
#         Directory containing .pickle files of saved frames from capture.
#     full_fps : bool
#         True if you wish to view every frame, False to view every 10th frame.
#     downsample_voxel_size : float, optional
#         Voxel size for downsampling the point cloud to improve performance.
#         Set to 0 or None to disable downsampling. Default is 0.005 (5mm).
#     """
#     global keep_running
#     keep_running = True # Reset flag at start
#     signal.signal(signal.SIGINT, signalHandler) # Setup signal handler
#
#     print("Initializing Open3D Visualizer...")
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name='DynaMo Point Cloud Viewer')
#
#     # Optional: Add coordinate frame geometry for reference
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
#     vis.add_geometry(coordinate_frame)
#
#     pcd = o3d.geometry.PointCloud() # Create an empty point cloud geometry
#     is_first_frame = True
#
#     # Register escape key callback to close window
#     def close_window_callback(vis):
#         global keep_running
#         print("Escape key pressed, closing window.")
#         keep_running = False
#         return False # Return False to close the window
#
#     vis.register_key_callback(256, close_window_callback) # 256 is the key code for Escape
#
#
#     i = 0
#     file_list = sorted([f for f in os.listdir(folderDirectory) if f.endswith('.pickle')])
#     num_files = len(file_list)
#     print(f"Found {num_files} pickle files in '{folderDirectory}'.")
#
#     frame_step = 1 if full_fps else 10
#
#     while keep_running and i < num_files:
#         start_time = time.time()
#         fname = os.path.join(folderDirectory, file_list[i])
#
#         try:
#             with open(fname, 'rb') as file:
#                 frame_data = pickle.load(file)
#         except FileNotFoundError:
#             print(f"Error: File not found {fname}")
#             i += frame_step
#             continue
#         except Exception as e:
#             print(f"Error loading file {fname}: {e}")
#             i += frame_step
#             continue
#
#         # print(f"Processing frame {i} ({file_list[i]})")
#         combined_points, combined_colors = getPointCloud(frame_data)
#
#         if combined_points is None:
#             print(f"Warning: No points generated for frame {i}. Skipping.")
#             i += frame_step
#             # Allow polling events even if frame is skipped
#             if not vis.poll_events(): break
#             vis.update_renderer()
#             continue
#
#
#         # --- Update Open3D Point Cloud ---
#         pcd.points = o3d.utility.Vector3dVector(combined_points)
#
#         if combined_colors is not None:
#             # Convert colors to float (0-1 range) for Open3D
#             pcd.colors = o3d.utility.Vector3dVector(combined_colors.astype(np.float64) / 255.0)
#         else:
#             # If no colors, clear existing colors or set a default
#             pcd.colors = o3d.utility.Vector3dVector() # Clear colors
#
#         # Optional Downsampling
#         if downsample_voxel_size and downsample_voxel_size > 0:
#              pcd_downsampled = pcd.voxel_down_sample(downsample_voxel_size)
#              # print(f"Downsampled from {len(pcd.points)} to {len(pcd_downsampled.points)} points")
#              pcd_to_update = pcd_downsampled
#         else:
#              pcd_to_update = pcd
#
#         # Update geometry in the visualizer
#         if is_first_frame:
#             vis.add_geometry(pcd_to_update, reset_bounding_box=True) # Add on first frame
#             is_first_frame = False
#         else:
#             vis.update_geometry(pcd_to_update) # Update subsequent frames
#
#
#         # Poll events and update renderer
#         if not vis.poll_events():
#             keep_running = False # Break if window is closed
#             break
#         vis.update_renderer()
#
#         # Control playback speed (optional)
#         elapsed_time = time.time() - start_time
#         target_frame_time = 1.0 / 30.0 # Target 30 FPS
#         sleep_time = target_frame_time - elapsed_time
#         if sleep_time > 0:
#              time.sleep(sleep_time)
#
#         i += frame_step # Move to the next frame index
#
#     # --- Cleanup ---
#     print("Closing viewer window.")
#     vis.destroy_window()
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="View DynaMo captured point clouds using Open3D.")
#     parser.add_argument("--folder", help="Folder containing .pickle data files.",
#                         type=str, default='data')
#     parser.add_argument("--full", help="Playback every frame (default is every 10th frame).",
#                         action='store_true') # Changed to action='store_true'
#     parser.add_argument("--voxel_size", help="Voxel size for downsampling (e.g., 0.005). Set to 0 to disable.",
#                         type=float, default=0.005)
#
#     args = parser.parse_args()
#
#     if not os.path.isdir(args.folder):
#         print(f"Error: Folder not found '{args.folder}'")
#         sys.exit(1)
#
#     # Call the viewing function
#     viewPointClouds(args.folder, args.full, args.voxel_size)
