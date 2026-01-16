#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_ros
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import yaml
import pyrealsense2 as rs
from cv2 import aruco

class FrankaStateListener(Node):
    """
    A ROS2 node that subscribes to robot states, listens to TF transforms,
    streams from a RealSense camera, detects a ChArUco board, and saves
    synchronized data on keypress.
    """
    def __init__(self):
        super().__init__('franka_state_and_camera_recorder')

        # Add camera extrinsic storage
        self.camera_extrinsics = None
        self.camera_extrinsics_history = []
        self.uncertainty_history = []
        
        # --- Data Storage Setup ---
        self.output_dir = "calibration_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.get_logger().info(f"Created data storage directory at: {self.output_dir}")
        self.capture_count = 0
        self.latest_joint_state = None
        self.latest_transform = None

        # --- ROS2 Subscribers and Listeners ---
        self.joint_state_subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info("Subscribed to /joint_states.")

        # --- ChArUco Board and Camera Calibration Setup ---
        # IMPORTANT: Define your ChArUco board's properties here
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
       
        
        self.charuco_board = aruco.CharucoBoard(
            (8, 8),          # Number of squares (width, height)
            0.015,            # Square side length in meters
            0.011,            # Marker side length in meters
            self.aruco_dict)
        self.charuco_board.setLegacyPattern(True)

        self.aruco_params = aruco.DetectorParameters()
        
        # IMPORTANT: You MUST replace these with your camera's calibrated values.
        # These are just placeholders.
        self.camera_matrix = np.array([
            [613.88, 0, 323.53],
            [0, 613.88, 240.67],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32) # Assuming no distortion for now
        # self.charuco_params = aruco.CharucoParameters(self.camera_matrix, self.dist_coeffs) 
        self.charuco_params = aruco.CharucoParameters() 
        self.charuco_params.cameraMatrix = self.camera_matrix
        self.charuco_params.distCoeffs = self.dist_coeffs
        # --- RealSense Camera Setup ---
        self.pipeline = None
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.get_logger().info("Intel RealSense camera stream started successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense camera: {e}")
            self.pipeline = None

        # --- PLOTTING AND INTERACTION SETUP ---
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.get_logger().info("3D plot initialized. Press 'c' in the plot window to capture data.")
        self.get_logger().info("Close the plot window to exit.")

        self.timer = self.create_timer(0.05, self.timer_callback) # Faster timer for smoother video

    def joint_states_callback(self, msg):
        self.latest_joint_state = msg

    def on_key_press(self, event):
        self.get_logger().info(f"Pressed key: {event.key}")
        if event.key == 'c':
            self.capture_and_save_data()

    def detect_and_draw_charuco_pose(self, image):
        """
        Detects a ChArUco board in the given image, estimates its pose,
        and draws the axes on the image.

        Args:
            image: The input image in BGR format.

        Returns:
            A tuple containing:
            - The image with the pose axes drawn on it.
            - The rotation vector (rvec) of the board pose. None if not found.
            - The translation vector (tvec) of the board pose. None if not found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TODO maybe remove this, as the charuco_detector.detectBoard(gray) does anyway
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
         
        charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, self.charuco_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) > 0:
            obj_points, img_points = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            # The solvePnP function returns the rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
            if ret:
                # return rvec, tvec, charuco_corners, charuco_ids
                # draw the pose on the image
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
                return image, rvec, tvec
            else:
                return None, None, None
        else:
            return None, None, None

    def calculate_camera_extrinsics(self, robot_pose, charuco_pose):
        """
        Calculate camera extrinsics using robot end-effector pose and detected ChArUco pose.
        
        Args:
            robot_pose: Transform from robot base to end-effector
            charuco_pose: (rvec, tvec) of detected ChArUco board
            
        Returns:
            Dictionary containing camera extrinsics (rotation matrix and translation vector)
        """
        # Get robot end-effector pose as 4x4 transformation matrix
        trans = robot_pose.transform.translation
        rot = robot_pose.transform.rotation
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        R_robot = np.array([
            [1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[2]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
            [2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[0]*q[3])],
            [2*(q[0]*q[2] - q[1]*q[3]), 2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[0]**2 + q[1]**2)]
        ])
        t_robot = np.array([trans.x, trans.y, trans.z])
        T_robot = np.eye(4)
        T_robot[:3,:3] = R_robot
        T_robot[:3,3] = t_robot

        # Get ChArUco pose as 4x4 transformation matrix
        rvec, tvec = charuco_pose
        R_charuco, _ = cv2.Rodrigues(rvec)
        T_charuco = np.eye(4)
        T_charuco[:3,:3] = R_charuco
        T_charuco[:3,3] = tvec.flatten()

        # Calculate camera extrinsics
        # T_camera = T_robot * T_charuco^-1
        T_camera = T_robot @ np.linalg.inv(T_charuco)
        
        return {
            'rotation_matrix': T_camera[:3,:3].tolist(),
            'translation_vector': T_camera[:3,3].tolist()
        }

    def calculate_uncertainty(self, robot_pose, charuco_pose):
        """
        Calculate uncertainty in camera extrinsics estimation.
        This is a simplified uncertainty model based on:
        1. Robot pose uncertainty (assumed constant)
        2. ChArUco detection uncertainty (based on reprojection error)
        
        Args:
            robot_pose: Transform from robot base to end-effector
            charuco_pose: (rvec, tvec) of detected ChArUco board
            
        Returns:
            Dictionary containing uncertainty values for rotation and translation
        """
        # Simplified uncertainty model
        # Robot pose uncertainty (assumed constant)
        robot_rot_uncertainty = 0.001  # rad
        robot_trans_uncertainty = 0.001  # m
        
        # ChArUco detection uncertainty (simplified)
        charuco_rot_uncertainty = 0.002  # rad
        charuco_trans_uncertainty = 0.002  # m
        
        # Combined uncertainty (assuming independent errors)
        rot_uncertainty = np.sqrt(robot_rot_uncertainty**2 + charuco_rot_uncertainty**2)
        trans_uncertainty = np.sqrt(robot_trans_uncertainty**2 + charuco_trans_uncertainty**2)
        
        return {
            'rotation_uncertainty': rot_uncertainty,
            'translation_uncertainty': trans_uncertainty
        }

    def update_uncertainty_plot(self):
        """Update the uncertainty visualization plot."""
        if not hasattr(self, 'uncertainty_fig'):
            # Create new figure for uncertainty visualization
            self.uncertainty_fig, (self.uncertainty_ax1, self.uncertainty_ax2) = plt.subplots(2, 1, figsize=(10, 8))
            self.uncertainty_fig.suptitle('Camera Extrinsics Uncertainty')
            
        # Clear previous plots
        self.uncertainty_ax1.clear()
        self.uncertainty_ax2.clear()
        
        # Prepare data
        captures = range(1, len(self.uncertainty_history) + 1)
        rot_uncertainties = [u['rotation_uncertainty'] for u in self.uncertainty_history]
        trans_uncertainties = [u['translation_uncertainty'] for u in self.uncertainty_history]
        
        # Plot rotation uncertainty
        self.uncertainty_ax1.plot(captures, rot_uncertainties, 'b-', label='Rotation Uncertainty')
        self.uncertainty_ax1.fill_between(captures, 
                                        [r - 0.0005 for r in rot_uncertainties],
                                        [r + 0.0005 for r in rot_uncertainties],
                                        alpha=0.2)
        self.uncertainty_ax1.set_xlabel('Capture Number')
        self.uncertainty_ax1.set_ylabel('Uncertainty (rad)')
        self.uncertainty_ax1.set_title('Rotation Uncertainty')
        self.uncertainty_ax1.grid(True)
        self.uncertainty_ax1.legend()
        
        # Plot translation uncertainty
        self.uncertainty_ax2.plot(captures, trans_uncertainties, 'r-', label='Translation Uncertainty')
        self.uncertainty_ax2.fill_between(captures,
                                        [t - 0.0005 for t in trans_uncertainties],
                                        [t + 0.0005 for t in trans_uncertainties],
                                        alpha=0.2)
        self.uncertainty_ax2.set_xlabel('Capture Number')
        self.uncertainty_ax2.set_ylabel('Uncertainty (m)')
        self.uncertainty_ax2.set_title('Translation Uncertainty')
        self.uncertainty_ax2.grid(True)
        self.uncertainty_ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def capture_and_save_data(self):
        """Capture the current robot state, find the charuco pose, and save everything."""
        self.get_logger().info("'c' pressed. Attempting to capture data...")

        current_image = None
        if self.pipeline:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            if color_frame:
                current_image = np.asanyarray(color_frame.get_data())

        if current_image is None:
            self.get_logger().warn("Cannot capture: No image available from camera.")
            return
        if self.latest_joint_state is None:
            self.get_logger().warn("Cannot capture: No joint states received yet.")
            return
        if self.latest_transform is None:
            self.get_logger().warn("Cannot capture: No TF transform available yet.")
            return
            
        # --- Detect Charuco and get pose ---
        visualized_image, rvec, tvec = self.detect_and_draw_charuco_pose(current_image.copy())
        if rvec is None or tvec is None:
            self.get_logger().warn("Cannot capture: Failed to detect ChArUco board pose.")
            return

        # --- Calculate camera extrinsics and uncertainty ---
        self.camera_extrinsics = self.calculate_camera_extrinsics(self.latest_transform, (rvec, tvec))
        uncertainty = self.calculate_uncertainty(self.latest_transform, (rvec, tvec))
        
        self.camera_extrinsics_history.append(self.camera_extrinsics)
        self.uncertainty_history.append(uncertainty)
        
        # Update uncertainty visualization
        self.update_uncertainty_plot()
        
        self.get_logger().info("Calculated camera extrinsics:")
        self.get_logger().info(f"Rotation: {self.camera_extrinsics['rotation_matrix']}")
        self.get_logger().info(f"Translation: {self.camera_extrinsics['translation_vector']}")
        self.get_logger().info("Uncertainty:")
        self.get_logger().info(f"Rotation: ±{uncertainty['rotation_uncertainty']:.6f} rad")
        self.get_logger().info(f"Translation: ±{uncertainty['translation_uncertainty']:.6f} m")

        # --- Prepare File Paths ---
        self.capture_count += 1
        file_prefix = f"capture_{self.capture_count:03d}"
        image_path = os.path.join(self.output_dir, f"{file_prefix}.png")
        data_path = os.path.join(self.output_dir, f"{file_prefix}.yaml")

        # --- Save Image with Visualization ---
        cv2.imwrite(image_path, visualized_image)
        self.get_logger().info(f"Saved visualized image to: {image_path}")

        # --- Assemble and Save Data ---
        trans = self.latest_transform.transform.translation
        rot = self.latest_transform.transform.rotation
        data = {
            'end_effector_pose': {
                'translation': {'x': trans.x, 'y': trans.y, 'z': trans.z},
                'rotation_quaternion': {'x': rot.x, 'y': rot.y, 'z': rot.z, 'w': rot.w}
            },
            'joint_state': {
                'name': self.latest_joint_state.name,
                'position': self.latest_joint_state.position.tolist()
            },
            'charuco_pose': {
                'tvec': tvec.flatten().tolist(),
                'rvec': rvec.flatten().tolist()
            },
            'camera_extrinsics': self.camera_extrinsics,
            'uncertainty': uncertainty
        }
        with open(data_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.get_logger().info(f"Saved robot and marker data to: {data_path}")
        self.get_logger().info(f"--- Capture {self.capture_count} complete ---")

    def timer_callback(self):
        # --- Update Robot Pose Plot ---
        try:
            transform = self.tf_buffer.lookup_transform('fr3_link0', 'fr3_link7', rclpy.time.Time())
            self.latest_transform = transform
            self.update_plot(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass # Warnings will be logged in capture function if needed
        
        # --- Update Live Camera Feed --- 
        if self.pipeline:
            frames = self.pipeline.wait_for_frames(timeout_ms=200) # Wait for a new frame
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    image = np.asanyarray(color_frame.get_data())
                    # Detect markers and draw on the live feed
                    display_image, _, _ = self.detect_and_draw_charuco_pose(image)
                    cv2.imshow("Live Camera Feed with Detection", display_image)
                    cv2.waitKey(1)

        # Check for window closures
        if not plt.fignum_exists(self.fig.number):
            self.get_logger().info("Plot window closed. Shutting down node.")
            self.destroy_node()
            rclpy.shutdown()

    def update_plot(self, transform):
        """Updates the 3D plot with the new end-effector pose and camera extrinsics."""
        self.ax.cla()
        self.ax.scatter(0, 0, 0, color='black', marker='o', s=100, label='World Frame')
        self.ax.quiver(0, 0, 0, 0.2, 0, 0, color='r', arrow_length_ratio=0.3)
        self.ax.quiver(0, 0, 0, 0, 0.2, 0, color='g', arrow_length_ratio=0.3)
        self.ax.quiver(0, 0, 0, 0, 0, 0.2, color='b', arrow_length_ratio=0.3)
        
        # Plot robot end-effector
        trans = transform.transform.translation
        self.ax.scatter(trans.x, trans.y, trans.z, color='blue', marker='o', s=100, label='End-Effector')
        rot = transform.transform.rotation
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        R = np.array([
            [1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[2]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
            [2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[0]*q[3])],
            [2*(q[0]*q[2] - q[1]*q[3]), 2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[0]**2 + q[1]**2)]
        ])
        axis_length = 0.1
        self.ax.quiver(trans.x, trans.y, trans.z, R[0,0], R[1,0], R[2,0], length=axis_length, color='r')
        self.ax.quiver(trans.x, trans.y, trans.z, R[0,1], R[1,1], R[2,1], length=axis_length, color='g')
        self.ax.quiver(trans.x, trans.y, trans.z, R[0,2], R[1,2], R[2,2], length=axis_length, color='b')

        # Plot camera extrinsics if available
        if self.camera_extrinsics is not None:
            R_cam = np.array(self.camera_extrinsics['rotation_matrix'])
            t_cam = np.array(self.camera_extrinsics['translation_vector'])
            self.ax.scatter(t_cam[0], t_cam[1], t_cam[2], color='purple', marker='o', s=100, label='Camera')
            self.ax.quiver(t_cam[0], t_cam[1], t_cam[2], R_cam[0,0], R_cam[1,0], R_cam[2,0], length=axis_length, color='r')
            self.ax.quiver(t_cam[0], t_cam[1], t_cam[2], R_cam[0,1], R_cam[1,1], R_cam[2,1], length=axis_length, color='g')
            self.ax.quiver(t_cam[0], t_cam[1], t_cam[2], R_cam[0,2], R_cam[1,2], R_cam[2,2], length=axis_length, color='b')

        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')
        self.ax.set_title('FR3 End-Effector and Camera Pose (Press \'c\' to capture)')
        self.ax.legend()
        self.ax.set_xlim([-1.0, 1.0]); self.ax.set_ylim([-1.0, 1.0]); self.ax.set_zlim([0, 1.5])
        self.ax.set_aspect('equal', adjustable='box')
        plt.draw()
        plt.pause(0.001)

    def destroy_node(self):
        """Custom cleanup."""
        if self.pipeline:
            self.pipeline.stop()
            self.get_logger().info("RealSense camera pipeline stopped.")
        cv2.destroyAllWindows()
        plt.close(self.fig)
        if hasattr(self, 'uncertainty_fig'):
            plt.close(self.uncertainty_fig)
        self.get_logger().info("Windows closed and node destroyed.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    recorder_node = FrankaStateListener()
    
    while rclpy.ok():
        try:
            rclpy.spin_once(recorder_node, timeout_sec=0.1)
        except KeyboardInterrupt:
            break
        # Break loop if plot window is closed
        if not plt.fignum_exists(recorder_node.fig.number):
            break

    recorder_node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()

if __name__ == '__main__':
    main()
