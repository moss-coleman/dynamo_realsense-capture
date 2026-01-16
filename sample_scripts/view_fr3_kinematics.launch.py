# my_franka_subscriber/launch/view_fr3_kinematics.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_franka_description = get_package_share_directory('franka_description')
    urdf_xacro_path = os.path.join(pkg_franka_description, 'robots', 'fr3', 'fr3.urdf.xacro')
    robot_description_raw = xacro.process_file(urdf_xacro_path).toxml()
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_raw}]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    franka_listener_node = Node(
        package='my_franka_subscriber', # your package name
        executable='franka_listener',     # from your setup.py
        name='franka_listener',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        franka_listener_node
    ])
