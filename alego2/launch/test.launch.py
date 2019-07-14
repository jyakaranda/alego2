from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='alego2', node_executable='IP', output='log'),
        launch_ros.actions.Node(
            package='alego2', node_executable='LO', output='log'),
        launch_ros.actions.Node(
            package='alego2', node_executable='LM', output='log')
    ])
