import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'gesture-arm-demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') or glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gilsocojp',
    maintainer_email='gilsocojp@gmail.com',
    description='Simple ROS 2 driver for a 4-DOF + gripper DYNAMIXEL XM430 arm over a U2D2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver_node = gesture_arm_demo.driver_node:main',
            'calibrate = tools.calibrate:main',
            'gesture_detector = gesture_arm_demo.gesture_detector:main',
            'gesture_to_motion = gesture_arm_demo.gesture_to_motion:main',
            'pose_mimic = gesture_arm_demo.pose_mimic:main',
        ],
    },
)