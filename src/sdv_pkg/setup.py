import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sdv_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*.[sd][ta][le]*')))
    ],
    install_requires=['setuptools',
                      'rclpy',
                      'sensor_msgs',
                      'cv_bridge',
                      'vision_msgs',
                      'message_filters',
                      'visualization_msgs',
                      ],
    zip_safe=True,
    maintainer='santiagortegab',
    maintainer_email='santiagortegab@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'video_pub = sdv_pkg.video_pub:main',
            'yolo_detection = sdv_pkg.yolo_detection:main',
            'yolo_depth = sdv_pkg.yolo_depth:main',
            'BEV = sdv_pkg.BEV:main'
        ],
    },
)
