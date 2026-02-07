from setuptools import setup
import os
from glob import glob

package_name = 'turtlebot3_autorace_driving'

setup(
    name=package_name,
    version='2.1.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='Driving package',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_lane = turtlebot3_autorace_driving.control_lane:main',
            'prof_driver = turtlebot3_autorace_driving.prof_driver:main',
        ],
    },
)
