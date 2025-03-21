import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'video_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/srv', glob('srv/*.srv')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seokjun Song',
    maintainer_email='suwdle1917@gmail.com',
    description='video_publisher node for the AI pipeline project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_publisher_node = video_publisher.video_publisher_node:main'
        ],
    },
)
