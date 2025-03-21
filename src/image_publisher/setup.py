import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'image_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seokjun Song',
    maintainer_email='suwdle1917@gmail.com',
    description='image_publisher node for the AI pipeline project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher_node = image_publisher.image_publisher_node:main'
        ],
    },
)
