from setuptools import setup

package_name = 'tts_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seokjun',
    maintainer_email='seokjun@todo.todo',
    description='ROS2 Text-to-Speech Node using gTTS',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tts_node = tts_node.tts_node:main',
        ],
    },
    install_requires=[
        'setuptools',
        'gTTS',
        'pygame',
    ],
) 