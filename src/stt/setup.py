from setuptools import setup

package_name = 'stt'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'openai-whisper',
        'pyaudio',
        'numpy',
        'pvporcupine',  # Wake word detection
        'python-dotenv'  # 환경변수 관리
    ],
    zip_safe=True,
    maintainer='seokjun',
    maintainer_email='seokjun@todo.todo',
    description='ROS2 Speech-to-Text Node using Whisper with Wake Word Detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt = stt.stt:main',
        ],
    },
) 