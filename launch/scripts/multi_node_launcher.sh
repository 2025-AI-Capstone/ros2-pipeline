#!/bin/bash

# 1. Build and environment setup
set -e
echo "[multi_node_launcher] Starting build..."
colcon build --symlink-install
echo "[multi_node_launcher] Build completed!"

echo "[multi_node_launcher] Sourcing environment..."
source install/setup.bash
echo "[multi_node_launcher] Environment sourced!"

# 2. Launch nodes (each node in background)
echo "[multi_node_launcher] Launching nodes..."
ros2 launch camera camera.launch.py &
camera_pid=$!
echo "camera.launch.py launched (PID: $camera_pid)"

ros2 launch detector detector.launch.py &
detector_pid=$!
echo "detector.launch.py launched (PID: $detector_pid)"

ros2 launch falldetector falldetector.launch.py &
falldetector_pid=$!
echo "falldetector.launch.py launched (PID: $falldetector_pid)"

ros2 launch regenerator regenerator.launch.py &
regenerator_pid=$!
echo "regenerator.launch.py launched (PID: $regenerator_pid)"

ros2 launch fall_alert fall_alert.launch.py &
fall_alert_pid=$!
echo "fall_alert.launch.py launched (PID: $fall_alert_pid)"

# 3. Handle termination signals and cleanup background processes
trap "echo 'Termination signal detected. Killing all nodes.'; kill $camera_pid $detector_pid $falldetector_pid $regenerator_pid $fall_alert_pid; exit 1" INT TERM ERR

# 4. Wait for all background jobs to finish
wait

echo "All nodes have exited successfully."
exit 0
