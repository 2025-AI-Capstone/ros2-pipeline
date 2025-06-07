#!/bin/bash

# 1. Build and environment setup
set -e
echo "[stt_agent_tts_launcher] Starting build..."
colcon build --symlink-install
echo "[stt_agent_tts_launcher] Build completed!"

echo "[stt_agent_tts_launcher] Sourcing environment..."
source install/setup.bash
echo "[stt_agent_tts_launcher] Environment sourced!"

# 2. Launch nodes (each node in background)
echo "[stt_agent_tts_launcher] Launching nodes..."
ros2 launch stt stt.launch.py &
stt_pid=$!
echo "stt.launch.py launched (PID: $stt_pid)"

ros2 launch agent agent.launch.py &
agent_pid=$!
echo "agent.launch.py launched (PID: $agent_pid)"

ros2 launch tts tts.launch.py &
tts_pid=$!
echo "tts.launch.py launched (PID: $tts_pid)"

# 3. Handle termination signals and cleanup background processes
trap "echo 'Termination signal detected. Killing all nodes.'; kill $stt_pid $agent_pid $tts_pid; exit 1" INT TERM ERR

# 4. Wait for all background jobs to finish
wait

echo "All nodes have exited successfully."
exit 0
