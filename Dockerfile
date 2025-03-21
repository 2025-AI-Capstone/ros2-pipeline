FROM kolinguo/ros2:cuda118-devel

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN sudo apt-get update
RUN sudo apt-get install ros-humble-vision-msgs