# Use winpty for Mintty Terminals (like Git Bash)
# Remove if run from CMD or PowerShell
winpty docker run -it --rm --net=host -e DISPLAY=192.168.189.96:0.0 \
    -v $(pwd)/bag_files:/bag_files \
    ros-rviz
    # -v D:/Programmazione/Thesis/masterThesis/ros/bag_files:/bag_files \