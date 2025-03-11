All the project is based on OpenPCDet works: https://github.com/open-mmlab/OpenPCDet. Follow the instruction for its installation, but do not clone the repo, which you can use as a reference for the change I made. Be aware of some changes that you might do if you install spconv version 1 (I installed version 2). Be aware also that Nvidia Jetpack requires specific packages, for instance for [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html). Next, I installed [cumm from source](https://github.com/FindDefinition/cumm?tab=readme-ov-file#build-from-source-for-development-jit-recommend-for-develop) and then [spconv from source](https://github.com/traveller59/spconv?tab=readme-ov-file#build-from-source-for-development-jit-recommend). 

The first thing to know is that even when using streaming data, I initialized a DemoDataset instance even if this not strictly necessary, so you need to indicate a pointcloud or a folder of pointclouds for that purpose. This is done for easy of implementation since PCDet is built for batch data. 

Another thing you will change are the absolute path, i.e., `CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth`. I may have used it many times becasue of easy of implementation. A relative path may be used as well most of the times.

To run a docker container in which you can install this application you can check inside the `docker` folder. I suggest checking also the OpenPCDet original, as I runned it a while ago. You need to run `cd docker bash run_image` to run the image.   

To install ROS, follow the ROS official instructions. If you need to install it inside a Docker container, I suggests doing it from inside the container (but you may wish to do it from Dockerfile for automation).
For what concerns its use, you have some processor that consume a pointcloud arrived at /ouster/points topic. This processor is just a node in ROS. Notice that on the Jetson there is ROS 2, so be aware of the change you need to do if you want to use ROS 1. On the server (westworld), I runned ROS 1. You have an example of that at `ros_demo/pointcloud_subscriber.py`, but that was a first version of our node. You can run it on your ros package, pasting the code into your node function.
Similarly can be done for ROS 2. The `ros_demo/pointcloud_processor_ros2.py` is the base processor. With the same name but with different ending there is `ros_demo\pointcloud_processor_ros2_crop_gpu.py`, which use our custom cropping on GPU. 

To test on nuscenes I used `tools/test_nuscenes.sh`. Here is defined the file to use to test, that can be `tools/test.py` or `tools/test_less_computation.sh` to use our method on the nuscenes dataset (follow the PCDet instruction to install it, but keep in mind that I used nuscenes Mini, so you may have to change some configuration files). I created `tools/eval_utilis/evaluation_less_computation.py` for our function, to separate it from the original evaluation. 

From `tools/eval_utilis/evaluation_less_computation.py`:
```
from tools.streaming_utilis.temporal_state import update_temporal_state, predict_from_state
from tools.streaming_utilis.crop import crop_point_cloud
```
In `tools/streaming_utilis` are defined some functions used for nuScenes. They are quite similar the ones used in ROS for our custom dataset. Notice that we do not need to make use the speeds from our function since on nuScenes we already get it.

To check the result for singular saved pointcloud I used the `tools/run_demo_headless` file. I never used the demo with visualization, as I didn't have a screen (and didn't get the permission for X11 forwarding). The `tools/run_demo_headless.sh` run the base model on our custom dataset, saving the predictions in a file to visualize it locally on my pc (keep in mind that a visualization is necessary!). 

For the pretrained weights you can check the [model zoo](https://github.com/open-mmlab/OpenPCDet/tree/master?tab=readme-ov-file#model-zoo) of PCDet.