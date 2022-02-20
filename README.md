# Stereo Structure and Motion (SaM / SfM) library

## Build

```bash
#--BUILD OPENCV

# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# Create build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.x
# Build
cmake --build .

sudo make install

#--BUILD PACKAGES
cmake --build .
```

## Egomotion
* Implemented using Ceres Solver, two basic variants with Quaternions and Euler angles
* Examples in `stereo_egomotion/main`

## Dense stereo
* Implementation of Semi-global matching in `reconstruction/base`
* Examples in `reconstruction/main`

## Feature tracking
* Monocular and steresopic variants.
* Implementation in `tracker` directory

## Publications

* [Improving the Egomotion Estimation by Correcting the Calibration Bias](http://www.cvlibs.net/datasets/kitti/eval_odometry_detail.php?&result=3ef2e95144c13778b66cec9b1d4c887c68684cea)
* Code in `deformation_field`
