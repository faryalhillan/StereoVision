# StereoVision Project

StereoVision is a GPU-accelerated computer vision project focused on stereo camera calibration, real-time 3D reconstruction, and object tracking using stereo video data. The project leverages CUDA and modern deep learning for research and practical applications in surveillance, UAVs, or autonomous systems.

## Features
- **Stereo Camera Calibration**: Complete calibration pipeline with parameter extraction and validation
- **Real-time 3D Reconstruction**: GPU-accelerated stereo processing with disparity-to-depth conversion
- **Advanced Object Tracking**: YOLO-based detection with multi-object tracking and ID assignment
- **Kalman Filtering**: Position smoothing and noise reduction for stable 3D coordinates
- **Bird's Eye View (BEV) Mapping**: Real-time top-down visualization with track history and grid overlay
- **Multi-threaded Processing**: Producer-consumer architecture for synchronized stereo video processing
- **Data Logging & Export**: CSV export with timestamps, 3D coordinates, and disparity values
- **Performance Benchmarking**: Comprehensive model performance analysis with detailed metrics and visualizations
- **Data Analysis & Validation**: Ground truth comparison with flight data and accuracy assessment
- **GPU Optimization**: CUDA support for video decoding, rectification, and neural network inference
- **Modular Configuration**: YAML-based configuration for all processing parameters

## Folder Structure
```
StereoVision/
├── config/           # Configuration files (YAML)
├── data/             # All data (videos, calibration, datasets, tracking results)
│   ├── airdata/      # Airdata CSV logs
│   ├── calibration/  # Calibration images and parameters
│   ├── content/      # Stereo video files
│   ├── dataset/      # Training/validation data for models
│   └── tracking/     # Tracking results
├── train/            # Training scripts, models, and results
├── stereo_vision.py  # Main stereo vision script
├── stereo_calibration.ipynb  # Calibration notebook
├── data_analysis.ipynb       # Data analysis notebook
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA and cuDNN (Tested on CUDA 12.9.1 and CUDNN 9.10.2)
- NVIDIA GPU with compute capability 6.0+ (recommended)
- OpenCV with CUDA support (Tested on OpenCV 4.12.0)
- PyTorch with CUDA support
- TensorRT for optimized inference

### Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/faryalhillan/StereoVision.git
   cd StereoVision
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   Note: 
   For CUDA-enabled OpenCV, you may need to install the prebuilt wheels at:
   https://github.com/cudawarped/opencv-python-cuda-wheels
   For CUDA-enabled PyTorch, you need to follow this guide:
   https://pytorch.org/get-started/locally/

### Configuration
- Edit YAML files in `config/` to set up camera parameters, model paths, and tracking options.
- `stereo_vision.yaml`: Main configuration for video paths, BEV settings, Kalman filtering
- `bytetrack.yaml`: ByteTrack multi-object tracking parameters

### Usage

#### 1. Stereo Camera Calibration (`stereo_calibration.ipynb`)
**Source**: https://github.com/ChristianOrr/stereo-camera-calibration

Interactive Jupyter notebook for complete stereo camera calibration

#### 2. Model Training and Performance Analysis (`train/train.ipynb`)
Comprehensive YOLO training pipeline with detailed performance benchmarking

#### 3. Stereo Vision Processing (`stereo_vision.py`)
The main stereo vision script provides advanced real-time processing capabilities:

**Core Features:**
- **GPU-Accelerated Processing**: Utilizes CUDA for video decoding, frame rectification, and YOLO inference
- **Synchronized Stereo Video Processing**: Producer-consumer architecture for efficient frame synchronization
- **Real-time Object Detection**: YOLO-based detection with configurable batch processing
- **3D Position Estimation**: Computes world coordinates (X, Y, Z) from stereo disparity using Q matrix
- **Y-Constraint Stereo Matching**: Improved accuracy with configurable Y-difference tolerance (±10px)
- **Optional Kalman Filtering**: Position smoothing with configurable noise parameters
- **Bird's Eye View (BEV) Visualization**: Real-time top-down view with customizable grid and track history
- **CSV Data Logging**: Timestamped tracking data export with automatic folder creation
- **Multi-threaded Architecture**: Separate threads for frame production and processing

**Technical Specifications:**
- **Frame Queue Management**: Configurable queue size for smooth processing
- **Display Configuration**: Dual window setup with customizable positions and sizes
- **Model Support**: Both PyTorch (.pt) and TensorRT (.engine) models
- **Calibration Integration**: Direct .npz calibration file loading
- **Configuration via YAML**: All parameters configurable through `config/stereo_vision.yaml`

Run with: `python stereo_vision.py`

#### 4. Data Analysis and Validation (`data_analysis.ipynb`)
Comprehensive analysis notebook for comparing tracking results with ground truth:

**Analysis Features:**
- **Ground Truth Comparison**: Flight data vs stereo tracking accuracy assessment
- **GPS Coordinate Conversion**: Haversine distance calculations and local coordinate transformation
- **Time Synchronization**: Automated alignment of flight and tracking data
- **Error Analysis**: RMSE calculations, statistical accuracy metrics
- **Coordinate System Alignment**: Rotation and offset corrections for coordinate matching
- **Performance Visualization**: Multiple plot types for error distribution and trends
- **Quality Assessment**: Categorized accuracy ratings (excellent/good/acceptable/poor)
- **Trajectory Analysis**: Path comparison and velocity analysis

## Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 6.0+ (GTX 1060 or better)
- **VRAM**: Minimum 4GB for YOLOv11s, 6GB+ recommended for batch processing
- **RAM**: 8GB minimum, 16GB+ recommended for large datasets
- **Storage**: SSD recommended for video processing and model training

### Software Stack
- **Deep Learning**: YOLOv11 (Ultralytics), PyTorch, TensorRT
- **Computer Vision**: OpenCV 4.12.0 with CUDA support
- **Filtering**: FilterPy (Kalman filtering)
- **Data Processing**: NumPy, Pandas, Matplotlib
- **Configuration**: YAML-based parameter management

## Citation
If you use this project in your research, please cite appropriately and reference the repository.

## Contact
For questions, collaboration, or support:
- **Faryal Hillan**: faryalhillan@gmail.com
- **Repository**: https://github.com/faryalhillan/StereoVision
