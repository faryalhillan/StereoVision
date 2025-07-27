# StereoVision Project

StereoVision is a computer vision project focused on stereo camera calibration, 3D reconstruction, and object tracking using stereo video data. The project is designed for research and practical applications in surveillance, UAVs, or autonomous systems.

## Features
- **Stereo Camera Calibration**: Complete calibration pipeline with parameter extraction and validation
- **Real-time 3D Reconstruction**: GPU-accelerated stereo processing with disparity-to-depth conversion
- **Advanced Object Tracking**: YOLO-based detection with multi-object tracking and ID assignment
- **Kalman Filtering**: Position smoothing and noise reduction for stable 3D coordinates
- **Bird's Eye View (BEV) Mapping**: Real-time top-down visualization with track history and grid overlay
- **Multi-threaded Processing**: Producer-consumer architecture for synchronized stereo video processing
- **Data Logging & Export**: CSV export with timestamps, 3D coordinates, and disparity values
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
- CUDA and CUDNN
- opencv-python with CUDA module enabled (you can get the wheels at: https://github.com/cudawarped/opencv-python-cuda-wheels)
- PyTorch with CUDA (https://pytorch.org/get-started/locally/)
- Recommended: NVIDIA GPU for deep learning models

### Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/faryalhillana/StereoVision.git
   cd StereoVision
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Configuration
- Edit YAML files in `config/` to set up camera parameters, model paths, and tracking options.

### Usage

#### Training Models
- Use scripts and notebooks in `train/` to train or fine-tune detection models.

#### Stereo Calibration
- Use `stereo_calibration.ipynb` to calibrate your stereo camera setup.

#### Stereo Vision Processing (`stereo_vision.py`)
The main stereo vision script provides advanced real-time processing capabilities:

**Core Features:**
- **GPU-Accelerated Processing**: Utilizes CUDA for video decoding, frame rectification, and YOLO inference
- **Synchronized Stereo Video Processing**: Producer-consumer architecture for efficient frame synchronization
- **Real-time Object Detection**: YOLO-based detection with batch processing for optimal performance
- **3D Position Estimation**: Computes world coordinates (X, Y, Z) from stereo disparity using calibration matrices
- **Kalman Filtering**: Optional position smoothing with configurable noise parameters for stable tracking
- **Bird's Eye View (BEV) Visualization**: Real-time top-down view with customizable grid, track history, and positioning
- **CSV Data Logging**: Timestamped tracking data export with configurable intervals
- **Multi-threaded Architecture**: Separate threads for frame production and processing to maintain real-time performance

**Technical Capabilities:**
- Stereo rectification using pre-computed calibration maps
- Y-constraint stereo matching for improved accuracy
- Configurable batch processing (1-8 frames per batch)
- Real-time display of annotated left/right camera views
- Track history visualization with unlimited path memory
- Automatic timestamped output folder creation

Run with: `python stereo_vision.py` (configuration via `config/stereovision.yaml`)

#### Data Analysis
- Explore results and analyze data with `data_analysis.ipynb`.

## Citation
If you use this project in your research, please cite appropriately.

## Contact
For questions or collaboration, please contact:
- Faryal Hillan (faryalhillan@gmail.com)
