# StereoVision Project

StereoVision is a computer vision project focused on stereo camera calibration, 3D reconstruction, and object tracking using stereo video data. The project is designed for research and practical applications in surveillance, UAVs, or autonomous systems.

## Features
- Stereo camera calibration and parameter extraction
- 3D depth estimation from stereo images/videos
- Object detection and tracking (YOLO, ByteTrack)
- Data analysis and visualization tools
- Modular configuration via YAML files

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
   git clone <repo-url>
   cd StereoVision
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Configuration
- Edit YAML files in `config/` to set up camera parameters, model paths, and tracking options.

### Usage
- **Training Models:**
  - Use scripts and notebooks in `train/` to train or fine-tune detection models.
- **Stereo Calibration:**
  - Use `stereo_calibration.ipynb` to calibrate your stereo camera setup.
- **Stereo Vision Processing:**
  - Run `stereo_vision.py` to process stereo videos and perform 3D reconstruction or tracking.
- **Data Analysis:**
  - Explore results and analyze data with `data_analysis.ipynb`.

## Citation
If you use this project in your research, please cite appropriately.

## Contact
For questions or collaboration, please contact:
- Faryal Hillan (faryalhillan@gmail.com)
