
## Project Overview
This project focuses on real-time hair segmentation and color transfer using deep learning techniques. It utilizes convolutional neural networks (U-Net architecture) to segment hair in images and videos, followed by realistic color transformation. The implementation is designed to be portable and deployable on embedded systems such as Raspberry Pi.


## Features
- **Hair Segmentation**: Uses a deep learning model trained on the CelebAMask-HQ dataset to accurately segment hair in images.
- **Color Transfer**: Implements color transformation using the HSV color space to achieve realistic hair color changes.
- **Embedded System Deployment**: The model is integrated into a Raspberry Pi system for real-time processing.
- **Web Interface (Prototype)**: Initial attempts were made to develop a web interface for user interaction.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Embedded System**: Raspberry Pi
- **Web Development**: HTML, CSS, JavaScript (prototype)

⚠ Warning: Before running the script, ensure you modify the file paths in the script to match your system configuration.

## Installation
### Requirements:
```bash
Python 3.x
PyTorch
OpenCV
Raspberry Pi (for embedded implementation)
```

### Steps:
```bash
# Clone this repository
git clone https://github.com/gabrielpecoraro/Hair_Segmentation.git

# Download the Dataset # or use the small one in dataset
https://www.kaggle.com/datasets/arghyadutta1/celebmask


# Run the hair segmentation and color transfer script
python training_main.py

# Run the testing phase on a chosen file :
# Either 
test_model.py
# Or
test_model2.py
# Or
test_models.py
```

(Optional) Deploy on Raspberry Pi following the instructions in the documentation.

## Usage
```bash
# Standalone Mode
Run the Python script to process images and videos.

# Embedded Mode
Capture and process images in real-time using a Raspberry Pi and an attached camera.

# Web Interface (Work in Progress)
Intended for user-friendly interaction but requires further development.
```

## Future Improvements
- Enhance robustness of color transfer algorithms.
- Improve real-time performance on embedded systems.
- Finalize and integrate the web interface with a backend API.

## Contributors
- **Sami Belasri**
- **Gabriel Pecoraro**
- **Supervisor**: Rémi Giraud

## Acknowledgments
Special thanks to our supervisor for guidance and valuable insights throughout the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

