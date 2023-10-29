# Gesture Classifier 🚀

![GitHub](https://img.shields.io/github/license/rigvedrs/Red-Hen-Gesture-Classifier)

![GitHub last commit](https://img.shields.io/github/last-commit/rigvedrs/Red-Hen-Gesture-Classifier)

The Gesture Classifier is a Python application that allows you to classify hand gestures in images. It uses computer vision techniques to detect persons in images, identify keypoints for pose estimation, and classify gestures based on the detected poses.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Uploading an Image](#uploading-an-image)
  - [Classifying Poses](#classifying-poses)
  - [Displaying Results](#displaying-results)
- [Predefined Test Images](#predefined-test-images)
- [License](#license)

## Getting Started

### Prerequisites

Before using the Gesture Classifier, make sure you have the required software and packages installed. You can install the packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rigvedrs/Red-Hen-Gesture-Classifier.git
   cd Red-Hen-Gesture-Classifier
   ```

2. Run the Streamlit application:

   ```bash
   streamlit run run_pipeline.py
   ```

### Usage

#### Uploading an Image

1. Run the Streamlit application as described in the installation section.

2. Upload an image or select a predefined test image.

#### Classifying Poses

- Click the "Classify Poses" button to perform gesture classification.

#### Displaying Results

- Click the "Display Images" button to view the results.
- Detected gestures and images will be displayed with their corresponding classifications.

### Predefined Test Images

A set of predefined test images is included in the "test_imgs" folder for your convenience.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/rigvedrs/Red-Hen-Gesture-Classifier/blob/main/LICENSE) file for details.

---
