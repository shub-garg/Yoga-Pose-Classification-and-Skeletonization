#!/bin/bash

# Install required Python packages
pip install kaggle opencv-python mediapipe numpy

# Kaggle dataset details
DATASET_OWNER="niharika41298"
DATASET_NAME="yoga-poses-dataset"
DOWNLOAD_PATH="yoga_dataset.zip"
EXTRACT_PATH="yoga_dataset"

# Download the dataset using Kaggle API
kaggle datasets download -d $DATASET_OWNER/$DATASET_NAME -p . --unzip

# Create output directory if it doesn't exist
mkdir -p output_skeletonized

# Run the skeletonization Python script
python skeletonization.py
