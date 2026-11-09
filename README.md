
# Image Recognition

Project 01/30 — 30 Projects Challenge

A Python desktop application that classifies images using a pretrained EfficientNetB3 deep learning model.

The application supports image uploads, live camera previews, and photo capture with a centered focus area.

## Features

- Upload images from your computer
- Open the webcam and preview live video
- Capture photos directly from the camera
- Focus on a specific region before classification
- Display the top 5 predictions with confidence scores
- Modern dark-themed desktop interface

## Technologies

- Python
- TensorFlow / Keras
- EfficientNetB3 (ImageNet pretrained)
- OpenCV
- CustomTkinter
- Pillow
- NumPy

## Project Structure

```text
app/
├── __init__.py
├── config.py
├── classifier.py
├── camera.py
├── image_processing.py
└── ui.py

Image Recognition.py
requirements.txt
README.md
.gitignore
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

2. Open the project folder:

```bash
cd YOUR_REPOSITORY
```

3. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

4. Run the application:

```bash
python "Image Recognition.py"
```

## How It Works

1. Upload an image or open the webcam.
2. If using the webcam, position the object inside the green focus box.
3. Capture a photo.
4. The selected image region is processed by EfficientNetB3.
5. The application displays the five highest-scoring ImageNet classes.

## Limitations

- The model recognizes only categories included in ImageNet.
- Predictions may be inaccurate for unfamiliar objects.
- Confidence scores do not guarantee correctness.
- Focus mode crops a region; it does not automatically detect objects.

## Future Improvements

- Improve the prediction display
- Add optional image saving
- Evaluate classification performance on more examples

## Author

Saria Askar

30 Projects Challenge — Project 01
