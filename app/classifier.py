
"""Image classification using a pretrained EfficientNetB3 model."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB3,
    decode_predictions,
    preprocess_input,
)

from app.config import MODEL_INPUT_SIZE, TOP_PREDICTIONS


@dataclass(frozen=True)
class Prediction:
    """Represent a single image classification prediction."""

    label: str
    confidence: float


class ImageClassifier:
    """Classify images using EfficientNetB3."""

    def __init__(self) -> None:
        """Load the pretrained model once."""
        self.model = EfficientNetB3(weights="imagenet")

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Prepare an image for model inference."""
        rgb_image = image.convert("RGB")
        resized_image = rgb_image.resize(MODEL_INPUT_SIZE)

        image_array = np.asarray(resized_image, dtype=np.float32)
        batched_image = np.expand_dims(image_array, axis=0)

        return preprocess_input(batched_image)

    def classify_image(self, image: Image.Image) -> list[Prediction]:
        """Return the most likely ImageNet classifications."""
        prepared_image = self.preprocess_image(image)

        predictions = self.model.predict(
            prepared_image,
            verbose=0,
        )

        decoded_predictions = decode_predictions(
            predictions,
            top=TOP_PREDICTIONS,
        )[0]

        return [
            Prediction(
                label=label.replace("_", " "),
                confidence=float(confidence) * 100,
            )
            for _, label, confidence in decoded_predictions
        ]
