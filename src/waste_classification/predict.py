import os

import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from pydantic import BaseModel, HttpUrl

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class PredictRequest(BaseModel):
    url: HttpUrl


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float


def preprocess_pytorch_style(X):
    X = X / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std
    
    return X.astype(np.float32)


def predict_waste(url: str):
    preprocessor = create_preprocessor(
        preprocess_pytorch_style,
        target_size=(224, 224)
    )

    session = ort.InferenceSession(
        get_model_path(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    X = preprocessor.from_url(url)

    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    predictions_dict = dict(zip(classes, float_predictions))
    
    top_class = max(predictions_dict, key=predictions_dict.get)
    top_probability = predictions_dict[top_class]
    
    return predictions_dict, top_class, top_probability


def get_model_path() -> str:
    return os.environ.get("WASTE_CLASSIFICATION_MODEL_PATH", "model/waste_classifier_mobilenet_v4.onnx")