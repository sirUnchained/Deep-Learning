# Cats vs Dogs Classifier with EfficientNetB0

This model is a deep learning classifier capable of distinguishing between images of cats and dogs. It leverages the power of `EfficientNetB0` for robust feature extraction and is fine-tuned for this binary classification task.

## Dataset

- **Name**: `cats_vs_dogs` from `tensorflow_datasets`
- **Description**: Contains 23,262 images in total, split into approximately equal numbers of cat and dog images.
  - Cats are labeled as `0`.
  - Dogs are labeled as `1`.

## Model Architecture

The model is built using a pre-trained `EfficientNetB0` backbone (pre-trained on ImageNet) and custom classification layers:

- **Base Model**: `EfficientNetB0` (without top layers, `include_top=False`)
- **Input Shape**: `(150, 150, 3)` (images are resized to this dimension)
- **Custom Layers**:
  - `GlobalAveragePooling2D()`: Reduces spatial dimensions of feature maps.
  - `Dense(units=100, activation="relu")`: A fully connected layer with ReLU activation.
  - `Dense(units=1, activation="sigmoid")`: Output layer for binary classification, producing a probability between 0 and 1.

## Training Details

- **Epochs**: 7 (with EarlyStopping)
- **Batch Size**: 32
- **Optimizer**: Adam with a learning rate of 5e-5
- **Loss Function**: Binary Crossentropy
- **Callbacks**: EarlyStopping (patience=3, monitor='val_loss'), ModelCheckpoint (saves best model based on 'val_loss')

## Evaluation Metrics

The model was evaluated on a test set of 4653 images. Here are the key metrics:

- **Test Loss**: 0.0334
- **Test Accuracy**: 0.9884

**Classification Report**:
```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      2273
           1       0.99      0.99      0.99      2380

    accuracy                           0.99      4653
   macro avg       0.99      0.99      0.99      4653
weighted avg       0.99      0.99      0.99      4653
```

## How to Use

You can use this model for image classification. Here's a Python snippet using TensorFlow and the `huggingface_hub` library:

```python
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
import numpy as np

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="sirunchained/cats-vs-dogs",
    filename="cats-vs-dogs.keras"
)

# Load the model
model = tf.keras.models.load_model(model_path)

classes = ["Cat", "Dog"]
IMG_SIZE = 150

def predict_image(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32)

    predictions = model.predict(img_array)
    predicted_confidence = predictions[0][0] # Since it's a binary classifier with sigmoid output

    if predicted_confidence >= 0.5:
        predicted_class = "Dog"
        confidence = predicted_confidence
    else:
        predicted_class = "Cat"
        confidence = 1 - predicted_confidence

    return predicted_class, confidence

# Example usage:
# predicted_class, confidence = predict_image("path/to/your/image.jpg")
# print(f"Predicted: {predicted_class} with confidence: {confidence:.2f}")
```

## Gradio Demo

You can interact with a live demo of this model through Gradio:

[Gradio Demo Link](https://huggingface.co/spaces/sirunchained/cats-vs-dogs-gradio)