import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download

classes = ["Cat", "Dog"]

model_path = hf_hub_download(
    repo_id="sirunchained/cats-vs-dogs",
    filename="cats-vs-dogs.keras"
)

model = tf.keras.models.load_model(model_path)

def predict_img(img):
  """
  Predict image is a dog or cat
  """

  img = img.resize((150, 150))
  img = np.array(img)

  img = tf.expand_dims(img, axis=0)
  img = tf.cast(img, tf.float32)

  predictions = model.predict(img)
  predicted_confidences = predictions[0]

  result = {classes[i]: float(predicted_confidences[0]) for i in range(len(classes))}

  return result

iface = gr.Interface(
    fn=predict_img,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"), 
    title="Cats vs Dogs Classifier with EfficientNetB0",
    description="Upload an image to get its predicted category and confidence."
)

iface.launch()
