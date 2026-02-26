# Food-101 Image Classifier

This is an image classification model capable of identifying 101 different food categories from the Food-101 dataset. The model leverages transfer learning using a pre-trained EfficientNetB0 as its base.

## Model Details

*   **Architecture**: EfficientNetB0 (feature extractor) + Custom Dense Head
*   **Task**: Image Classification
*   **Dataset**: [Food-101](https://huggingface.co/datasets/ethz/food101)
    *   101 food categories
    *   101,000 images (750 training, 250 validation per class)
    *   Images rescaled to a maximum side length of 512 pixels in original dataset.

## Training Details

### Approach
The model was trained using **transfer learning (feature extraction)**. The pre-trained `EfficientNetB0` model, which was originally trained on the ImageNet dataset, had its layers frozen. A new custom output layer (a `GlobalAveragePooling2D` followed by a `Dense` layer with softmax activation) was added on top of the frozen EfficientNetB0 base. Only this new head was trained on the Food-101 dataset.

### Preprocessing
Images from the Food-101 dataset were preprocessed as follows:
1.  Resized to `(256, 256)` pixels.
2.  Pixel values cast to `tf.float32`.

### Training Configuration
*   **Optimizer**: Adam
*   **Loss Function**: SparseCategoricalCrossentropy (suitable for integer-encoded labels)
*   **Metrics**: Accuracy
*   **Epochs**: 5 (with EarlyStopping if validation loss did not improve for 3 epochs)
*   **Batch Size**: 32
*   **Mixed Precision**: Enabled (`mixed_float16`) for faster training on compatible GPUs.

### Performance
After 5 epochs of training, the model achieved the following performance on the validation set:
*   **Validation Loss**: 0.9174
*   **Validation Accuracy**: 0.7482

## How to Use

To use this model for prediction, you'll need TensorFlow and the corresponding `EfficientNetB0` application.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

# Define the image size used during training
IMAGE_SIZE = 256

# Load the trained model
# Make sure to replace 'path/to/your/model/effnetB0_food_model.keras' with the actual path
loaded_model = tf.keras.models.load_model('./models/effnetB0_food_model.keras')

# Get class names from the dataset info
# (Assuming dsInfo was loaded earlier)
# If you don't have dsInfo, you can manually create the list of class names
class_names = [
  "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
  "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
  "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
  "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
  "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
  "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
  "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
  "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
  "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
  "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
  "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
  "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
  "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
  "mussels", "nachos", "omelette", "onion_rings", "oysters",
  "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
  "pho", "pizza", "pork_chop", "poutine", "prime_rib",
  "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
  "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
  "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
  "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.cast(img, tf.float32) # Already normalized implicitly by EfficientNet's internal preprocessing
    img = tf.expand_dims(img, axis=0) # Add batch dimension
    return img

# Example usage with a dummy image path (replace with your actual image)
# You might need to download a sample food image for testing
# For example, from the Food-101 dataset itself or any food image.
# dummy_image_path = tf.keras.utils.get_file('pizza.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Eq_pizza_italy_vs_us.jpg/640px-Eq_pizza_italy_vs_us.jpg')

# Preprocess the image
# preprocessed_image = preprocess_image(dummy_image_path)

# Make a prediction
# predictions = loaded_model.predict(preprocessed_image)
# predicted_class_index = np.argmax(predictions[0])
# predicted_class_name = class_names[predicted_class_index]

# print(f"The predicted food item is: {predicted_class_name}")
# print(f"Prediction probabilities: {predictions[0][predicted_class_index]:.4f}")


```

## Demo

You can try a demo in [here](https://huggingface.co/spaces/sirunchained/Food-101-image-classifier-gradio)

## License

[MIT]