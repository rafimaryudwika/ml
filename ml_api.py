from flask import Flask, request, jsonify
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
from dotenv import dotenv_values,load_dotenv

# Load the environment variables

load_dotenv() 

model_name = os.getenv('TF_MODEL_FILE')
model_labels = os.getenv('TF_MODEL_LABELS')

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Load the trained model
model = keras.models.load_model(model_name)
class_names = load_labels(model_labels) # Update with your class names

# Create the Flask app
app = Flask(__name__)

# Define a route for your API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
  
      
    # # Get the image from the request
    image_path = request.files['image']
    if image_path.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
      
      
    if image_path:
        filename = secure_filename(image_path.filename)
        image_path.save('uploads/images/' + filename)

        # Retrieve the filename
        uploaded_filename = ('uploads/images/' + filename)
    # image = load_img(image, target_size=(224, 224))
    # input_arr = img_to_array(image) / 255.0
    # input_arr = tf.expand_dims(input_arr, axis=0)

    # # Make predictions using the loaded model
    # predictions = model.predict(input_arr)
    # predicted_class = tf.argmax(predictions[0])
    image = load_img(uploaded_filename, target_size=(224, 224))
    input_arr = img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

  # Make predictions
    predictions = model.predict(input_arr)[0]
    max_index = np.argmax(predictions)
    class_label = class_names[max_index]

    # freshness_status = "segar" if class_label.endswith("segar") else "busuk"
    
    if class_label.endswith("segar"):
      fruit_name = class_label[0:-5]  # Remove "fresh" prefix
      freshness_percentage = round(predictions[max_index] * 100, 2)
    else:
      fruit_name = class_label[0:-5]  # Remove "rotten" prefix
      freshness_percentage = round((1 - predictions[max_index]) * 100, 2)
      
    if freshness_percentage >= 75:
          status_kesegaran = 'Sangat segar'
    elif freshness_percentage >= 50 and freshness_percentage < 75:
          status_kesegaran = 'Cukup segar'
    elif freshness_percentage >= 25 and freshness_percentage < 50:
          status_kesegaran = 'Sudah membusuk'
    elif freshness_percentage < 25:
          status_kesegaran = 'Sangat busuk'  
    # Prepare the response
    result = {
       'fruit_name': fruit_name,
       'freshness_status': status_kesegaran,
       'freshness_percentage':freshness_percentage,
    }

    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)
