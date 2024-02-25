import numpy as np
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import joblib
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as keras_image

app = Flask(__name__)

# Load the LightGBM model
model = joblib.load('xray_lightgbm.pkl')

# Define the label encoder or preprocessing steps if needed
label = {0: 'PCOS', 1: 'NORMAL'}

vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in vggmodel.layers:
    layer.trainable = False


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            data = request.get_json()
            # Extract the URL from the JSON data
            image_url = data.get('url')
            if image_url:
                # Fetch the image from the URL
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Read the image from the response content
                    img = Image.open(BytesIO(response.content))
                    # Preprocess the image
                    img = img.resize((256, 256))
                    img_array = keras_image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  # Normalize the image
                    # Extract features using VGG16
                    features = vggmodel.predict(img_array)
                    features = features.reshape(features.shape[0], -1)
                    # Make prediction using the model
                    prediction = model.predict(features)[0]

                    # Convert NumPy array to Python list
                    prediction_list = prediction.tolist()
                    # Determine the predicted class
                    predicted_class = label[np.argmax(prediction_list)]

                    # Return the prediction result as JSON
                    return jsonify({
                        'prediction': prediction_list,
                        'predicted_class': predicted_class
                    })
                else:
                    return jsonify({'error': 'Failed to fetch image from the URL'})
            else:
                return jsonify({'error': 'No image URL provided'})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Invalid request method'})

if __name__ == '__main__':
    app.run(debug=True)
