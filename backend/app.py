from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import sys

sys.path.append('utils')

from cnn_utils import *
from decision_tree_utils import *
from db_utils import *


app = Flask(__name__)
CORS(app) 

db_params = {
    'dbname': 'chertov',
    'user': 'postgres',
    'password': '3220',
    'host': 'localhost',
    'port': '5432'
}


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if the 'image' file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the image file from the request
        image_file = request.files['image']

        # Read the image using OpenCV
        image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        #calculating data for decision tree model
        black_and_white_img = get_silhouette(image_np)  
        silhouette_ratios_list = get_silhouette_ratios(black_and_white_img)
        output_class = get_CNN_prediction(image_np)

        #getting Decision Tree Classifier results
        is_good = get_decision_tree_prediction(silhouette_ratios_list, output_class)
        class_name = "Tree" if output_class == 1 else "Bush"

        #operations, needed to uplaod image to DB
        connection, cursor = create_connection(db_params)
        create_image_table(connection, cursor)
        upload_image_to_DB(connection, cursor, image_file, is_good)
        close_connection(connection, cursor)

        return jsonify({'isGood': is_good, "class": str(class_name)}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)