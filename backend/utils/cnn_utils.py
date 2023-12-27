import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_CNN_prediction(image):
	model = load_model('trained_models/imageclassifier.h5')

	# Resize the image to match the input size of the CNN model
	image_for_classfier = cv2.resize(image, (256, 256))

	# Normalize pixel values to the range [0, 1]
	image_for_classfier = image_for_classfier / 255.0

	# Expand dimensions to create a batch (if the model expects batch input)
	image_for_classfier = np.expand_dims(image_for_classfier, axis=0)

	predictions = model.predict(image_for_classfier)

	# Optionally, get the class with the highest probability
	predicted_class = np.argmax(predictions, axis=1)

	if predictions[0][0] > 0.5:
		#bush
		return 0
	elif predictions[0][0] < 0.5: 
		#tree
		return 1