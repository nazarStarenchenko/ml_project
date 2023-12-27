import cv2
import numpy as np
import joblib
import random


def read_image(image_path):
	# Read the image
	return cv2.imread(image_path)


def get_silhouette(img):
	if img is None:
		return "Error: Unable to load the image. Check if the file format is supported."

	# Convert the image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply thresholding
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Noise removal using morphological operations
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

	# Sure background area
	sure_bg = cv2.dilate(opening, kernel, iterations=3)

	return sure_bg


def get_silhouette_ratios(sure_bg):
	# Get the height and width of the image
	height, width = sure_bg.shape[:2]

	# Define the region of interest (ROI) for the top half
	roi_top_half = sure_bg[:height//2, width//4:3*width//4]

	# Calculate the number of white and black pixels in the top half
	white_pixels_top_half = np.sum(roi_top_half == 255)
	black_pixels_top_half = np.sum(roi_top_half == 0)

	# Calculate the ratio in the top half
	black_to_white_ratio = white_pixels_top_half / black_pixels_top_half

	random_int_wdith = random.randint(-150, 175)
	random_int_height = random.randint(-200, 250)

	height_ratio = height + random_int_height
	wdith_ratio = 2 * width // 4 + random_int_wdith

	return [black_to_white_ratio, height_ratio, wdith_ratio, wdith_ratio/height_ratio]


def get_decision_tree_prediction(silhouette_ratios_list, CNN_model_output_class):
	data_to_feed = [[silhouette_ratios_list[0], silhouette_ratios_list[1], silhouette_ratios_list[2], silhouette_ratios_list[3], CNN_model_output_class]]

	DecisionTreeModel = joblib.load('trained_models/decision_tree_model.joblib')
	predictions = DecisionTreeModel.predict(data_to_feed)

	return True if predictions[0] == 1 else False

