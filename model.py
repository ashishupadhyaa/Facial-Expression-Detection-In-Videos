from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np

class FacialExpression():
	emot_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	"""docstring for FacialExpression"""
	def __init__(self, json_file, weight_file):
		with open(json_file, 'r') as model_struct:
			model_j = model_struct.read()
			self.model_load = model_from_json(model_j)

		self.model_load.load_weights(weight_file)
		self.model_load.make_predict_function()

	def predict_emotion(self, img):
		emotion = self.model_load.predict(img)
		return FacialExpression.emot_list[np.argmax(emotion)]
