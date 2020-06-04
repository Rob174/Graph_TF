# Adapté depuis https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
class GradCAM:
	""" Permet de visualiser l'activation d'un layer spécifié à la construction
	Dans cette adaptation, on adapte ce module à une évaluation régulière des activation et d'un résultat sous forme de GIF
	"""
	def __init__(self, model, classIdx,img_path, layerName=None,identifieur_model="model",identifieur_layer=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

		# load the original image from disk (in OpenCV format) and then
		# resize the image to its target dimensions
		orig = cv2.imread(img_path)
		self.orig = orig
		# load the input image from disk (in Keras/TensorFlow format) and
		# preprocess it
		image = load_img(img_path)
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		self.image = image
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()
		preds = model.predict(image)

		#Vérifie que le dossier de sortie existe
		identifieur_layer = identifieur_layer if identifieur_layer != None else self.layerName
		analysis_path = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2]+[identifieur_model,"analyse"])+"/"
		if os.path.isdir(analysis_path) == False:
			os.mkdir(analysis_path)
		if os.path.isdir(analysis_path+"grad_"+identifieur_layer+"/") == False:
			os.mkdir(analysis_path+"grad_"+identifieur_layer+"/")
		if os.path.isdir(analysis_path+"grad_"+identifieur_layer+"/heatmap/") == False:
			os.mkdir(analysis_path+"grad_"+identifieur_layer+"/heatmap/")
		if os.path.isdir(analysis_path+"grad_"+identifieur_layer+"/overlay_heatmap/") == False:
			os.mkdir(analysis_path+"grad_"+identifieur_layer+"/overlay_heatmap/")
		#Store identifier name of layer
		self.identifieur_layer = identifieur_layer
		self.save_path = analysis_path + "grad_"+identifieur_layer + "/"
    def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    def compute_heatmap(self, informations,eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
        # record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(self.image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (self.image.shape[2], self.image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
		heatmap = cv2.resize(heatmap, (self.original_image.shape[1], self.original_image.shape[0]))
		(heatmap, output) = cam.overlay_heatmap(heatmap, self.original_image, alpha=0.5)
		# Enregistrement dans le dossier des images
		cv2.imwrite(self.save_path+"/heatmap/"+informations+".png",heatmap)
		cv2.imwrite(self.save_path+"/overlay_heatmap/"+informations+".png",output)
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)

		
