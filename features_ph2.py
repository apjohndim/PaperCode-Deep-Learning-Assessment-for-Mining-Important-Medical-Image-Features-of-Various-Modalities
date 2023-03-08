import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_names, classifier_layer_names
):
    
    
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)



    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


#%%
name = 'ISIC_0000426.jpg'
imagePath = 'E:\\Data (Biomarkers)\\SKIN\\nv\\{}'.format(name)
preprocess_input = keras.applications.mobilenet.preprocess_input
img_array = preprocess_input(get_img_array(imagePath, size=(200, 150)))



last_conv_layer_name = "block5_conv3"
classifier_layer_names = ['global_average_pooling2d_62','dense_34', 'batch_normalization_62', 'dropout_62', 'dense_35' ]


heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names)


heatmap= np.uint8(255 * heatmap)
jet = cm.get_cmap("coolwarm")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]


img = cv2.imread(imagePath)
img = cv2.resize(img, (200, 150))
# We create an image with RGB colorized heatmap

big_heatmap = cv2.resize(jet_heatmap, dsize=(200, 150), 
                         interpolation=cv2.INTER_CUBIC)

big_heatmap = keras.preprocessing.image.img_to_array(big_heatmap)

jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)





jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)



# Superimpose the heatmap on original image
superimposed_img = big_heatmap * 300 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = 'C:\\Users\\User\\ZZZ. ARE THEY BIOMARKERS\\Reproduction Code\\{}.jpg'.format(name)
superimposed_img.save(save_path)


# Display Grad CAM
from IPython.display import Image
display(Image(save_path))


#%%
from matplotlib import pyplot
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model2 = tf.keras.Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape


img =get_img_array(imagePath, size=(200, 150))
# expand dimensions so that it represents a single 'sample'
# prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model2.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gist_gray')
			ix += 1
	# show the figure
	pyplot.show()


#%%