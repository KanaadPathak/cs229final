'''Find input images that maximize the CNN codes
based on Kera's example conv_filter_visualization
and adopted new regularization from https://arxiv.org/pdf/1506.06579v1.pdf
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import resnet50
from keras import backend as K
import os
import sys
import math


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def process(path):
    # dimensions of the generated pictures for each filter.
    img_width = 128
    img_height = 128
    stage = 0 # 0 for test


    K.set_learning_phase(stage)
    model = resnet50.ResNet50(include_top=False)
    model.summary()

    K.set_learning_phase(stage)

    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    # the name of the layer we want to visualize
    layer_name = 'activation_49'
    #misnomer: this is the input images that maximize
    kept_filters = []
    for filter_index in range(0, 20):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the loss wrt to input picture (variables)
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
        K.set_learning_phase(stage)

        # step size for gradient ascent
        step = 1.

        #initialize with random
        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            xx = np.random.random((1, 3, img_width, img_height))
        else:
            xx = np.random.random((1, img_width, img_height, 3))
        xx = (xx - 0.5) * 20 + 128

        #for regularization
        decay = 0.01
        blur_radius = 0.3
        lower_percentile = 0.1
        norm_percentile = 0.1
        # we run gradient ascent for 20 steps
        for i in range(20):
            #get gradient and object function
            loss_value, grads_value = iterate([xx])
            #update and apply regularization
            #L2 norm: x <= x + nebla * learning_rate - decay * x
            xx = xx *(1-decay) + grads_value * step
            #Gaussian blur
            from scipy.ndimage import gaussian_filter
            for channel in range(3):
                cimg = gaussian_filter(xx[0,channel], blur_radius)
                xx[0,channel] = cimg

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(xx[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best 64 filters on a 8 x 8 grid.
    #n = math.sqrt(len(kept_filters))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    #kept_filters.sort(key=lambda x: x[1], reverse=True)
    #kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    #margin = 5
    #width = n * img_width + (n - 1) * margin
    #height = n * img_height + (n - 1) * margin
    #stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    #for i in range(n):
    #    for j in range(n):
    #        img, loss = kept_filters[i * n + j]
    #        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
    #                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    for i in range(len(kept_filters)):
        # save the result to disk
        img, loss = kept_filters[i]
        imsave(os.path.join(path, '%d.jpg'%(i)), img)

if __name__ == '__main__':
    assert len(sys.argv) == 2, "output_path"
    output_path = sys.argv[1]
    process(output_path)
