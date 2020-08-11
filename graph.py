from keras.preprocessing import image

from model import DrowningDetectionModel

import numpy as np

model = DrowningDetectionModel("model.json", "model_weights.h5")


class ImageGen(object):
    def __init__(self):
       print("started")

    #predictions
    def get_prediction(self):
        test_im = image.load_img('test/test_t.png', target_size=(300, 300))
        test_im = image.img_to_array(test_im)
        test_im = np.expand_dims(test_im, axis=0)
        pred = model.predict_drowning(test_im)

        return pred
