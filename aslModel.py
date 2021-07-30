# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:59:21 2020

@author: Shreyash
"""

from keras.models import model_from_json
import numpy as np

class ASLModel(object):

    SIGN_LIST = ['N', 'R', 'A', '6', '4', 'U', '9', 'J', 'Y', 'M', '2', '8', 
                 'L', '1', 'K', '7', '_', 'G', 'E', 'O', 'C', 'P', 'F', 'Z', 
                 'I', '5', 'D', 'B', '0', '3', 'T', 'W', 'Q', 'V', 'H', 'X', 'S']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict(self, img):
        self.preds = self.loaded_model.predict(img)
        return ASLModel.SIGN_LIST[np.argmax(self.preds)]


