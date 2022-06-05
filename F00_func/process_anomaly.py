import cv2, os, torch, shutil, re
from git import Repo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from py_topping.image_processing import lazy_ImageProcessing,  SSIMLoss
from glob import glob

class anomaly_processing :
    def __init__(self, config, advance_config) :
        self.config, self.advance_config = config, advance_config
        self.setup_model( model_name = self.config['model']['model_name']
                        , device_type = self.config['model']['device_type']
                        , size = (128,128))

    def setup_model(self, model_name, size, device_type) :
        print('Set up Model')
        self.model = lazy_ImageProcessing(model_path = model_name, model_type = 'autoencoder', loss_function = SSIMLoss
                                        , custom_objects = {"SSIMLoss": SSIMLoss } , size = size
                                        , normalize_type = np.float32, cvtColor = cv2.COLOR_BGR2GRAY
                                        , normalize_divide = 255, normalize_add = 0
                                        , device = device_type)
    import logging
    logging.getLogger('tensorflow').disabled = True



    def processing_frame(self, img) :
        # results = self.model.predict(img[350:-10,470:-250] , output_type = 'both')
        results = self.model.predict(img , output_type = 'both')
        # frame = np.zeros((img.shape[0]
        #                  ,img.shape[1] + results['image'].shape[1],3),dtype=np.uint8) + 255
        # frame[0:img.shape[0],0:img.shape[1]] = img
        # frame[0:results['image'].shape[0],img.shape[1]:img.shape[1] + results['image'].shape[1]] = results['image']*255
        # frame = cv2.rectangle(frame, (470, 350), (img.shape[1]-250, img.shape[0]-10 ), (0,0,255), 2)
        # v_img = cv2.hconcat([img, v_img])
        df = float(results['loss'])
        return {'frame' : results['image'] , 'info' : df}