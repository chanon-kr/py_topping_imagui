import cv2, os, torch, shutil, re
from git import Repo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from py_topping.general_use import lazy_LINE, healthcheck, timeout
from py_topping.data_connection.database import lazy_SQL
from py_topping.data_connection.gcp import lazy_GCS
from glob import glob

class yolo_processing :
    def __init__(self, config, advance_config) :
        self.config, self.advance_config = config, advance_config
        if self.config['aim']['checkaim'] :
            pass
        else :
            self.setup_model( model_name = self.config["model"]['model_name']
                            , force_reload = self.advance_config['framework']['force_reload']
                            , device_type = self.config["model"]['device_type']
                            , conf = float(self.config["model"]['conf'])
                            , iou = float(self.config["model"]['iou'])
                            , class_detect = self.config["model"]['class_detect']
                            , local_framework  = self.advance_config['framework']['local_framework'] )

    @timeout(30)
    def update_local_framework(self) :
        if not os.path.isdir('ultralytics') : os.makedirs(os.path.join(os.getcwd(),'ultralytics'))
        os.chdir('ultralytics')
        if os.path.isdir('yolov5') : 
            print('Update YOLOv5 from Ultralytics')
            os.chdir('yolov5')
            repo = Repo()
            repo.remotes.origin.pull()
        else :
            print('Clone YOLOv5 from Ultralytics')
            yolov5_url = 'https://github.com/ultralytics/yolov5.git'
            Repo.clone_from(yolov5_url , os.path.join(os.getcwd(),'yolov5'))
            
    def setup_model(self, model_name, force_reload, device_type, conf, iou, class_detect, local_framework) :
        print('Set up Model')
        if local_framework :
            base_dir = os.getcwd()
            try : self.update_local_framework()
            except : print('Update Fail')
            os.chdir(base_dir)
            self.model = torch.hub.load(os.path.join(base_dir,'ultralytics','yolov5'), 'custom'
                                    , path = model_name, source ='local'
                                    , force_reload = force_reload, device = device_type) 
        else :
            self.model = torch.hub.load('ultralytics/yolov5' , 'custom', path = model_name
                                    , force_reload = force_reload, device = device_type) 
        self.model.conf, self.model.iou = conf, iou 
        if class_detect != [] : self.model.classes = class_detect
        # Aim will only get 1 Target
        if self.config['aim']['active'] : self.model.max_det = 1
    
    def aim(self, frame, yolo_df) :
        xtarget = self.config['aim']['xtarget']*frame.shape[1]
        ytarget = self.config['aim']['ytarget']*frame.shape[0]
        xallowance = self.config['aim']['xallowance']*frame.shape[1]
        yallowance = self.config['aim']['yallowance']*frame.shape[0]
        object_crosshair_color = (self.config['aim']['object_crosshair_color']['B'],self.config['aim']['object_crosshair_color']['G'],self.config['aim']['object_crosshair_color']['R'])
        target_crosshair_color = (self.config['aim']['target_crosshair_color']['B'],self.config['aim']['target_crosshair_color']['G'],self.config['aim']['target_crosshair_color']['R'])
        crosshair_size = self.config['aim']['crosshair_size']
        crosshair_thickness = self.config['aim']['crosshair_thickness']
        arrow_thick = self.config['aim']['arrow_thick']
        arrow_color = (self.config['aim']['arrow_color']['R'], self.config['aim']['arrow_color']['G'], self.config['aim']['arrow_color']['B'])
        box_color = (self.config['aim']['box_color']['R'], self.config['aim']['box_color']['G'], self.config['aim']['box_color']['B'])
        box_thickness = self.config['aim']['box_thickness']
        box_clearance = self.config['aim']['box_clearance']
        process_frame, self.target_lock = frame.copy(), True
        # Central of Target
        cv2.line(process_frame, (int(xtarget - crosshair_size), int(ytarget))
                        , (int(xtarget + crosshair_size), int(ytarget))
                        , target_crosshair_color, crosshair_thickness)
        cv2.line(process_frame, (int(xtarget), int(ytarget - crosshair_size))
                        , (int(xtarget), int(ytarget + crosshair_size))
                        , target_crosshair_color, crosshair_thickness)
        cv2.rectangle(process_frame, (int(xtarget + xallowance), int(ytarget + yallowance))
                    , (int(xtarget - xallowance), int(ytarget - yallowance)), target_crosshair_color, 1)
        if len(yolo_df) <= 0 & ~self.config['aim']['checkaim'] : return process_frame
        # Central of Object
        xcen, ycen = yolo_df[['xmin','xmax']].loc[0].mean(), yolo_df[['ymin','ymax']].loc[0].mean()
        cv2.line(process_frame, (int(xcen - crosshair_size), int(ycen))
                            , (int(xcen + crosshair_size), int(ycen)), object_crosshair_color, crosshair_thickness)
        cv2.line(process_frame, (int(xcen), int(ycen - crosshair_size))
                            , (int(xcen), int(ycen + crosshair_size)), object_crosshair_color, crosshair_thickness)
        # Check in X Axis
        if xcen > xtarget + xallowance : 
            # X Direction to Left
            pts = [(int(yolo_df.loc[0,'xmin']), int(yolo_df.loc[0,'ymin']))
                , (int(yolo_df.loc[0,'xmin']), int(yolo_df.loc[0,'ymax']))
                , (int(yolo_df.loc[0,'xmin'] - arrow_thick), int(ycen))]
            cv2.fillPoly(process_frame, np.array([pts]), arrow_color)
            self.target_lock = False
        elif xcen < xtarget - xallowance : 
            # X Direction to Right
            pts = [(int(yolo_df.loc[0,'xmax']), int(yolo_df.loc[0,'ymin']))
                , (int(yolo_df.loc[0,'xmax']), int(yolo_df.loc[0,'ymax']))
                , (int(yolo_df.loc[0,'xmax'] + arrow_thick), int(ycen))]
            cv2.fillPoly(process_frame, np.array([pts]), arrow_color)
            self.target_lock = False
        else : 
            # X Direction Ok
            cv2.line(process_frame, (int(yolo_df.loc[0,'xmin'] - box_clearance), int(yolo_df.loc[0,'ymin'] - box_clearance))
                                , (int(yolo_df.loc[0,'xmin'] - box_clearance), int(yolo_df.loc[0,'ymax'] + box_clearance))
                                , box_color, box_thickness)
            cv2.line(process_frame, (int(yolo_df.loc[0,'xmax'] + box_clearance), int(yolo_df.loc[0,'ymin'] - box_clearance))
                                , (int(yolo_df.loc[0,'xmax'] + box_clearance), int(yolo_df.loc[0,'ymax'] + box_clearance))
                                , box_color, box_thickness)
        # Check in Y Axis
        if ycen > ytarget + yallowance : 
            # Y Direction to Up
            pts = [(int(yolo_df.loc[0,'xmax']), int(yolo_df.loc[0,'ymin']))
                , (int(yolo_df.loc[0,'xmin']), int(yolo_df.loc[0,'ymin']))
                , (int(xcen), int(yolo_df.loc[0,'ymin'] - arrow_thick))]
            cv2.fillPoly(process_frame, np.array([pts]), arrow_color)
            self.target_lock = False
        elif ycen < ytarget - yallowance : 
            # Y Direction to Down
            pts = [(int(yolo_df.loc[0,'xmax']), int(yolo_df.loc[0,'ymax']))
                , (int(yolo_df.loc[0,'xmin']), int(yolo_df.loc[0,'ymax']))
                , (int(xcen), int(yolo_df.loc[0,'ymax'] + arrow_thick))]
            cv2.fillPoly(process_frame, np.array([pts]), arrow_color)
            self.target_lock = False
        else : 
            # Y Direction Ok
            cv2.line(process_frame, (int(yolo_df.loc[0,'xmin'] - box_clearance), int(yolo_df.loc[0,'ymin'] - box_clearance))
                                , (int(yolo_df.loc[0,'xmax'] + box_clearance), int(yolo_df.loc[0,'ymin'] - box_clearance))
                                , box_color, box_thickness)
            cv2.line(process_frame, (int(yolo_df.loc[0,'xmin'] - box_clearance), int(yolo_df.loc[0,'ymax'] + box_clearance))
                                , (int(yolo_df.loc[0,'xmax'] + box_clearance), int(yolo_df.loc[0,'ymax'] + box_clearance))
                                , box_color, box_thickness)
        if self.target_lock :
            label = 'OK'
            text_location = (int(yolo_df.loc[0,'xmin']), int(yolo_df.loc[0,'ymin'] - 10 ))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) 
            cv2.rectangle(process_frame , (int(yolo_df.loc[0,'xmin'] ), int(yolo_df.loc[0,'ymin'] - labelSize[1] - 10 ))
                                        , (int(yolo_df.loc[0,'xmin'])+labelSize[0], int(yolo_df.loc[0,'ymin'] + baseLine-10))
                                        , box_color, cv2.FILLED) 
            cv2.putText(process_frame, 'OK' ,text_location, cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2,cv2.LINE_AA)
        return process_frame

    def processing_frame(self, raw_img) :
        img = raw_img.copy()
        if self.config['aim']['checkaim'] :
            frame, df = img, []
        else : 
            results = self.model(img, size = self.config["model"]['predict_size'])
            frame, df = results.render()[0] , results.pandas().xyxy[0]
            df['target_lock'] = 0
        if self.config['aim']['active'] : 
            if self.config['aim']['only_crosshair'] : 
                frame = self.aim(raw_img, df)
            else : 
                frame = self.aim(frame, df)
            df['target_lock'] += self.target_lock
        return {'frame' : frame, 'info' : df}