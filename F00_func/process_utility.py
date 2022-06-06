import cv2, os, torch, shutil, re
from git import Repo
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from py_topping.general_use import lazy_LINE, healthcheck, timeout
from py_topping.data_connection.database import lazy_SQL
from py_topping.data_connection.gcp import lazy_GCS
from glob import glob
from git import Repo

@timeout(30)
def update_repository() :
    repo = Repo()
    repo.remotes.origin.pull()

def create_model_object(basic_config, advance_config) :
    if basic_config['type'] == 'objectdetection' : from F00_func.process_yolo import yolo_processing as processing 
    elif basic_config['type'] == 'anomaly' : from F00_func.process_anomaly import anomaly_processing as processing
    return processing(config = basic_config, advance_config = advance_config)                

class utility_processing :
    def __init__(self, config, advance_config, additional_config) :
        self.config, self.advance_config, self.additional_config = config, advance_config, additional_config
        self.fps_list, self.fps = [], 0
        self.info_list, self.frame_no,self.action, self.found = [], 0, 0, 0
        if self.config["type"] == "objectdetection" : self.info_list = pd.DataFrame()
        # Crop Area
        y1, y2 = config['video']['y1'], 1 - config['video']['y2']
        assert (y1 < y2), 'Please recheck y1,y2'
        x1, x2 = config['video']['x1'], 1 - config['video']['x2']
        assert (x1 < x2), 'Please recheck x1,x2'
        self.y1, self.y2 = int(additional_config['height']*y1),int(additional_config['height']*y2)
        self.x1, self.x2 = int(additional_config['width']*x1),int(additional_config['width']*x2)
        # Cal Resize
        self.resize_size = {}
        self.resize_size['raw'] = (int(additional_config['width']*config['local_data']['raw_video_size']) , int(additional_config['height']*config['local_data']['raw_video_size']))
        self.resize_size['out'] = (int(additional_config['width']*config['local_data']['out_video_size']) , int(additional_config['height']*config['local_data']['out_video_size']))
        self.resize_size['show'] = (int(additional_config['width']*config['video']['show_size']) , int(additional_config['height']*config['video']['show_size']))
        # Create Video
        self.init_video()

    def init_video(self) :    
        # Save Video
        masterpath, self.video_write = os.getcwd(), {}
        self.video_init_time = datetime.now()
        video_init_time, video_init_date = self.video_init_time.strftime('%Y_%m_%d_%H_%M'), self.video_init_time.strftime('%Y_%m_%d')
        for i in ['raw','out'] :
            if self.config['local_data'][f'save_video_{i}'] :
                directory = os.path.join(masterpath,'F03_clip',i)
                if not os.path.isdir(directory) : os.mkdir(directory)
                directory = os.path.join(directory,video_init_date)
                if not os.path.isdir(directory) : os.mkdir(directory)
                save_path = os.path.join(directory,f'{video_init_time}.avi')
                self.video_write[i] = cv2.VideoWriter(save_path
                                                    , cv2.VideoWriter_fourcc(*'DIVX')  
                                                    , self.config['local_data'][f'{i}_video_fps']  
                                                    , self.resize_size[i]) 
    
    def release(self) :
        for i in ['raw','out'] :
            if self.config['local_data'][f'save_video_{i}'] : 
                self.video_write[i].release()

    def pre_processing(self, raw_img) :
        self.save_video(raw_img, 'raw')
        out_img = self.send_crop_picture(raw_img)
        return out_img

    def post_processing(self, raw_img, processed) :
        processed_img = processed['frame']
        if self.config["type"] == "objectdetection" : 
            out_img = self.replace_crop_picture(raw_img, processed_img)
        elif self.config["type"] == "anomaly" : 
            out_img = self.combine_picture(raw_img, processed_img, info = processed['info'])
        out_img = self.insert_fps(out_img)
        self.save_video(out_img, 'out')
        out_img = self.resize_output(out_img, 'show')
        self.check_action(processed, out_img)
        # if self.action : self.take_action(processed, out_img)
        self.frame_no += 1
        return out_img          

    def household(self) :
        time_pass = (datetime.now() - self.video_init_time).total_seconds()/60
        if time_pass >= self.config['slot_minute'] :
            self.release()
            # Upload
            # Move File
            # Delete Old File
            self.init_video()

    def check_action(self, processed, last_img) :
        if self.config["type"] == "objectdetection" : 
            df_ = processed['info']
            df_['frame_no'] = self.frame_no
            if self.info_list.shape[0] > 0 : 
                self.found += 1
                self.info_list = self.info_list[self.info_list['frame_no'] >= self.frame_no - self.config['action']['decision_frame']]
                self.info_list = pd.concat([self.info_list,df_], axis = 0,ignore_index=True)
                if self.config["aim"]['active'] :
                    self.action = sum(self.info_list['target_lock']) >= self.config['action']['alert_frame'] 
                    if self.action :
                        if self.config['action']['send_line']  : 
                            cv2.imwrite('send_noti.jpg' , last_img)
                            line = lazy_LINE(self.config['action']['line_token'])
                            line.send(f"\n---\n{self.config['job_name']}\n---\n{df_['name'].iloc[0]} in Position", picture = 'send_noti.jpg')     
                            # Flush DF
                            self.info_list = pd.DataFrame()
                else :
                    for i in list(df_['name'].unique()) :
                        frame_found = ((self.info_list['name'] == i) & (self.info_list['confidence'] >= self.config['action']['alert_conf'])).sum()
                        if frame_found >= self.config['action']['alert_frame'] :
                            # Save Image
                            if self.config['action']['send_line']  : 
                                cv2.imwrite('send_noti.jpg' , last_img)
                                line = lazy_LINE(self.config['action']['line_token'])
                                line.send(f"\n---\n{self.config['job_name']}\n---\nFound {i}", picture = 'send_noti.jpg')     
                            # Flush DF
                            self.info_list = self.info_list[self.info_list['name'] != i]
        elif self.config["type"] == "anomaly" : 
            if self.action : self.info_list = []
            self.info_list.append(processed['info'])
            self.info_list = self.info_list[-self.config['action']['decision_frame']:]
            alerted_frame = sum(x > self.config['action']['alert_conf'] for x in self.info_list) 
            self.action = alerted_frame >= self.config['action']['alert_frame']
            if self.action :
                if self.config['action']['send_line']  : 
                    cv2.imwrite('send_noti.jpg' , last_img)
                    line = lazy_LINE(self.config['action']['line_token'])
                    line.send(f"\n---\n{self.config['job_name']}\n---\nFound anomaly", picture = 'send_noti.jpg')    

    def resize_output(self, raw_img, mode) :
        out_img = raw_img.copy()
        out_img = cv2.resize(out_img, self.resize_size[mode])
        return out_img

    def save_video(self, raw_img, type) :
        if self.config['local_data'][f'save_video_{type}'] : 
            save_img = self.resize_output(raw_img, type)
            self.video_write[type].write(save_img)

    def insert_fps(self, raw_img) :
        out_img = raw_img.copy()
        text_location = (int(self.additional_config['width'] - 100),int(self.additional_config['height'] - 10))
        cv2.putText(out_img, str(self.fps) ,text_location, cv2.FONT_HERSHEY_SIMPLEX , 1,(0,255,0),2,cv2.LINE_AA)
        return out_img

    def cal_fps(self, current, previous, show = False):
        fps_cal = (current - previous).total_seconds()
        fps_cal = 60 if fps_cal == 0 else 1/fps_cal
        self.fps_list.append(fps_cal)
        if len(self.fps_list) >= 10 : self.fps_list = self.fps_list[1:]
        self.fps = round(np.mean(self.fps_list),2)
        return datetime.now()

    def send_crop_picture(self, raw_img) :
        return raw_img[self.y1:self.y2, self.x1:self.x2]

    def replace_crop_picture(self, raw_img, processed_img) :
        out_img = raw_img.copy()
        out_img[self.y1:self.y2, self.x1:self.x2] = processed_img
        cv2.rectangle(out_img, (self.x2,self.y2), (self.x1,self.y1), (0, 255, 255), 2)
        return out_img

    def combine_picture(self, raw_img, processed_img, info) :
        raw_y, raw_x, _ = raw_img.shape
        out_img = np.zeros((raw_y
                         ,raw_x + 10 + processed_img.shape[1],3),dtype=np.uint8) + 255
        out_img[0:raw_y,0:raw_x] = raw_img
        out_img[0:processed_img.shape[0],raw_x + 10:raw_x + 10 + processed_img.shape[1]] = processed_img*255
        out_img = cv2.rectangle(out_img, (self.x2,self.y2), (self.x1,self.y1), (0, 255, 255), 2)
        text_location = (int(raw_x + 10), int(processed_img.shape[0] - 5))
        cv2.putText(out_img, str(round(info,2)) ,text_location, cv2.FONT_HERSHEY_SIMPLEX , 1,(0,0,255),2,cv2.LINE_AA)
        return out_img

def send_heartbeat(process_name_in, table_name_in, heart_beat_config_in, ignore_error_in = False, job_name_in = '', message_in = '') :
    des_sql = lazy_SQL(sql_type = heart_beat_config_in['type'] 
                       , host_name = heart_beat_config_in['host_name']
                       , database_name = heart_beat_config_in['db_name'] 
                       , user = heart_beat_config_in['user']
                       , password = heart_beat_config_in['password'] 
                       , mute = True)
    df_sql = pd.DataFrame(healthcheck())
    df_sql['t_stamp'] = datetime.now()
    df_sql['process_name'], df_sql['error_message'] = process_name_in, message_in
    df_sql['job_name'] = job_name_in
    try : des_sql.sub_dump(df_sql , table_name_in ,'append')
    except Exception as e: 
        if not ignore_error_in : raise Exception(e)
        print(e)

def modify_df(df_in, now_in, fps_in, frame_no_in, start_time_in, slot_time_in, job_name_in, area_detect_in) :
    df_out = df_in.copy()
    x1_in, y1_in = area_detect_in[1]
    df_out['t_stamp'], df_out['fps'] = now_in.strftime('%Y-%m-%d %H:%M:%S'), fps_in
    df_out['frame_no'], df_out['start_time'] = frame_no_in, start_time_in
    df_out['slot_time'], df_out['job_name'] = slot_time_in, job_name_in
    for i_ in ['xmin','xmax'] : df_out[i_] += x1_in
    for i_ in ['ymin','ymax'] : df_out[i_] += y1_in
    df_out['area'] = str(area_detect_in)
    return df_out

def send_LINE(target_in , line_token_in, path_in) :
    print('Send Line Message\n---\n{}'.format(target_in))
    line = lazy_LINE(line_token_in)
    line.send(target_in, picture = path_in)

def flush_old(flush, local_record_config_in, temp_table_in) :
    if flush : 
        clean_folder = glob('F03_clip/*.*')
        for i in clean_folder : os.remove(i)
        sqlite = lazy_SQL(sql_type = local_record_config_in['type'] 
                        , host_name = local_record_config_in['host_name']
                        , database_name = '', user = '', password = '' 
                        , mute = True)
        sqlite.engine.execute("DROP TABLE IF EXISTS {}".format(temp_table_in))

def record_result(df_in, temp_table_in , local_record_config_in ) :
    df_out = df_in.copy()
    sqlite = lazy_SQL(sql_type = local_record_config_in['type'] 
                      , host_name = local_record_config_in['host_name']
                      , database_name = '', user = '', password = '' 
                      , mute = True)
    sqlite.main_dump(df_out , temp_table_in,'append')


def upload_result(temp_table_in , now_in, db_table_in, local_record_config_in, db_record_config_in, ignore_error_in = False) :
    sqlite = lazy_SQL(sql_type = local_record_config_in['type']
                      , host_name = local_record_config_in['host_name']
                      , database_name = '',user = '',password = ''
                      , mute = True )
    des_sql = lazy_SQL(sql_type = db_record_config_in['type'] 
                       , host_name = db_record_config_in['host_name']
                       , database_name = db_record_config_in['db_name'] 
                       , user = db_record_config_in['user']
                       , password = db_record_config_in['password']
                       , mute = True )
    date_query = [(now_in + timedelta(hours = i)).strftime('%Y-%m-%d %H') for i in [-3,-2,-1,0,1]]
    date_query = str(tuple(date_query))
    condition_statement = """FROM {} WHERE SUBSTR(slot_time ,1, 13) IN {}""".format(temp_table_in, date_query)
    df_sql = sqlite.read("""SELECT * {}""".format(condition_statement), raw = True )
    print('Start Upload Database')
    try :
        des_sql.main_dump(df_sql , db_table_in,'append')
        sqlite.engine.execute("""DELETE {}""".format(condition_statement))
    except Exception as e: 
        if not ignore_error_in : raise Exception(e)
        print(e)

def remove_uploaded_file(uploaded_folder, video_expire_after) :
    video_file = glob('{}/*'.format(uploaded_folder))
    date_file = [re.search(r'_([\d]+)_',i).group(1) for i in video_file]
    date_file = [(datetime.now() - datetime.strptime(i, '%Y%m%d')).days for i in date_file]
    date_file = [i >= video_expire_after for i in date_file]
    video_file = [i for (i, v) in zip(video_file, date_file) if v]
    for i_ in video_file : os.remove(i_)

def upload_clip(video_folder_in, current_video_in, bucket_folder_name_in, storage_config_in, ignore_error_in = False, video_expire_after = 5, turn_on_upload_clip = True) :
    bucket_config_in = storage_config_in['gcs']
    # Create Connection
    # print(bucket_config_in) ## For Debug
    if turn_on_upload_clip : 
        gcs = lazy_GCS(project_id = bucket_config_in['project_id']
                    , bucket_name = bucket_config_in['bucket_name']
                    , credential = bucket_config_in['credential'])
    # Create Uploaded Folder
    uploaded_folder = os.path.join(video_folder_in,'uploaded')
    if not os.path.isdir(uploaded_folder) : os.makedirs(os.path.join(os.getcwd(),uploaded_folder))
    # Create Video List
    video_file = glob('{}/*.*'.format(video_folder_in))
    if current_video_in in video_file : video_file.remove(current_video_in)
    np.random.shuffle(video_file)
    folder_len = 1 + len(video_folder_in)
    # Main Upload Loop
    print('Start Upload Video')
    for i_ in video_file :
        bucket_file_name = '{}/{}'.format(bucket_folder_name_in , i_[folder_len:])
        try :
            # print(bucket_file_name) ## For Debug
            if turn_on_upload_clip : gcs.upload(bucket_file = bucket_file_name , local_file = i_)
            shutil.move(i_ , i_.replace(video_folder_in, uploaded_folder))
        except Exception as e: 
            if not ignore_error_in : raise Exception(e)
            print(e)
    # Remove File
    print('Upload Complete\nDelete Expired Clip(s)')  
    remove_uploaded_file(uploaded_folder, video_expire_after)
            
def update_model(model_source_in , model_name_in, model_source_config_in) :
    gcs = lazy_GCS(project_id = model_source_config_in['gcs']['project_id']
                   , bucket_name = model_source_config_in['gcs']['bucket_name']
                   , credential = model_source_config_in['gcs']['credential'])
    gcs.download(bucket_file = model_source_in, local_file = model_name_in)