import json

def format_config(raw_config) :
    # STR
    for i in ["type","job_name"] :
        raw_config[i] = str(raw_config[i])
    # INT
    for i in ["ignore_error", "slot_minute", "upload_minute", "restart_minute","auto_startup"] :
        raw_config[i] = int(raw_config[i])
    # Action STR
    for i in ["line_token"] :
        raw_config['action'][i] = str(raw_config['action'][i])
    # Action INT
    for i in ["alert_frame", "decision_frame"] :
        raw_config['action'][i] = int(raw_config['action'][i])
    # Action FLOAT
    for i in ["alert_conf"] :
        raw_config['action'][i] = float(raw_config['action'][i])
    # Upload STR
    for i in ["db_table", "bucket_folder_name"] :
        raw_config['upload'][i] = str(raw_config['upload'][i])
    # Local Data INT
    for i in ["flush","raw_video_fps","out_video_fps","video_expire_after", "detected_saveforever","save_video_raw","save_video_out"] :
        raw_config["local_data"][i] = int(raw_config["local_data"][i])
    # Local Data STR
    for i in ["temp_table"] :
        raw_config["local_data"][i] = str(raw_config["local_data"][i])
    # Local Data FLOAT
    for i in ["raw_video_size","out_video_size"] :
        raw_config["local_data"][i] = float(raw_config["local_data"][i])
    # Model STR
    for i in ["device_type", "model_name", "model_source"] :
        raw_config["model"][i] = str(raw_config["model"][i])
    # Model INT
    for i in ["predict_size"] :
        raw_config["model"][i] = int(raw_config["model"][i])
    # Model FLOAT
    for i in ["conf", "iou"] :
        raw_config["model"][i] = float(raw_config["model"][i])
    # Video FLOAT
    for i in ["show_size", "y1", "y2", "x1", "x2"] :
        raw_config['video'][i] = float(raw_config['video'][i])
    # Video INT
    for i in ["show_fps"] :
        raw_config['video'][i] = int(raw_config['video'][i])
    # video_source
    if len(raw_config['video']['video_source']) == 1 : 
        raw_config['video']['video_source'] = int(raw_config['video']['video_source'])
    else : raw_config['video']['video_source'] = str(raw_config['video']['video_source'])
    # Model Class
    raw_config["model"]['class_detect'] = [int(x) for x in raw_config["model"]['class_detect'].split(',') if x != ""]
    # AIM INT
    for i in ["only_crosshair", "active", "checkaim", "crosshair_size", "crosshair_thickness", "arrow_thick", "box_thickness", "box_clearance"] :
        raw_config["aim"][i] = int(raw_config["aim"][i])
    # AIM FLOAT
    for i in ["xtarget", "ytarget", "xallowance", "yallowance"] :
        raw_config["aim"][i] = min(1,float(raw_config["aim"][i]))
    # AIM RGB
    for i in ["object_crosshair_color", "target_crosshair_color", "arrow_color", "box_color"] :
        for j in ['R','G','B'] : raw_config["aim"][i][j] = int(raw_config["aim"][i][j])
    return raw_config

def format_advance(raw_config) : 
    raw_config["framework"]["local_framework"] = int(raw_config["framework"]["local_framework"])
    raw_config["framework"]["force_reload"] = int(raw_config["framework"]["force_reload"])
    return raw_config

def load_config() :
    with open('F01_config/config.json', 'r') as f :
        basic_config = format_config(json.loads(f.read()))
    with open('F01_config/advance_config.json', 'r') as f :
        advance_config = format_advance(json.loads(f.read()))
    return basic_config, advance_config

def update_config(gui_dict, gui_key , file_name) :
    buffer_dict = {x.split(f'{gui_key}||')[-1] : y for x, y in gui_dict.items() if x.startswith(f'{gui_key}||')}
    out_dict = {}
    for i, j in buffer_dict.items() :
        buffer = i.split('||')
        if len(buffer) == 1 :
            out_dict[buffer[0]] = j
        elif len(buffer) == 2 :
            out_dict[buffer[0]] = out_dict.get(buffer[0],{})
            out_dict[buffer[0]][buffer[1]] = j
        elif len(buffer) == 3 :
            out_dict[buffer[0]] = out_dict.get(buffer[0],{})
            out_dict[buffer[0]][buffer[1]] = out_dict[buffer[0]].get(buffer[1],{})
            out_dict[buffer[0]][buffer[1]][buffer[2]]  = j
    with open(file_name, 'w') as f :
        f.write(json.dumps(out_dict, indent=4))