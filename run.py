import PySimpleGUI as sg
# import PySimpleGUIWeb as sg
import cv2
from F00_func.input_format import *
from F00_func.process_utility import *
from F00_func.camera import *
from datetime import datetime

def main():    
    basic_config, advance_config = load_config()
    previous_tstamp, fps_list, target_fps = datetime.now(), [], 15
    # Set Up Theme
    sg.theme('Black')
    # Define the window layout
    from F00_tab.tab1 import tab1_gen
    from F00_tab.tab2 import tab2_gen
    from F00_tab.tab3 import tab3_gen
    tab1_layout, tab2_layout, tab3_layout = tab1_gen(), tab2_gen(basic_config), tab3_gen(advance_config)
    # Put in Layout
    tab_group_layout = [[tab1_layout,tab2_layout,tab3_layout]]
    layout = [[sg.Button('Start', size=(10, 1), font='Helvetica 12'),
               sg.Button('Stop', size=(10, 1), font='Helvetica 12'),
               sg.Button('Update', size=(10, 1), font='Helvetica 12'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 12'), ],
              [sg.TabGroup(tab_group_layout,enable_events=True,key='-TABGROUP-')],
              [sg.Text('Version', justification='right')]]

    # create the window and show it without the plot
    window = sg.Window('Object Detection', layout 
                     , location=(0, 0), finalize=True, resizable=True) 

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    recording = False
    if basic_config['auto_startup'] : window['Start'].click()
    while True:
        event, values = window.read(timeout=1)
        
        if event == 'Update' :
            window['Stop'].click()
            update_yolov5_pipe()
            window['Exit'].click()

        elif event == 'UpdateAdConfig' : 
            update_config(values, 'advance' , 'F01_config/advance_config.json')
            sg.popup_auto_close('Update Complete')

        elif event == 'UpdateConfig' : 
            update_config(values, 'config' , 'F01_config/config.json')
            sg.popup_auto_close('Update Complete')

        elif event == 'Exit' or event == sg.WIN_CLOSED : return

        elif event == 'Stop':
            recording = False
            window['image'].update(filename='')
            cap.release(), utility_model.release()

        elif event == 'Start':
            basic_config, advance_config = load_config()
            sg.Print('Connecting Camera', font = '24')
            cap, recording = Camera(basic_config['video']['video_source']), True
            sg.Print('Create Utility')
            utility_model = utility_processing(basic_config, advance_config, cap.additional_config)
            sg.Print('Create Model')
            model = create_model_object(basic_config, advance_config)
            sg.PrintClose()

        if recording:
            current_tstamp = datetime.now()
            # Grab Frame
            ret, frame = cap.getFrame()
            if not ret : 
                print('Camera Error')
                continue
            # Cal FPS
            previous_tstamp = utility_model.cal_fps(current_tstamp, previous_tstamp, show = False)
            # Process Image
            preprocessed = utility_model.pre_processing(frame)
            processed = model.processing_frame(preprocessed)
            cooked = utility_model.post_processing(frame, processed)
            # 
            utility_model.household()
            # Show Result
            imgbytes = cv2.imencode('.png', cooked)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

if __name__ == '__main__' : 
    main()