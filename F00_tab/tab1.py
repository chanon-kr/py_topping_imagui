import PySimpleGUI as sg

def tab1_gen() :
    tab1_layout = [ #[sg.Text('Object Detection', size=(40, 1), justification='center', font='Helvetica 20')],
                        # [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
                        # sg.Button('Stop', size=(10, 1), font='Any 14'),
                        # sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ],
                        [sg.Image(filename='', key='image')]]
    # return tab1_layout
    return sg.Tab('Monitor', tab1_layout, font='Courier 15', key='-TAB1-')