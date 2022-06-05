import PySimpleGUI as sg

def tab2_gen(input_df) :
    config_column = []
    for _key1, _item1 in input_df.items():
        if type(_item1) != dict :
            config_column.append([sg.Text(_key1.capitalize()) , sg.InputText(_item1, key=f'config||{_key1}')])
        else :
            config_column.append([sg.Text(_key1.capitalize())])
            for _key2, _item2 in _item1.items() :
                if type(_item2) != dict :
                    config_column.append([sg.Text(f'   -{_key2.capitalize()}'), sg.InputText(_item2, key=f'config||{_key1}||{_key2}')])
                else :
                    config_column.append([sg.Text(f'   -{_key2.capitalize()}')])
                    for _key3, _item3 in _item2.items() :
                        config_column.append([sg.Text(f'       -{_key3.capitalize()}'), sg.InputText(_item3, key=f'config||{_key1}||{_key2}||{_key3}')])

    tab2_layout =   [[sg.Text('Will Be Setting na')]
                    ,[sg.Button('UpdateConfig', size=(13, 1), font='Helvetica 14'),
                        sg.Button('ReloadConfig', size=(13, 1), font='Any 14')]
                    ,[sg.Column(config_column, scrollable=True,  vertical_scroll_only=True)]
                    ]
    # return tab2_layout
    return sg.Tab('Setting', tab2_layout, key='-TAB2-')