import PySimpleGUI as sg

def tab3_gen(input_advance_config) :
    advance_column = []
    for _key1, _item1 in input_advance_config.items():
        if type(_item1) != dict :
            advance_column.append([sg.Text(_key1.capitalize()) , sg.InputText(_item1, key=f'advance||{_key1}')])
        else :
            advance_column.append([sg.Text(_key1.capitalize())])
            for _key2, _item2 in _item1.items() :
                if type(_item2) != dict :
                    advance_column.append([sg.Text(f'   -{_key2.capitalize()}'), sg.InputText(_item2, key=f'advance||{_key1}||{_key2}')])
                else :
                    advance_column.append([sg.Text(f'   -{_key2.capitalize()}')])
                    for _key3, _item3 in _item2.items() :
                        advance_column.append([sg.Text(f'       -{_key3.capitalize()}'), sg.InputText(_item3, key=f'advance||{_key1}||{_key2}||{_key3}')])

    tab3_layout = [[sg.Text('Advance Setting of py-topping-imagui')]
                  ,[sg.Button('SaveConfig' , key = 'UpdateAdConfig', size=(13, 1), font='Helvetica 14'),
                    # sg.Button('ReloadConfig' , key= 'ReloadAdConfig', size=(13, 1), font='Any 14')
                    ]
                  ,[sg.Column(advance_column, scrollable=True,  vertical_scroll_only=True)]]

    # return tab3_layout
    return sg.Tab('Advance Setting', tab3_layout, key='-TAB3-')