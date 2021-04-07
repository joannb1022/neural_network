import PySimpleGUI as sg
from keras import Model
from preparedata import create_data, create_training_set, show_random_shapes

class InputWindow:

    def __init__(self, default_number):
        self.image_number = default_number
        self.layout = [[sg.Text('Shape Classification', size=(40, 1), justification='center', font='Helvetica 20')],                       [sg.Text("Pick number of shapes :")],
                       [sg.Spin([i for i in range(1, 20, 1)], initial_value=4, key="-NUM-")],
                       [sg.Text("Or pick image:"), sg.FileBrowse()],
                       [sg.Checkbox('Show plots', key="-PLOT-")],
                       [sg.Button('Train', size=(10, 1),  font='Helvetica 14')],
                       [sg.Button('Recognize', size=(10, 1), font='Helvetica 14')],
                       [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]
        self.interrupted = False
        self.plots = False

    def show(self):
        window = sg.Window('Main Window', self.layout, location=(800, 400))


        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WIN_CLOSED:
                self.interrupted = True
                break
            if event == "Train":
                model = Model()
                model.train()
            if event == "Recognize":
                self.image_number = values["-NUM-"]
                if model.model == None:
                    print("nieee")
                    return 0
                model.predict(self.image_number)
                self.plots = values["-PLOT-"]
                break



        window.close()
