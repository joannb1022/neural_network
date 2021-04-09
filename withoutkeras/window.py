import PySimpleGUI as sg
from tensorflow_model import my_model
from train_fun import train, recognize


class InputWindow:

    def __init__(self, default_number):
        self.image_number = default_number
        self.layout = [[sg.Text('Shape Classification', size=(40, 1), justification='center', font='Helvetica 20')],
                       [sg.Text("There are 4 shapes:")],
                       [sg.Text("circle, square, triangle, star")],
                       [sg.Checkbox('Show plots', key="-PLOT-")],
                       [sg.Button('Start', size=(10, 1), font='Helvetica 14')],
                       [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]
        self.interrupted = False
        self.plots = False
        self.model = None
    def show(self):
        window = sg.Window('Main Window', self.layout, location=(800, 400))

        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WIN_CLOSED:
                self.interrupted = True
                break
            if event == "Start":
                model = my_model()
                self.model = model
                self.plots = values["-PLOT-"]
                train(model,self.plots)
                images = recognize(self.model)
                images.show()
                images.close()


                break

        window.close()
