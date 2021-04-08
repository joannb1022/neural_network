import PySimpleGUI as sg
from keras import Model
from preparedata import create_data, show_random_shapes, create_data_with_labels
from withoutkeras.data import Data


class InputWindow:

    def __init__(self, default_number, model):
        self.image_number = default_number
        self.layout = [[sg.Text('Shape Classification', size=(40, 1), justification='center', font='Helvetica 20')],
                       [sg.Text("There are 4 shapes:")],
                       [sg.Text("circle, square, triangle, star")],
                       [sg.Button('Start', size=(10, 1), font='Helvetica 14')],
                       [sg.Button('Exit', size=(10, 1), font='Helvetica 14')]]
        self.interrupted = False
        self.plots = False
        self.model = model

    def show(self):
        window = sg.Window('Main Window', self.layout, location=(800, 400))

        while True:
            event, values = window.read()
            if event == 'Exit' or event == sg.WIN_CLOSED:
                self.interrupted = True
                break
            if event == "Start":
                categories = ["square", "circle", "triangle", "star"]
                paint_path = "C:/Users/chleb/Desktop/neural_network-master/paint_data"
                img_size = 32
                train_data = create_data(categories, img_size, paint_path)
                td, _ = create_data_with_labels(train_data, img_size)
                td = td / 255.0
                print(td.shape)
                print(td[0].shape)
                break

        window.close()
