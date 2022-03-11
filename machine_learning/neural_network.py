import tkinter as tk
from tkinter import ttk
from tools.functions import deserialize, update_entry, save_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


class NNView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.geometry('500x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        frm = tk.LabelFrame(self)
        frm.pack(fill=tk.X)
        tk.Label(frm, text='Пожалуйста, выберите решаемую задачу:').pack()
        tasks = ['Классификация', 'Прогнозирование']
        self.task_cb = ttk.Combobox(frm, values=tasks)
        self.task_cb.current(0)
        self.task_cb.pack(pady=5)
        ttk.Button(frm, text='Выбрать', command=self.get_task).pack(pady=3)
        classification_frm = ClassificationFrame(self, self.entry, self.pd_data)
        forecasting_frm = ForecastingFrame(self, self.entry, self.pd_data)
        self.tasks = dict(zip(tasks, [classification_frm, forecasting_frm]))
        self.current = None
        self.get_task()

    def get_task(self):
        task = self.task_cb.get()
        self.title(f'Построение нейронных сетей. {task}.')
        if self.current is not None:
            self.current.pack_forget()
        self.tasks[task].pack(fill=tk.BOTH, expand=1)
        self.current = self.tasks[task]


class ClassificationFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        pad = {
            'padx': 5,
            'pady': 3
        }
        self.layer_configure_frm = ttk.LabelFrame(self, text='Конфигурация слоев')
        self.layer_configure_frm.pack(fill=tk.BOTH, expand=1, **pad)
        self.model_frm = ttk.LabelFrame(self, text='Работа с моделью')
        self.model_frm.pack(fill=tk.BOTH, expand=1, **pad)
        configuration_frm = tk.Frame(self.layer_configure_frm)
        configuration_frm.pack(side=tk.LEFT, anchor=tk.N, **pad)
        layer_frm = ttk.LabelFrame(self.layer_configure_frm, text='Настройка слоев')
        layer_frm.pack(side=tk.RIGHT, anchor=tk.N, **pad)

        tk.Label(configuration_frm, text='Количество слоев:').pack(**pad)
        self.layer_amount_sb = ttk.Spinbox(configuration_frm, from_=1, to=2, width=20)
        self.layer_amount_sb.set(1)
        self.layer_amount_sb.pack(**pad)
        ttk.Label(configuration_frm, text='Целевая переменная:').pack()
        self.y_cb = ttk.Combobox(configuration_frm, values=list(self.pd_data.columns))
        self.y_cb.pack(**pad)
        ttk.Button(configuration_frm, text='Подтвердить', command=lambda: self.get_layers(layer_frm)).pack(**pad)

        ent_width = {'width': 15}
        activation = ['relu', 'sigmoid']
        self.layer_unit_lbl_1 = ttk.Label(layer_frm, text='unit')
        self.layer_unit_ent_1 = ttk.Entry(layer_frm, **ent_width)
        self.layer_unit_lbl_2 = ttk.Label(layer_frm, text='unit')
        self.layer_unit_ent_2 = ttk.Entry(layer_frm, **ent_width)
        self.layer_unit_lbl_end = ttk.Label(layer_frm, text='unit')
        self.layer_unit_ent_end = ttk.Entry(layer_frm, **ent_width)
        self.layer_activation_lbl_1 = ttk.Label(layer_frm, text='activation')
        self.layer_activation_cb_1 = ttk.Combobox(layer_frm, values=activation, **ent_width)
        self.layer_activation_lbl_2 = ttk.Label(layer_frm, text='activation')
        self.layer_activation_cb_2 = ttk.Combobox(layer_frm, values=activation, **ent_width)
        self.layer_activation_lbl_end = ttk.Label(layer_frm, text='activation')
        self.layer_activation_cb_end = ttk.Combobox(layer_frm, values=activation, **ent_width)
        self.layer_1_lbl = ttk.Label(layer_frm, text='Слой 1:')
        self.layer_2_lbl = ttk.Label(layer_frm, text='Слой 2:')
        self.layer_end_lbl = ttk.Label(layer_frm, text='Выходной:')

        self.labels = {
            'unit': [
                self.layer_unit_lbl_1,
                self.layer_unit_ent_1,
                self.layer_unit_lbl_2,
                self.layer_unit_ent_2,
                self.layer_unit_lbl_end,
                self.layer_unit_ent_end,
            ],
            'activation': [
                self.layer_activation_lbl_1,
                self.layer_activation_cb_1,
                self.layer_activation_lbl_2,
                self.layer_activation_cb_2,
                self.layer_activation_lbl_end,
                self.layer_activation_cb_end,
            ],
            'layer': [
                self.layer_1_lbl,
                self.layer_2_lbl,
                self.layer_end_lbl,
            ]
        }
        self.get_layers(layer_frm)

    def get_layers(self, frm):
        amount = int(self.layer_amount_sb.get())
        print(amount)
        row, column = 0, 0
        pad = {
            'padx': 5,
            'pady': 2
        }
        # создаем копию списков, чтобы не удалять слои из основного словаря
        l1 = self.labels['unit'] * 1
        l2 = self.labels['activation'] * 1
        l3 = self.labels['layer'] * 1
        if amount == 1:
            del l1[2:4]
            del l2[2:4]
            del l3[1]
        for frame in frm.winfo_children():
            frame.pack_forget()
        for layer in l3:
            layer.grid(row=row + 1, column=column, **pad)
            row += 2
        row = 0
        column += 1
        for layers in [l1, l2]:
            for layer in layers:
                layer.grid(row=row, column=column, **pad)
                row += 1
            row = 0
            column += 1
        ttk.Button(frm, text='Подтвердить', command=self.fit).grid(row=6, column=0, columnspan=3, pady=5)

    def fit(self):
        y = self.pd_data[self.y_cb.get()]
        x = self.pd_data.drop(self.y_cb.get(), axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model = Sequential()
        model.add(Dense(units=int(self.layer_unit_ent_1.get()),
                        activation=self.layer_activation_cb_1.get(),
                        input_shape=(x_train.shape[1],)))
        if int(self.layer_amount_sb.get()) == 2:
            model.add(Dense(units=int(self.layer_unit_ent_2.get()),
                            activation=self.layer_activation_cb_2.get()))
        model.add(Dense(units=int(self.layer_unit_ent_end.get()),
                        activation=self.layer_activation_cb_end.get()))
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['AUC'])
        model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=30,
                            validation_split=0.2)
        score = model.evaluate(x=x_test, y=y_test)
        print(score)


class ForecastingFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.model = Sequential()
