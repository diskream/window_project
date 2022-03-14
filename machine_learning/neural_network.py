import tkinter as tk
from tkinter import ttk
from io import StringIO
from sqlite3 import connect
from tools.functions import deserialize, update_entry, save_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

summary = '''Model:
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________'''


class NNView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        # self.tk.call('source', r'tools/Sun-Valley-ttk-theme-master/sun-valley.tcl')
        # self.tk.call('set_theme', 'light')
        self.geometry('820x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        pad = {
            'padx': 5,
            'pady': 5
        }
        self.left_frm = ttk.LabelFrame(self, text='Работа с моделью')
        self.left_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, **pad)
        self.right_frm = ttk.LabelFrame(self, text='Информация по слоям')
        self.right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1,  **pad)

        selection_frm = ttk.Frame(self.left_frm)
        selection_frm.pack(fill=tk.X, padx=5)
        central_selection_frm = ttk.Frame(selection_frm)
        central_selection_frm.pack()

        model_download_frm = ttk.LabelFrame(self.right_frm, text='Загрузка модели')
        model_download_frm.pack(side=tk.BOTTOM, fill=tk.X, **pad)

        ttk.Label(central_selection_frm, text='Пожалуйста, выберите решаемую задачу:').grid(row=0, column=0, columnspan=2)
        tasks = ['Классификация', 'Прогнозирование']
        self.task_cb = ttk.Combobox(central_selection_frm, values=tasks)
        self.task_cb.current(0)
        self.task_cb.grid(row=1, column=0, **pad)
        ttk.Button(central_selection_frm, text='Выбрать', command=self.get_task).grid(row=1, column=1, pady=5)

        self.model_summary = ttk.Label(self.right_frm, text=summary)
        self.model_summary.pack(fill=tk.X, side=tk.TOP)
        classification_frm = ClassificationFrame(self.left_frm, self, self.entry, self.pd_data)
        forecasting_frm = ForecastingFrame(self.left_frm, self.entry, self.pd_data)
        self.tasks = dict(zip(tasks, [classification_frm, forecasting_frm]))
        self.current = None
        self.get_task()
        self.model = None
        self.score = None

        ttk.Label(model_download_frm, text='Выберите модель для загрузки:').grid(row=0, column=0, columnspan=2, **pad)
        self.model_download_cb = ttk.Combobox(model_download_frm, values=get_models_list())
        self.model_download_cb.grid(row=1, column=0, **pad)
        ttk.Button(model_download_frm, text='Загрузить', command=self.get_model).grid(row=1, column=1, **pad)

        ttk.Label(self.left_frm, text='Название модели:').pack(side=tk.LEFT, **pad)
        self.model_name_ent = ttk.Entry(self.left_frm)
        self.model_name_ent.pack(side=tk.LEFT, **pad)
        ttk.Button(self.left_frm, text='Сохранить', command=self.save_model).pack(side=tk.LEFT, **pad)
        self.model_save_lbl = ttk.Label(self.left_frm, text='')
        self.model_save_lbl.pack(side=tk.LEFT, **pad)

    def get_task(self):
        task = self.task_cb.get()
        self.title(f'Построение нейронных сетей. {task}.')
        if self.current is not None:
            self.current.pack_forget()
        self.tasks[task].pack(fill=tk.BOTH, expand=1)
        self.current = self.tasks[task]

    def models(self):
        print(self.model.summary())

    def get_summary(self, model, score):
        buf = StringIO()
        model.summary(print_fn=lambda x: buf.write(x + '\n'))
        y = self.pd_data['Survived']
        x = self.pd_data.drop('Survived', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['AUC'])
        score.append(model.evaluate(x=x_test, y=y_test))
        score = "\n\t".join(str(x) for x in score)
        self.model_summary.configure(text=buf.getvalue() + f'\n Score: \n\t{score}', justify=tk.LEFT)

    def get_model(self):
        model = self.model_download_cb.get()
        conn = connect('main.sqlite3')
        cur = conn.cursor()
        try:
            _temp = deserialize(cur.execute(f'SELECT bin_file FROM Models WHERE name = "{model}"').fetchone()[0])
            self.model = deserialize_keras_object(_temp, custom_objects={'Sequential':Sequential})
            self.score = cur.execute(f'SELECT accuracy FROM Models WHERE name = "{model}"').fetchone()[0]
            self.get_summary(self.model, [self.score])
        finally:
            conn.close()

    def save_model(self):
        name = self.model_name_ent.get()
        if name == '':
            name = None
        try:
            save_model(self.entry, self.model, accuracy=self.score[-1], name=name)
            self.model_save_lbl.configure(text='Сохранено успешно!')
        except Exception as _ex:
            print(_ex)
            self.model_save_lbl.configure(text='Произошла ошибка!')


class ClassificationFrame(tk.Frame):
    def __init__(self, parent, master, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.master = master
        self.parent = parent
        self.entry = entry
        self.pd_data = pd_data
        pad = {
            'padx': 5,
            'pady': 3
        }
        self.layer_configure_frm = ttk.LabelFrame(self, text='Конфигурация слоев')
        self.layer_configure_frm.pack(fill=tk.X, **pad)
        self.model_frm = ttk.LabelFrame(self, text='Конфигурация модели')
        self.model_frm.pack(fill=tk.BOTH, expand=1, padx=5)
        configuration_frm = tk.Frame(self.layer_configure_frm)
        configuration_frm.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=15)
        layer_frm = ttk.LabelFrame(self.layer_configure_frm, text='Настройка слоев')
        layer_frm.pack(side=tk.RIGHT, anchor=tk.N, padx=5, pady=5)

        model_comp_frm = ttk.LabelFrame(self.model_frm, text='Model Compile')
        model_comp_frm.grid(row=0, column=0, padx=5, pady=3)
        model_fit_frm = ttk.LabelFrame(self.model_frm, text='Model Fit')
        model_fit_frm.grid(row=0, column=1, padx=5, pady=3)

        tk.Label(configuration_frm, text='Количество слоев:').pack(**pad)
        self.layer_amount_sb = ttk.Spinbox(configuration_frm, from_=1, to=2, width=20)
        self.layer_amount_sb.set(1)
        self.layer_amount_sb.pack(**pad)
        ttk.Label(configuration_frm, text='Целевая переменная:').pack(**pad)
        self.y_cb = ttk.Combobox(configuration_frm, values=list(self.pd_data.columns), width=18)
        self.y_cb.pack(**pad)
        ttk.Button(configuration_frm, text='Подтвердить', command=lambda: self.get_layers(layer_frm)).pack(**pad)

        ent_width = {'width': 15}
        activation = ['relu', 'sigmoid', 'softmax', 'softplus', 'tanh', 'exponential']  # функции активации слоя
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

        # Compile
        w = {
            'width': 12
        }
        self.comp_default = {
            'metrics': tk.StringVar(model_comp_frm, value='accuracy'),
            'jit_compile': tk.BooleanVar(model_comp_frm, value=False),
        }
        opts = ['adam', 'SGD', 'RMSprop']
        ttk.Label(model_comp_frm, text='optimizer').grid(row=0, column=0, **pad)
        self.optimizer_cb = ttk.Combobox(model_comp_frm, values=opts, **w)
        self.optimizer_cb.current(0)
        self.optimizer_cb.grid(row=1, column=0, **pad)
        self.comp_default['optimizer'] = self.optimizer_cb

        loss = ['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error']
        ttk.Label(model_comp_frm, text='loss').grid(row=0, column=1, **pad)
        self.loss_cb = ttk.Combobox(model_comp_frm, values=loss, **w)
        self.loss_cb.current(0)
        self.loss_cb.grid(row=1, column=1, **pad)
        self.comp_default['loss'] = self.loss_cb

        ttk.Label(model_comp_frm, text='metrics').grid(row=2, column=0, **pad)
        self.metrics_ent = ttk.Entry(model_comp_frm, textvariable=self.comp_default['metrics'], width=15)
        self.metrics_ent.grid(row=3, column=0, **pad)

        self.jit_compile_check = ttk.Checkbutton(model_comp_frm, variable=self.comp_default['jit_compile'],
                                                 onvalue=True, offvalue=False, text='jit_compile')
        self.jit_compile_check.grid(row=3, column=1, **pad)

        # Fit
        self.fit_default = {
            'batch_size': None,
            'epochs': None,
            'validation_split': None,
            'use_multiprocessing': tk.BooleanVar(model_fit_frm, value=False)
        }
        ttk.Label(model_fit_frm, text='epochs').grid(row=0, column=0, **pad)
        self.epochs_sb = ttk.Spinbox(model_fit_frm, from_=1, to=100, **w)
        self.epochs_sb.set(1)
        self.fit_default['epochs'] = self.epochs_sb
        self.epochs_sb.grid(row=1, column=0, **pad)

        ttk.Label(model_fit_frm, text='batch_size').grid(row=0, column=1, **pad)
        self.batch_size_sb = ttk.Spinbox(model_fit_frm, from_=1, to=100, **w)
        self.batch_size_sb.set(32)
        self.fit_default['batch_size'] = self.batch_size_sb
        self.batch_size_sb.grid(row=1, column=1, **pad)

        ttk.Label(model_fit_frm, text='validation_split').grid(row=2, column=0, **pad)
        self.validation_split_sb = ttk.Spinbox(model_fit_frm, from_=0.0, to=1.0, increment=0.1, **w)
        self.validation_split_sb.set(0.0)
        self.fit_default['validation_split'] = self.validation_split_sb
        self.validation_split_sb.grid(row=3, column=0, **pad)

        self.use_multiprocessing_check = ttk.Checkbutton(model_fit_frm,
                                                         variable=self.fit_default['use_multiprocessing'],
                                                         onvalue=True, offvalue=False, text='use_multiprocessing')
        self.use_multiprocessing_check.grid(row=3, column=1, **pad)

        ttk.Button(self.model_frm, text='Построить нейронную сеть', command=self.fit).grid(row=1, column=0,
                                                                                           columnspan=2, pady=5)

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
        model.compile(**self.get_compile())
        model.fit(x_train, y_train,verbose=0, **self.get_fit())
        score = model.evaluate(x=x_test, y=y_test)
        print(type(model))
        self.master.model = model
        self.master.score = score
        self.master.get_summary(self.master.model, score)

    def get_compile(self):
        params = {}
        for param, obj in self.comp_default.items():
            print(obj.get(), type(obj.get()))
            if param == 'metrics':
                params[param] = obj.get().split()
            else:
                params[param] = obj.get()
        return params

    def get_fit(self):
        params = {}
        for param, obj in self.fit_default.items():
            if param == 'validation_split':
                params[param] = float(obj.get())
            else:
                params[param] = int(obj.get())
        return params





class ForecastingFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.model = Sequential()


def get_models_list() -> list:
    conn = connect('main.sqlite3')
    cur = conn.cursor()
    try:
        models = []
        for model in cur.execute('SELECT name FROM Models').fetchall():
            models.append(model[0])
        return models
    finally:
        conn.close()
