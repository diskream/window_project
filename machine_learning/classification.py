import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
from tools.functions import deserialize, update_entry, save_model, get_models_list
from tools.models import Model
import pandas as pd
from sqlite3 import connect
# Для графиков
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

matplotlib.use('TkAgg')


class ClassificationView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        # Инициализация окна
        self.w, self.h, p = self.winfo_screenwidth(), self.winfo_screenheight(), 0.4
        self.geometry(f'{int(self.w * p)}x{int(self.h * p)}')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        # Разделение окна на фреймы
        pad = {
            'padx': 5,
            'pady': 5
        }
        self.left_frm = ttk.LabelFrame(self, text='Создание модели')
        self.left_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, **pad)
        self.right_frm = WorkWithModel(self)
        self.right_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, **pad)

        self.upper_frm = ttk.Labelframe(self.left_frm, text='Выбор алгоритма')
        self.upper_frm.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.alg_cb = ttk.Combobox(self.upper_frm, values=['Дерево решений',
                                                           'Случайный лес',
                                                           'k Ближайших соседей'],
                                   )
        self.lower_frm = tk.Frame(self.left_frm)
        self.lower_frm.pack(side=tk.BOTTOM)
        # Выбор алгоритма
        self.alg_cb.current(0)
        self.alg_cb.pack(side=tk.TOP, pady=10)
        ttk.Button(self.upper_frm, text='Выбрать', command=self.get_alg).pack(side=tk.TOP, pady=5)
        self.update_title()
        # Фреймы алгоритмов
        self.dt_frm = DecisionTreeFrame(self.left_frm, self.entry, self.pd_data)
        self.rf_frm = RandomForestFrame(self.left_frm, self.entry, self.pd_data)
        self.knn_frm = KNeighborsFrame(self.left_frm, self.entry, self.pd_data)
        self.algorithms = {
            'Дерево решений': self.dt_frm,
            'Случайный лес': self.rf_frm,
            'k Ближайших соседей': self.knn_frm
        }
        self.current_alg = None
        self.get_alg()

    def update_title(self):
        self.title(f'Работа с {self.entry.name} с помощью алгоритма "{self.alg_cb.get()}"')

    def get_alg(self):
        alg = self.alg_cb.get()
        self.title(f'Работа с {self.entry.name} с помощью алгоритма "{alg}"')
        if self.current_alg is not None:
            self.current_alg.pack_forget()
        self.algorithms[alg].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.current_alg = self.algorithms[alg]


class WorkWithModel(ttk.LabelFrame):
    def __init__(self, master):
        ttk.LabelFrame.__init__(self, master)
        self.master = master
        self.configure(text='Работа с моделью')
        pad = {
            'padx': 5,
            'pady': 5
        }
        model_selection_frm = ttk.Frame(self)
        model_selection_frm.pack()
        # Выбор модели
        ttk.Label(model_selection_frm, text='Выберите модель из списка:').grid(row=0, column=0, **pad)
        self.model_cb = ttk.Combobox(model_selection_frm, values=get_models_list())
        self.model_cb.grid(row=0, column=1, **pad)
        ttk.Button(model_selection_frm, text='Выбрать', command=self.get_model).grid(row=0, column=2, **pad)
        ttk.Button(model_selection_frm, text='Обновить', command=self.update_cb).grid(row=0, column=3, **pad)

        self.model_entry = None  # Класс модели
        self.model = None  # сама обученная модель
        self.isSelected = False  # выбрана ли модель
        # Создания фреймов действий над моделью
        model_overview = ttk.LabelFrame(self, text='Обзор модели')
        model_overview.pack(fill=tk.BOTH, expand=1, **pad)
        model_management = ttk.LabelFrame(self, text='Управление моделью')
        model_management.pack(fill=tk.BOTH, expand=1, **pad)
        model_query = ttk.LabelFrame(self, text='Запрос к модели')
        model_query.pack(fill=tk.X, **pad)
        # Обзор модели
        self.model_overview = ttk.Label(model_overview, text=self.get_model_overview(), justify=tk.LEFT)
        self.model_overview.pack(side=tk.LEFT, anchor=tk.N, **pad)
        # Управление моделью
        ttk.Label(model_management, text='Изменение названия модели:').grid(row=0, column=0, columnspan=2, **pad)
        self.new_name_ent = ttk.Entry(model_management)
        self.new_name_ent.grid(row=1, column=0, **pad)
        ttk.Button(model_management, text='Подтвердить', command=self.update_name).grid(row=1, column=1, **pad)
        ttk.Label(model_management, text='Удаление модели:').grid(row=2, column=0, columnspan=2, **pad)
        ttk.Button(model_management, text='Удалить модель', command=self.delete_model).grid(row=3, column=0,
                                                                                            columnspan=2,**pad)

        ttk.Label(model_management, text='Изменение описания модели:').grid(row=0, column=2, columnspan=2, **pad)
        self.description_text = tk.Text(model_management, width=30, height=5)
        self.description_text.grid(row=1, column=2, columnspan=2, rowspan=6, **pad)
        # Управление моделью
        ttk.Label(model_query, text='Выберите таблицу БД:').grid(row=0, column=0, **pad)
        self.table_selection_cb = ttk.Combobox(model_query, values=['Tasks', 'Task_variant'])
        self.table_selection_cb.grid(row=0, column=1, **pad)
        self.table_selection_cb.current(1)
        ttk.Button(model_query, text='Выбрать', command=self.get_data_list).grid(row=0, column=2, **pad)
        ttk.Label(model_query, text='Выберите данные:', justify=tk.LEFT).grid(row=1, column=0, **pad)
        self.data_selection_cb = ttk.Combobox(model_query)
        self.data_selection_cb.grid(row=1, column=1, **pad)
        self.get_data_list()
        ttk.Button(model_query, text='Выбрать', command=self.get_data).grid(row=1, column=2, **pad)
        self.data = None

    def get_data_list(self):
        table = self.table_selection_cb.get()
        with connect('main.sqlite3') as conn:
            values = conn.cursor().execute(f"SELECT name FROM {table}").fetchall()
            self.data_selection_cb.configure(values=values)

    def get_data(self):
        table, data = self.table_selection_cb.get(), self.data_selection_cb.get()
        with connect('main.sqlite3') as conn:
            query = conn.cursor().execute(f"SELECT table_file FROM {table} WHERE name = ?", (data,)).fetchone()[0]
            self.data = deserialize(query)
            self.predict()

    def predict(self):
        data = self.data[list(self.model.feature_names_in_)]

        if set(data.columns) == set(self.model.feature_names_in_):
            data['Prediction'] = self.model.predict(data)
            # Вывод окна с данными
            window = tk.Toplevel(self.master)
            window.geometry(self.master.geometry())
            w1 = ttk.Frame(window)
            w1.pack(fill=tk.BOTH, expand=1)
            w2 = ttk.Frame(window)
            w2.pack(fill=tk.X)
            tv = ttk.Treeview(w1, show='headings')
            tv['columns'] = list(data.columns)
            for col in tv['columns']:
                tv.heading(col, text=col)
                tv.column(col, width=50)
            rows = data.to_numpy().tolist()
            for row in rows:
                tv.insert('', tk.END, values=row)
            ysb = ttk.Scrollbar(w1, orient=tk.VERTICAL, command=tv.yview)
            xsb = ttk.Scrollbar(w2, orient=tk.HORIZONTAL, command=tv.xview)
            tv.configure(yscroll=ysb.set, xscroll=xsb.set)
            ysb.pack(side=tk.RIGHT, fill=tk.Y)
            xsb.pack(side=tk.BOTTOM, fill=tk.X)
            tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        else:
            messagebox.showwarning("Столбцы данных и переменные модели не совпадают!")

    def get_model_overview(self):
        def get_data_name():
            """
            Возвращает название данных, по которым обучалась модель.
            Подразумевается, что данные обучались мо предобработанным данным
            (т.е. по данным из таблицы Task_variant)
            :return: название данных
            """
            try:
                with connect('main.sqlite3') as conn:
                    query = "SELECT name FROM Task_variant WHERE task_id = ? AND variant_id = ?"
                    print(self.model_entry.task_id, self.model_entry.variant_id)
                    return \
                    conn.cursor().execute(query, (self.model_entry.task_id, self.model_entry.variant_id)).fetchone()[0]
            except TypeError as _ex:
                self.model_overview.configure(text='При загрузке модели возникла ошибка.')
                print(_ex)

        if not self.isSelected:
            return '\n' * 10
        elif not isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier)):
            return "Выбранная модель не является алгоритмом классификации.\nПожалуйста, выберите другую модель."
        else:
            info = ''
            info += f'Данные, по которым была обучена модель: {get_data_name()};\n\n'
            info += f'Точность модели: {self.model_entry.acc};\n\nАлгоритм модели: {type(self.model).__name__};\n\n'
            info += f'Переменные в модели:\n{", ".join(self.model.feature_names_in_)}\n\n'
            info += f'Описание модели:\n {self.model_entry.desc}'
            return info

    def get_model(self):
        model = self.model_cb.get()
        with connect('main.sqlite3') as conn:
            query = conn.cursor().execute('SELECT * FROM Models WHERE name = ?', (model,)).fetchone()
            self.model_entry = Model(*query)
            self.model = deserialize(self.model_entry.bin)
            self.isSelected = True
            self.model_overview.configure(text=self.get_model_overview())

    def update_cb(self):
        self.model_cb.configure(values=get_models_list())

    def update_name(self):
        pass

    def delete_model(self):
        pass


class DecisionTreeFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.clf_conf_lb_frm = ttk.Labelframe(self, text='Конфигурация дерева решений')
        self.clf_conf_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.clf_conf_frm = tk.Frame(self.clf_conf_lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.model_lb_frm = ttk.Labelframe(self, text='Конфигурация параметров обучения')
        self.model_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.model_frm_l = tk.Frame(self.model_lb_frm)
        self.model_frm_l.pack(side=tk.LEFT)
        self.model_frm_r = tk.Frame(self.model_lb_frm)
        self.model_frm_r.pack(side=tk.RIGHT)
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)
        self.clf = DecisionTreeClassifier
        self.default_params = {
            'criterion': tk.StringVar(self.clf_conf_frm, value='entropy'),
            'max_depth': tk.StringVar(self.clf_conf_frm, value=15),
            'min_samples_split': tk.StringVar(self.clf_conf_frm, value=2),
            'min_samples_leaf': tk.StringVar(self.clf_conf_frm, value=1),
            'min_weight_fraction_leaf': tk.StringVar(self.clf_conf_frm, value=0.0),
            'random_state': tk.StringVar(self.clf_conf_frm, value='None')
        }
        self.params = {}
        info = '  Параметры классификатора:\n' + \
               '- criterion: [gini, entropy]; max_depth: максимальная глубина дерева,\n' + \
               '- min_samples_split: минимальное количество объектов для разделения внутри узла;\n' + \
               '- min_samples_leaf: минимальное количество объектов для разделения внутри листа;\n' + \
               '- min_weight_fraction_leaf: минимальный вес узла;\n' + \
               '- random_state: управляет случайностью оценки;' \
               '- n_estimators: количество решающих деревьев.'

        ttk.Button(self.clf_conf_lb_frm, text='Информация о параметрах',
                   command=lambda: messagebox.showinfo('Информация о параметрах', info)).pack(
            side=tk.TOP)
        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        tk.Label(self.clf_conf_frm, text='criterion').grid(row=0, column=0, padx=5)
        self.criterion_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['criterion'], **ent_options)
        self.criterion_ent.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='max_depth').grid(row=0, column=1, padx=5)
        self.max_depth_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['max_depth'], **ent_options)
        self.max_depth_ent.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_split').grid(row=0, column=2, padx=5)
        self.min_samples_split_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_split'],
                                               **ent_options)
        self.min_samples_split_ent.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_leaf').grid(row=2, column=0, padx=5)
        self.min_samples_leaf_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_leaf'],
                                              **ent_options)
        self.min_samples_leaf_ent.grid(row=3, column=0, padx=5, pady=10)

        tk.Label(self.clf_conf_frm, text='min_weight_fraction_leaf').grid(row=2, column=1, padx=5)
        self.min_weight_fraction_leaf_ent = ttk.Entry(self.clf_conf_frm,
                                                      textvariable=self.default_params['min_weight_fraction_leaf'],
                                                      **ent_options)
        self.min_weight_fraction_leaf_ent.grid(row=3, column=1, padx=5, pady=10)

        tk.Label(self.clf_conf_frm, text='random_state').grid(row=2, column=2, padx=5)
        self.random_state_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['random_state'],
                                          **ent_options)
        self.random_state_ent.grid(row=3, column=2, padx=5, pady=10)
        model_pack = {
            'side': tk.TOP,
            'padx': 5,
            'pady': 3,
        }
        tk.Label(self.model_frm_l, text='Выберите целевую переменную:').pack(**model_pack)
        self.col_cb = ttk.Combobox(self.model_frm_l, values=list(self.pd_data.columns))
        self.col_cb.pack(**model_pack)
        tk.Label(self.model_frm_l, text='Выберите процент тестовой выборки').pack(**model_pack)
        self.split_sb = ttk.Spinbox(self.model_frm_l, from_=25, to=40, width=20)
        self.split_sb.pack(**model_pack)
        self.check_var = tk.BooleanVar(self)
        self.check_split = ttk.Checkbutton(self.model_frm_l, text='Разделить выборку\nна тренировочную и тестовую',
                                           variable=self.check_var, onvalue=True, offvalue=False)
        self.check_split.pack(**model_pack)
        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }
        ttk.Button(self.btn_frm, text='Подтвердить', command=self.fit).pack(**btn_pack)
        ttk.Button(self.btn_frm, text='ROC-кривая', command=self.get_roc).pack(**btn_pack)
        ttk.Button(self.btn_frm, text='Открыть дерево', command=self.get_tree).pack(**btn_pack)
        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).pack(**btn_pack)

        self.cv_var = tk.BooleanVar(self)
        self.check_cv = ttk.Checkbutton(self.model_frm_r, text='Кросс-валидация', variable=self.cv_var, onvalue=True,
                                        offvalue=False)

        self.accuracy = '...'
        self.cv_accuracy = '...'
        self.check_cv.pack(**model_pack)
        tk.Label(self.model_frm_r, text='Количество разделений кросс-валидации').pack(**model_pack)
        self.cv_sb = ttk.Spinbox(self.model_frm_r, from_=3, to=7, width=20)
        self.cv_sb.pack(**model_pack)
        self.acc_lbl = tk.Label(self.model_frm_r, text=f'Точность модели:\n{str(self.accuracy)}\n'
                                                       f'Средняя точность при кросс-валидации:'
                                                       f'\n{str(self.cv_accuracy)}')
        self.acc_lbl.pack(**model_pack)

        self.isSplitted = False
        self.feature_names = None
        self.class_names = None

    def fit(self, *args, **kwargs):
        self.clf = self.get_clf(*args, **kwargs)
        if self.check_var.get() == 1:
            x_train, y_train, x_test, y_test = self.get_split_data()
            self.get_cv(x_train, y_train)
            self.clf.fit(x_train, y_train)
            self.accuracy = self.clf.score(x_test, y_test)
            self.acc_lbl.configure(text=f'Точность модели:\n{str(self.accuracy)}\n'
                                        f'Точность при кросс-валидации:\n{str(self.cv_accuracy)}')
            self.isSplitted = True
        else:
            x, y = self.get_split_data()
            self.get_cv(x, y)
            self.clf.fit(x, y)
            self.accuracy = self.clf.score(x, y)
            self.acc_lbl.configure(text=f'Точность модели:\n{self.accuracy}\n'
                                        f'Точность при кросс-валидации:\n{self.cv_accuracy}')
            self.isSplitted = False

    def get_cv(self, x, y):
        if self.cv_var.get():
            cv = cross_val_score(self.clf, x, y, cv=int(self.cv_sb.get()))
            self.cv_accuracy = sum(cv) / len(cv)
        else:
            self.cv_accuracy = '...'

    def get_split_data(self):
        target = self.col_cb.get()
        x = self.pd_data.drop(target, axis=1)
        y = self.pd_data[target]
        self.feature_names, self.class_names = list(x.columns), y.unique()
        if self.check_var.get() == 1:
            percent = float(self.split_sb.get()) / 100
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent)
            return x_train, y_train, x_test, y_test  # именно в формате x, y, x_test, y_test - для работы с графиками
        else:
            return x, y

    def get_clf(self, *args, **kwargs):
        params = self.get_params(*args, **kwargs)
        if isinstance(self.clf, DecisionTreeClassifier):
            self.clf = DecisionTreeClassifier  # Обновляем классификатор, если пользователь изменил параметры
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf = RandomForestClassifier
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = KNeighborsClassifier
        return self.clf(**params)

    def get_params(self, *args, **kwargs):
        if kwargs:
            ent_obj = kwargs['params']
        else:
            ent_obj = [self.criterion_ent, self.max_depth_ent, self.min_samples_split_ent, self.min_samples_leaf_ent,
                       self.min_weight_fraction_leaf_ent, self.random_state_ent]
            if args:
                for ent in args:
                    ent_obj.append(ent)
        params = {}
        for param, obj in zip(self.default_params.keys(), ent_obj):
            if obj.get() == 'None':
                params[param] = None
            elif param == 'min_weight_fraction_leaf':
                params[param] = float(obj.get())
            else:
                try:
                    params[param] = int(obj.get())
                except ValueError:
                    params[param] = obj.get()
        return params

    def get_roc(self) -> None:
        DecisionTreeModelInfo(self.clf, self.isSplitted, *self.get_split_data())

    def get_tree(self):
        plt.figure(1, figsize=(16, 9), dpi=500)
        plot_tree(self.clf,
                  feature_names=self.feature_names,
                  class_names=self.class_names.astype(str),
                  filled=True)
        plt.savefig('other/temp/tree.png')
        plt.clf()
        fig = plt.figure(1, figsize=(16, 9), dpi=500)
        fig.tight_layout()
        img = mpimg.imread('other/temp/tree.png')
        plt.imshow(img)
        plt.show()

    def save(self):
        save_model(self.entry, self.clf, self.accuracy)


class RandomForestFrame(DecisionTreeFrame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.clf_conf_lb_frm = ttk.Labelframe(self, text='Конфигурация случайного леса')
        self.clf_conf_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.clf_conf_frm = tk.Frame(self.clf_conf_lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.model_lb_frm = ttk.Labelframe(self, text='Конфигурация параметров обучения')
        self.model_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.model_frm_l = tk.Frame(self.model_lb_frm)
        self.model_frm_l.pack(side=tk.LEFT)
        self.model_frm_r = tk.Frame(self.model_lb_frm)
        self.model_frm_r.pack(side=tk.RIGHT)
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)
        self.clf = RandomForestClassifier
        self.default_params = {
            'criterion': tk.StringVar(self.clf_conf_frm, value='entropy'),
            'max_depth': tk.StringVar(self.clf_conf_frm, value=15),
            'min_samples_split': tk.StringVar(self.clf_conf_frm, value=2),
            'min_samples_leaf': tk.StringVar(self.clf_conf_frm, value=1),
            'min_weight_fraction_leaf': tk.StringVar(self.clf_conf_frm, value=0.0),
            'random_state': tk.StringVar(self.clf_conf_frm, value='None'),
            'n_estimators': tk.StringVar(self.clf_conf_frm, value=100)
        }
        self.params = {}
        info = '  Параметры классификатора:\n' + \
               '- criterion: [gini, entropy]; max_depth: максимальная глубина дерева,\n' + \
               '- min_samples_split: минимальное количество объектов для разделения внутри узла;\n' + \
               '- min_samples_leaf: минимальное количество объектов для разделения внутри листа;\n' + \
               '- min_weight_fraction_leaf: минимальный вес узла;\n' + \
               '- random_state: управляет случайностью оценки.'

        ttk.Button(self.clf_conf_lb_frm, text='Информация о параметрах',
                   command=lambda: messagebox.showinfo('Информация о параметрах', info)).pack(
            side=tk.TOP)
        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        tk.Label(self.clf_conf_frm, text='criterion').grid(row=0, column=0, padx=5)
        self.criterion_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['criterion'], **ent_options)
        self.criterion_ent.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='max_depth').grid(row=0, column=1, padx=5)
        self.max_depth_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['max_depth'], **ent_options)
        self.max_depth_ent.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_split').grid(row=0, column=2, padx=5)
        self.min_samples_split_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_split'],
                                               **ent_options)
        self.min_samples_split_ent.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_leaf').grid(row=2, column=0, padx=5)
        self.min_samples_leaf_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_leaf'],
                                              **ent_options)
        self.min_samples_leaf_ent.grid(row=3, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_weight_fraction_leaf').grid(row=2, column=1, padx=5)
        self.min_weight_fraction_leaf_ent = ttk.Entry(self.clf_conf_frm,
                                                      textvariable=self.default_params['min_weight_fraction_leaf'],
                                                      **ent_options)
        self.min_weight_fraction_leaf_ent.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='random_state').grid(row=2, column=2, padx=5)
        self.random_state_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['random_state'],
                                          **ent_options)
        self.random_state_ent.grid(row=3, column=2, padx=5, pady=5)
        tk.Label(self.clf_conf_frm, text='n_estimators').grid(row=4, column=0, padx=5)
        self.n_estimators_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['n_estimators'],
                                          **ent_options)
        self.n_estimators_ent.grid(row=5, column=0, padx=5, pady=5)
        # Доп параметры, которых нет в родительском классе DecisionTreeFrame
        extra_params = [self.n_estimators_ent]
        model_pack = {
            'side': tk.TOP,
            'padx': 5,
            'pady': 3,
        }
        tk.Label(self.model_frm_l, text='Выберите целевую переменную:').pack(**model_pack)
        self.col_cb = ttk.Combobox(self.model_frm_l, values=list(self.pd_data.columns))
        self.col_cb.pack(**model_pack)
        tk.Label(self.model_frm_l, text='Выберите процент тестовой выборки').pack(**model_pack)
        self.split_sb = ttk.Spinbox(self.model_frm_l, from_=25, to=40, width=20)
        self.split_sb.pack(**model_pack)
        self.check_var = tk.BooleanVar(self)
        self.check_split = ttk.Checkbutton(self.model_frm_l, text='Разделить выборку\nна тренировочную и тестовую',
                                           variable=self.check_var, onvalue=True, offvalue=False)
        self.check_split.pack(**model_pack)
        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }
        ttk.Button(self.btn_frm, text='Подтвердить', command=lambda: self.fit(*extra_params)).pack(**btn_pack)
        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).pack(**btn_pack)
        self.cv_var = tk.BooleanVar(self)
        self.check_cv = ttk.Checkbutton(self.model_frm_r, text='Кросс-валидация', variable=self.cv_var, onvalue=True,
                                        offvalue=False)

        self.accuracy = '...'
        self.cv_accuracy = '...'
        self.check_cv.pack(**model_pack)
        tk.Label(self.model_frm_r, text='Количество разделений кросс-валидации').pack(**model_pack)
        self.cv_sb = ttk.Spinbox(self.model_frm_r, from_=3, to=7, width=20)
        self.cv_sb.pack(**model_pack)
        self.acc_lbl = tk.Label(self.model_frm_r, text=f'Точность модели:\n{str(self.accuracy)}\n'
                                                       f'Средняя точность при кросс-валидации:'
                                                       f'\n{str(self.cv_accuracy)}')
        self.acc_lbl.pack(**model_pack)

        self.isSplitted = False
        self.feature_names = None
        self.class_names = None


class KNeighborsFrame(DecisionTreeFrame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.clf_conf_lb_frm = ttk.Labelframe(self, text='Конфигурация k ближайших соседей')
        self.clf_conf_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.clf_conf_frm = tk.Frame(self.clf_conf_lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.model_lb_frm = ttk.Labelframe(self, text='Конфигурация параметров обучения')
        self.model_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.model_frm_l = tk.Frame(self.model_lb_frm)
        self.model_frm_l.pack(side=tk.LEFT)
        self.model_frm_r = tk.Frame(self.model_lb_frm)
        self.model_frm_r.pack(side=tk.RIGHT)
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)
        self.clf = KNeighborsClassifier
        self.default_params = {
            'n_neighbors': tk.StringVar(self.clf_conf_frm, value=5),
            'weights': tk.StringVar(self.clf_conf_frm, value='uniform'),
            'algorithm': tk.StringVar(self.clf_conf_frm, value='auto'),
            'leaf_size': tk.StringVar(self.clf_conf_frm, value=30),
            'p': tk.StringVar(self.clf_conf_frm, value=2)
        }
        self.params = {}
        info = '  Параметры классификатора:\n' + \
               '- criterion: [gini, entropy]; max_depth: максимальная глубина дерева,\n' + \
               '- min_samples_split: минимальное количество объектов для разделения внутри узла;\n' + \
               '- min_samples_leaf: минимальное количество объектов для разделения внутри листа;\n' + \
               '- min_weight_fraction_leaf: минимальный вес узла;\n' + \
               '- random_state: управляет случайностью оценки.'

        ttk.Button(self.clf_conf_lb_frm, text='Информация о параметрах',
                   command=lambda: messagebox.showinfo('Информация о параметрах', info)).pack(
            side=tk.TOP)
        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        tk.Label(self.clf_conf_frm, text='n_neighbors').grid(row=0, column=0, padx=5)
        self.n_neighbors_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['n_neighbors'],
                                         **ent_options)
        self.n_neighbors_ent.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='weights').grid(row=0, column=1, padx=5)
        self.weights_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['weights'], **ent_options)
        self.weights_ent.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='algorithm').grid(row=0, column=2, padx=5)
        self.algorithm_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['algorithm'], **ent_options)
        self.algorithm_ent.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='leaf_size').grid(row=2, column=0, padx=5)
        self.leaf_size_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['leaf_size'], **ent_options)
        self.leaf_size_ent.grid(row=3, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='p').grid(row=2, column=1, padx=5)
        self.p_ent = ttk.Entry(self.clf_conf_frm, textvariable=self.default_params['p'], **ent_options)
        self.p_ent.grid(row=3, column=1, padx=5, pady=5)
        # Доп параметры, которых нет в родительском классе DecisionTreeFrame
        extra_params = {'params': [self.n_neighbors_ent, self.weights_ent, self.algorithm_ent, self.leaf_size_ent,
                                   self.p_ent]}
        model_pack = {
            'side': tk.TOP,
            'padx': 5,
            'pady': 3,
        }
        tk.Label(self.model_frm_l, text='Выберите целевую переменную:').pack(**model_pack)
        self.col_cb = ttk.Combobox(self.model_frm_l, values=list(self.pd_data.columns))
        self.col_cb.pack(**model_pack)
        tk.Label(self.model_frm_l, text='Выберите процент тестовой выборки').pack(**model_pack)
        self.split_sb = ttk.Spinbox(self.model_frm_l, from_=25, to=40, width=20)
        self.split_sb.pack(**model_pack)
        self.check_var = tk.BooleanVar(self)
        self.check_split = ttk.Checkbutton(self.model_frm_l, text='Разделить выборку\nна тренировочную и тестовую',
                                           variable=self.check_var, onvalue=True, offvalue=False)
        self.check_split.pack(**model_pack)
        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }
        ttk.Button(self.btn_frm, text='Подтвердить', command=lambda: self.fit(**extra_params)).pack(**btn_pack)
        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).pack(**btn_pack)
        self.cv_var = tk.BooleanVar(self)
        self.check_cv = ttk.Checkbutton(self.model_frm_r, text='Кросс-валидация', variable=self.cv_var, onvalue=True,
                                        offvalue=False)

        self.accuracy = '...'
        self.cv_accuracy = '...'
        self.check_cv.pack(**model_pack)
        tk.Label(self.model_frm_r, text='Количество разделений кросс-валидации').pack(**model_pack)
        self.cv_sb = ttk.Spinbox(self.model_frm_r, from_=3, to=7, width=20)
        self.cv_sb.pack(**model_pack)
        self.acc_lbl = tk.Label(self.model_frm_r, text=f'Точность модели:\n{str(self.accuracy)}\n'
                                                       f'Средняя точность при кросс-валидации:'
                                                       f'\n{str(self.cv_accuracy)}')
        self.acc_lbl.pack(**model_pack)

        self.isSplitted = False
        self.feature_names = None
        self.class_names = None


class DecisionTreeModelInfo(tk.Tk):
    def __init__(self, clf, is_splitted: bool, x, y, x_test=None, y_test=None):
        tk.Tk.__init__(self)
        self.clf = clf
        self.x = x
        self.y = y
        self.plot_size = (8, 6)
        self.is_splitted = is_splitted
        bg_opt = {
            'bg': 'white'
        }
        self.top_frm = tk.Frame(self, **bg_opt)
        self.top_frm.pack(side=tk.TOP)
        self.bottom_frm = tk.Frame(self, **bg_opt)
        self.bottom_frm.pack(side=tk.BOTTOM, fill=tk.X)
        if self.is_splitted:
            self.x_test = x_test
            self.y_test = y_test
        self.plot_feature_importance(self.plot_size)
        self.plot_roc(self.plot_size)
        prediction = self.clf.predict(self.x_test) if self.is_splitted else self.clf.predict(self.x)
        # report = classification_report(self.y_test if self.is_splitted else self.y, prediction)
        # tk.Label(self.bottom_frm, text=report, justify=tk.LEFT, **bg_opt).pack()

    def plot_feature_importance(self, plot_size: tuple) -> None:
        feature_importance = self.clf.feature_importances_
        feature_names = self.x.columns
        # Create a DataFrame using a Dictionary
        data = {'feature_names': feature_names, 'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        # Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
        figure = Figure(figsize=plot_size, dpi=100)
        ax = figure.add_subplot(1, 1, 1)
        sns.barplot(y=fi_df['feature_names'], x=fi_df['feature_importance'], ax=ax)
        f_i_graph = FigureCanvasTkAgg(figure, self.top_frm)
        f_i_graph.get_tk_widget().pack(side=tk.LEFT)

    def plot_roc(self, plot_size):
        y_predicted = self.clf.predict_proba(self.x_test if self.is_splitted else self.x)
        fpr, tpr, thresholds = roc_curve(self.y_test if self.is_splitted else self.y, y_predicted[:, 1])
        roc_auc = auc(fpr, tpr)
        figure = Figure(figsize=plot_size, dpi=100)
        plot = figure.add_subplot(1, 1, 1)
        plot.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_graph = FigureCanvasTkAgg(figure, self.top_frm)
        roc_graph.get_tk_widget().pack(side=tk.LEFT)


class DecisionTreePlot(tk.Tk):
    def __init__(self, clf, fn, cn):
        tk.Tk.__init__(self)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=100)
        plot_tree(clf,
                  feature_names=fn,
                  class_names=cn.astype(str),
                  filled=True)
        fig.savefig('temp/tree_test.png')
        tk.Label(self, text='Window is currently unavailable')
