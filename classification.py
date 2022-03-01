import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
from functions import serialize, deserialize, update_entry
import numpy as np
import pandas as pd
# Для графиков
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class MLView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.geometry('500x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        self.upper_frm = tk.LabelFrame(self, text='Выбор алгоритма')
        self.upper_frm.pack(side=tk.TOP, fill=tk.X)
        self.alg_cb = ttk.Combobox(self, values=['Дерево решений',
                                                 'Случайный лес',
                                                 'k Ближайших соседей'],
                                   )
        self.lower_frm = tk.Frame(self)
        self.lower_frm.pack(side=tk.BOTTOM)
        self.alg_cb.current(0)
        self.alg_cb.pack(side=tk.TOP, pady=10)
        tk.Button(self, text='Выбрать', command=self.get_alg).pack(side=tk.TOP, pady=5)
        self.update_title()
        self.dt_frm = DecisionTreeFrame(self, self.entry, self.pd_data)
        self.rf_frm = RandomForestFrame(self)
        self.knn_frm = KNeighborsFrame(self)
        self.algs = {
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
        self.algs[alg].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.current_alg = self.algs[alg]


class DecisionTreeFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.clf_conf_lb_frm = tk.LabelFrame(self, text='Конфигурация дерева решений')
        self.clf_conf_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.clf_conf_frm = tk.Frame(self.clf_conf_lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.model_lb_frm = tk.LabelFrame(self, text='Конфигурация параметров обучения')
        self.model_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
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
               '- random_state: управляет случайностью оценки.'

        tk.Button(self.clf_conf_lb_frm, text='Информация о параметрах',
                  command=lambda: messagebox.showinfo('Информация о параметрах', info)).pack(
            side=tk.TOP)
        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        tk.Label(self.clf_conf_frm, text='criterion').grid(row=0, column=0, padx=5)
        self.criterion_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['criterion'], **ent_options)
        self.criterion_ent.grid(row=1, column=0, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='max_depth').grid(row=0, column=1, padx=5)
        self.max_depth_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['max_depth'], **ent_options)
        self.max_depth_ent.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_split').grid(row=0, column=2, padx=5)
        self.min_samples_split_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_split'],
                                              **ent_options)
        self.min_samples_split_ent.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(self.clf_conf_frm, text='min_samples_leaf').grid(row=2, column=0, padx=5)
        self.min_samples_leaf_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples_leaf'],
                                             **ent_options)
        self.min_samples_leaf_ent.grid(row=3, column=0, padx=5, pady=10)

        tk.Label(self.clf_conf_frm, text='min_weight_fraction_leaf').grid(row=2, column=1, padx=5)
        self.min_weight_fraction_leaf_ent = tk.Entry(self.clf_conf_frm,
                                                     textvariable=self.default_params['min_weight_fraction_leaf'],
                                                     **ent_options)
        self.min_weight_fraction_leaf_ent.grid(row=3, column=1, padx=5, pady=10)

        tk.Label(self.clf_conf_frm, text='random_state').grid(row=2, column=2, padx=5)
        self.random_state_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['random_state'],
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
        self.check_split = tk.Checkbutton(self.model_frm_l, text='Разделить выборку\nна тренировочную и тестовую',
                                          variable=self.check_var, onvalue=True, offvalue=False)
        print(self.check_var.get())
        self.check_split.pack(**model_pack)
        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }
        tk.Button(self.btn_frm, text='Подтвердить', command=self.fit).pack(**btn_pack)
        tk.Button(self.btn_frm, text='ROC-кривая', command=self.get_roc).pack(**btn_pack)
        tk.Button(self.btn_frm, text='Открыть дерево', command=self.get_tree).pack(**btn_pack)
        tk.Button(self.btn_frm, text='Сохранить модель', command=self.save).pack(**btn_pack)
        self.cv_var = tk.BooleanVar(self)
        self.check_cv = tk.Checkbutton(self.model_frm_r, text='Кросс-валидация', variable=self.cv_var, onvalue=True,
                                       offvalue=False)

        self.accuracy = '...'
        self.cv_accuracy = '...'
        self.check_cv.pack(**model_pack)
        tk.Label(self.model_frm_r, text='Количество разделений кросс-валидации').pack(**model_pack)
        self.cv_sb = ttk.Spinbox(self.model_frm_r, from_=3, to=7, width=20)
        self.cv_sb.pack(**model_pack)
        self.acc_lbl = tk.Label(self.model_frm_r, text=f'Точность модели:\n{str(self.accuracy)}\n'
                                                       f'Средняя точность при кросс-валидации:\n{str(self.cv_accuracy)}')
        self.acc_lbl.pack(**model_pack)

        self.isSplitted = False

    def fit(self):
        self.clf = self.get_clf()
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
        if self.check_var.get() == 1:
            percent = float(self.split_sb.get()) / 100
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent)
            return x_train, y_train, x_test, y_test  # именно в формате x, y, x_test, y_test - для работы с графиками
        else:
            return x, y

    def get_clf(self):
        params = self.get_params()
        if isinstance(self.clf, DecisionTreeClassifier):
            self.clf = DecisionTreeClassifier  # Обновляем классификатор, если пользователь изменил параметры
        return self.clf(**params)

    def get_params(self):
        ent_obj = [self.criterion_ent, self.max_depth_ent, self.min_samples_split_ent, self.min_samples_leaf_ent,
                   self.min_weight_fraction_leaf_ent, self.random_state_ent]
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
        pass

    def save(self):
        pass


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
        self.bottom_frm = tk.Frame(self,**bg_opt)
        self.bottom_frm.pack(side=tk.BOTTOM, fill=tk.X)
        if self.is_splitted:
            self.x_test = x_test
            self.y_test = y_test
        self.plot_feature_importance(self.plot_size)
        self.plot_roc(self.plot_size)
        prediction = self.clf.predict(self.x_test) if self.is_splitted else self.clf.predict(self.x)
        report = classification_report(self.y_test if self.is_splitted else self.y, prediction)
        tk.Label(self.bottom_frm, text=report, justify=tk.LEFT, **bg_opt).pack()

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
        plt = figure.add_subplot(1, 1, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_graph = FigureCanvasTkAgg(figure, self.top_frm)
        roc_graph.get_tk_widget().pack(side=tk.LEFT)


class RandomForestFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text='Random Forest').pack()


class KNeighborsFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text='k Nearest Neighbors').pack()


class MLViewOld(tk.Tk):
    def __init__(self, table, data):
        tk.Tk.__init__(self)
        self.geometry('500x530')
        self.resizable(width=False, height=False)
        self.table = table[0]
        self.data = data
        self.dataframe = self.get_data()
        self.clf = 'classifier pattern'
        self.accuracy = 0
        tk.Label(self, text='Data: ' + self.data + '\nfrom ' + self.table + ' table.').grid(row=0, column=1)
        tk.Label(self, text='Please, choose ML algorithm:').grid(row=1, column=0)

        # configuring combobox
        self.alg_box = ttk.Combobox(self, values=[
            'Decision Tree',
            'Random Forest',
            'k Nearest Neighbors'
        ], width=15)
        self.alg_box.grid(row=2, column=0)
        self.alg_box.current(0)
        self.update_title()
        self.alg_box_button = tk.Button(self, text='Enter', command=self.alg_configuration
                                        ).grid(row=2, column=1, sticky=tk.W)

    def update_title(self):
        self.title('Working on {} model'.format(self.alg_box.get()))

    def alg_configuration(self):
        alg = self.alg_box.get()
        self.update_title()
        param_frame = tk.LabelFrame(self, text='{} configuration'.format(alg), width=450)
        param_frame.place(height=255, width=450, x=25, y=100)
        clf_frame = tk.LabelFrame(self, text='{} fit and score'.format(alg))
        clf_frame.place(x=25, y=360, width=450, height=150)

        def get_tree_params(outer_self, frame):
            params['criterion'] = criterion.get()
            params['max_depth'] = int(max_depth.get())
            params['min_samples_split'] = int(min_samples_split.get())
            params['min_samples_leaf'] = int(min_samples_leaf.get())
            params['min_weight_fraction_leaf'] = float(min_weight_fraction_leaf.get())
            params['random_state'] = random_state.get()
            if params['random_state'] == 'None':
                params['random_state'] = None

            params_list = ''
            for key, value in params.items():
                if key != 'random_state':
                    params_list += str(key) + ': ' + str(value) + '\n'
                else:
                    params_list += str(key) + ': ' + str(value)
            tk.Label(param_frame, text=params_list, justify=tk.LEFT).grid(row=11, column=0, columnspan=2, sticky=tk.W)
            return outer_self.get_clf(params, frame)

        tk.Button(param_frame, text='Get params', command=lambda: get_tree_params(self, clf_frame), width=14).grid(
            row=10, column=1)
        params = {}
        default_params = {
            'criterion': tk.StringVar(param_frame, value='entropy'),
            'max_depth': tk.StringVar(param_frame, value=15),
            'min_samples_split': tk.StringVar(param_frame, value=2),
            'min_samples_leaf': tk.StringVar(param_frame, value=1),
            'min_weight_fraction_leaf': tk.StringVar(param_frame, value=0.0),
            'random_state': tk.StringVar(param_frame, value='None')
        }
        if alg == 'Decision Tree':
            tk.Label(param_frame, text='criterion').grid(row=0, column=0)
            criterion = tk.Entry(param_frame, width=14, textvariable=default_params['criterion'])
            criterion.grid(row=1, column=0)

            tk.Label(param_frame, text='max_depth').grid(row=0, column=1)
            max_depth = tk.Entry(param_frame, width=14, textvariable=default_params['max_depth'])
            max_depth.grid(row=1, column=1)

            tk.Label(param_frame, text='min_samples_split').grid(row=0, column=2)
            min_samples_split = tk.Entry(param_frame, width=14, textvariable=default_params['min_samples_split'])
            min_samples_split.grid(row=1, column=2)

            tk.Label(param_frame, text='min_samples_leaf').grid(row=2, column=0)
            min_samples_leaf = tk.Entry(param_frame, width=14, textvariable=default_params['min_samples_leaf'])
            min_samples_leaf.grid(row=3, column=0)

            tk.Label(param_frame, text='min_weight_fraction_leaf').grid(row=2, column=1)
            min_weight_fraction_leaf = tk.Entry(param_frame, width=14,
                                                textvariable=default_params['min_weight_fraction_leaf'])
            min_weight_fraction_leaf.grid(row=3, column=1)

            tk.Label(param_frame, text='random_state').grid(row=2, column=2)
            random_state = tk.Entry(param_frame, width=14, textvariable=default_params['random_state'])
            random_state.grid(row=3, column=2)

    def get_clf(self, params, frame):
        self.clf = DecisionTreeClassifier(**params)

        tk.Label(frame, text='Please, set the target variable:').grid(row=0, column=0)
        target = ttk.Combobox(frame, values=list(self.dataframe.columns))
        target.grid(row=1, column=0)
        target.current(0)

        tk.Label(frame, text='Please, set the size of test data.', justify=tk.LEFT).grid(row=2, column=0)
        split = tk.Entry(frame)
        split.grid(row=3, column=0)

        def process_model(cur_frame, out_self, tar, spl):
            df = out_self.dataframe
            X = df.drop(tar, axis=1)
            y = df[tar]
            if spl != '':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(spl))
                out_self.clf.fit(X_train, y_train)
                out_self.accuracy = out_self.clf.score(X_test, y_test)
                tk.Label(cur_frame,
                         text='Model accuracy: ' + str(out_self.accuracy)
                         ).grid(row=0, column=1)

        tk.Button(frame, text='Fit the model',
                  command=lambda: process_model(frame, self, target.get(), split.get())
                  ).grid(row=4, column=0)

        def save_to_db(data, table, model, acc):
            conn = sqlite3.connect('main.sqlite3')
            cur = conn.cursor()
            try:
                query = cur.execute(f'SELECT * FROM {table} WHERE name = "{data}"').fetchall()[0]
                foreign_key = int(query[0])
                key = int(query[1])
                cur.execute('INSERT INTO Models (model_id, variant_id, task_id, name, bin_file, accuracy) '
                            'VALUES (?, ?, ?, ?, ?, ?)', (key, key, foreign_key, data + ' model', model, acc))
                conn.commit()
            finally:
                conn.close()

        tk.Button(frame, text='Save model to database',
                  command=lambda: save_to_db(self.data, self.table,
                                             serialize(self.clf),
                                             self.accuracy)).place(x=290, y=90)

    def get_data(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            query = f'SELECT table_file FROM {self.table} WHERE name = "{self.data}"'
            return deserialize(cur.execute(query).fetchall()[0][0])
        finally:
            conn.close()
