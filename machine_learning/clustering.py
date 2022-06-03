import tkinter as tk
from tkinter import ttk, messagebox
from numpy import arange
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

from tools.functions import deserialize, update_entry, save_model, get_models_list
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from fcmeans import FCM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tools.models import Model
from sqlite3 import connect
from sklearn import metrics


class ClusteringView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        # Инициализация окна
        self.w, self.h, p = self.winfo_screenwidth(), self.winfo_screenheight(), 0.4
        self.geometry(f'{int(self.w * p)}x{int(self.h * p)}')
        self.entry = entry
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        self.pd_data2 = deserialize(self.entry.table_file)
        self.pd_data3 = deserialize(self.entry.table_file)
        if isinstance(self.entry.name, tuple):
            self.entry.name = self.entry.name[0]
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
        self.alg_cb = ttk.Combobox(self.upper_frm, values=['K-means',
                                                           'DBSCAN',
                                                           'C-means',
                                                           # 'C-means(soft)',
                                                           'Иерархическая'],
                                   )
        self.lower_frm = tk.Frame(self.left_frm)
        self.lower_frm.pack(side=tk.BOTTOM)
        # Выбор алгоритма
        self.alg_cb.current(0)
        self.alg_cb.pack(side=tk.TOP, pady=10)
        ttk.Button(self.upper_frm, text='Выбрать', command=self.get_alg).pack(side=tk.TOP, pady=5)
        self.update_title()
        # Фреймы алгоритмов
        self.kmeans_frm = KMeansFrame(self.left_frm, self.entry, self.pd_data, self.pd_data2, self.pd_data3)
        self.dbscan_frm = DBSCANFrame(self.left_frm, self.entry, self.pd_data, self.pd_data2, self.pd_data3)
        self. cmeans_frm = CMeansFrame(self.left_frm, self.entry, self.pd_data, self.pd_data2, self.pd_data3)
        self.cmeansSoft_frm = CMeansSoftFrame(self.left_frm, self.entry, self.pd_data, self.pd_data2, self.pd_data3)
        self.aggl_frm = AgglFrame(self.left_frm, self.entry, self.pd_data, self.pd_data2, self.pd_data3)
        self.algorithms = {
            'K-means': self.kmeans_frm,
            'DBSCAN': self.dbscan_frm,
            'C-means': self.cmeans_frm,
            'C-means(soft)': self.cmeansSoft_frm,
            'Иерархическая': self.aggl_frm
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
        model_query = ttk.LabelFrame(self, text='Запрос к модели')
        model_query.pack(fill=tk.X, **pad)
        model_management = ttk.LabelFrame(self, text='Управление моделью')
        model_management.pack(fill=tk.BOTH, expand=1, **pad)

        # Обзор модели
        self.model_overview = ttk.Label(model_overview, text=self.get_model_overview(), justify=tk.LEFT)
        self.model_overview.pack(side=tk.LEFT, anchor=tk.N, **pad)
        # Запрос к модели
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
        # Управление моделью
        ttk.Label(model_management, text='Изменение названия модели:').grid(row=0, column=0, columnspan=2, **pad)
        self.new_name_ent = ttk.Entry(model_management)
        self.new_name_ent.grid(row=1, column=0, **pad)
        ttk.Button(model_management, text='Подтвердить', command=self.update_name).grid(row=1, column=1, **pad)
        ttk.Label(model_management, text='Удаление модели:').grid(row=2, column=0, columnspan=2, **pad)
        ttk.Button(model_management, text='Удалить модель', command=self.delete_model).grid(row=3, column=0,
                                                                                            columnspan=2, **pad)
        ttk.Label(model_management, text='Изменение описания модели:').grid(row=0, column=2, columnspan=2, **pad)
        self.description_text = tk.Text(model_management, width=30, height=5)
        self.description_text.grid(row=1, column=2, columnspan=2, rowspan=6, **pad)
        ttk.Button(model_management, text='Подтвердить', command=self.update_desc).grid(row=8, column=2, columnspan=2, **pad)

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
                    return \
                        conn.cursor().execute(query,
                                              (self.model_entry.task_id, self.model_entry.variant_id)).fetchone()[0]
            except TypeError as _ex:
                self.model_overview.configure(text='При загрузке модели возникла ошибка.')
                print(_ex)

        if not self.isSelected:
            return '\n' * 10
        elif not isinstance(self.model, (KMeans, DBSCAN, FCM, AgglomerativeClustering)):
            return "Выбранная модель не является алгоритмом кластеризации.\nПожалуйста, выберите другую модель."
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

    def update_desc(self):
        pass


# Всплывающее окно

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 47
        # 57
        y = y + cy + self.widget.winfo_rooty() +20
        # 27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


class KMeansFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data, pd_data2, pd_data3):
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.pd_data3 = pd_data3
        self.alg = KMeans
        self.lb_frm = ttk.Labelframe(self, text='Конфигурация алгоритма K-means')
        self.lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        # self.lb_frm.pack(fill=tk.BOTH, expand=1)
        # lb_frm.pack(fill=tk.BOTH, expand=1)
        self.clf_conf_frm = tk.Frame(self.lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        # self.model_lb_frm = ttk.Labelframe(self, text='Конфигурация параметров обучения')
        # self.model_lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        # self.model_frm_l = tk.Frame(self.model_lb_frm)
        # self.model_frm_l.pack(side=tk.LEFT)
        # self.model_frm_r = tk.Frame(self.model_lb_frm)
        # self.model_frm_r.pack(side=tk.RIGHT)
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)

        # elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров.
        # Для этого выберите точку перегиба на графике ниже
        # и занесите это количество в n_clusters."""
        # tk.Label(self.clf_conf_frm, text=elkan_info, justify=tk.LEFT).grid(row=0, column=0, pady=2)
        ttk.Button(self.clf_conf_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).grid(row=0, columnspan=2)

                                                                                                              # , pady=5)

        self.conf_frm = tk.Frame(self.lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.clf_conf_frm, value=8),
            'n_init': tk.StringVar(self.clf_conf_frm, value=10)
        }
        self.params = {}

        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        ttk.Label(self.clf_conf_frm, text='n_clusters').grid(row=2, column=0, columnspan=1, **pad)
        # , padx=5)
        self.n_clusters_sb = ttk.Spinbox(self.clf_conf_frm, textvariable=self.default_params['n_clusters'], from_=2, to=15, **ent_options)
        self.n_clusters_sb.grid(row=3, column=0, columnspan=1, **pad)
        # self.n_clusters_sb.grid_columnconfigure()
        # , padx=5, pady=5)
        CreateToolTip(self.n_clusters_sb, text='Кол-во кластеров')

        ttk.Label(self.clf_conf_frm, text='n_init').grid(row=2, column=1, columnspan=1, **pad)
        self.n_init_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['n_init'], **ent_options)
        self.n_init_ent.grid(row=3, column=1, columnspan=1, **pad)
        CreateToolTip(self.n_init_ent, text='Сколько раз выполнится алгоритм\n'
                                               'с разными начальными значениями\n'
                                               'центроидов')


        ttk.Label(self.clf_conf_frm, text='init').grid(row=4, column=0, columnspan=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.clf_conf_frm, values=['k-means++', 'random'], **ent_options)
        self.algorithm_cb.grid(row=5, column=0, columnspan=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['init'] = self.algorithm_cb
        CreateToolTip(self.algorithm_cb, text='Метод выбора начальных\n'
                                               'значений центров кластеров')


        ttk.Label(self.clf_conf_frm, text='algorithm').grid(row=4, column=1, columnspan=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.clf_conf_frm, values=['auto', 'full', 'elkan'], **ent_options)
        self.algorithm_cb.grid(row=5, column=1, columnspan=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb
        CreateToolTip(self.algorithm_cb, text='Используемый алгоритм k-means')


        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }
        # ttk.Button(self.btn_frm, text='Подтвердить', command=self.fit).pack(**btn_pack)
        # ttk.Button(self.btn_frm, text='ROC-кривая', command=self.get_roc).pack(**btn_pack)
        # ttk.Button(self.btn_frm, text='Открыть дерево', command=self.get_tree).pack(**btn_pack)
        # ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).pack(**btn_pack)

        ttk.Button(self.clf_conf_frm, text='Подтвердить', command=self.fit).grid(columnspan=2, padx=15, pady=8)
        self.isFitted = False

        self.accuracy = '...'  # Средняя точность
        self.acc_lbl = ttk.Label(self.clf_conf_frm, text=f'Точность модели: '+str(self.accuracy))
        self.acc_lbl.grid(columnspan=2, **pad)

        self.vis_lb_frm = ttk.LabelFrame(self.clf_conf_frm, text='Визуализация кластеров')
        self.vis_lb_frm.grid(columnspan=2)
        tk.Label(self.vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(self.vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                       **pad)
        self.col3_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col3_cb.grid(row=3, column=0, **pad)

        ttk.Button(self.vis_lb_frm, text='Профили кластеров', command=self.clstr_profile).grid(row=4, column=0, **pad)

        self.col4_cb = ttk.Combobox(self.vis_lb_frm)
        self.col4_cb.grid(row=3, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Характеристики кластера', command=self.clstr_characteristics).grid(row=4, column=1, columnspan=2, **pad)

        tk.Label(self.vis_lb_frm, text='Для сравнения кластеров выберите два кластера:').grid(row=5, column=0, columnspan=2, **pad)

        self.col5_cb = ttk.Combobox(self.vis_lb_frm)
        self.col5_cb.grid(row=6, column=0, **pad)

        self.col6_cb = ttk.Combobox(self.vis_lb_frm)
        self.col6_cb.grid(row=6, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Сравнение кластеров', command=self.clstr_compare).grid(row=7, column=0, columnspan=2, **pad)

        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).grid(**pad)

    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        self.pd_data3.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        # for pd_data3
        for name in self.pd_data3.columns:
            if (type(self.pd_data3[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data3[name] = le.fit_transform(self.pd_data3[name])
        #
        self.alg = self.get_alg()
        self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
        self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        unique_clstr = pd.unique(self.pd_data['Clusters'])
        sort_clstr = sorted(unique_clstr)
        # print(sort_clstr)
        self.accuracy = metrics.silhouette_score(self.pd_data3, self.alg.fit_predict(self.pd_data))
        self.acc_lbl.configure(text=f'Точность модели: '+str(self.accuracy))
        self.col4_cb.configure(values=sort_clstr)
        self.col5_cb.configure(values=sort_clstr)
        self.col6_cb.configure(values=sort_clstr)


    def get_params(self):
        params = {}
        for param, obj in self.default_params.items():
            try:
                if obj.get() == 'None':
                    params[param] = None
                else:
                    params[param] = int(obj.get())
            except ValueError:
                params[param] = obj.get()
        return params

    def get_alg(self):
        params = self.get_params()
        if isinstance(self.alg, KMeans):
            self.alg = KMeans
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            centroids = self.alg.cluster_centers_
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            sns.scatterplot(x=centroids[:, self.pd_data.columns.get_loc(col1)],
                            y=centroids[:, self.pd_data.columns.get_loc(col2)],
                            color='black', marker='s')
            plt.show()

    # def get_sort_clstr(self):
    #     if self.isFitted:
    #         # sort_clstr = []
    #         self.alg = self.get_alg()
    #         self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
    #         unique_clstr = pd.unique(self.pd_data['Clusters'])
    #         sort_clstr = sorted(unique_clstr)
    #         print(sort_clstr)
    #         print(self.pd_data['Clusters'])
    #         return sort_clstr

    def clstr_profile(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        len_clstr = len(unique_clstr)
        sort_clstr = sorted(unique_clstr)
        cols = []
        cols.insert(0, (self.col3_cb.get()))
        df_one = DataFrame()
        res1 = DataFrame()
        for j in range(len(cols)):
            unique_attr_2 = pd.unique(self.pd_data2[cols[j]])
            unique_amount_norm = self.pd_data2[cols[j]].value_counts().sort_index(ascending=False).to_frame()
            df_all = unique_amount_norm.T
            for i in sort_clstr:
                for index, row in self.pd_data2.iterrows():
                    if row['Clusters'] == sort_clstr[i]:
                        df_one = pd.concat([df_one, row.to_frame().T], ignore_index=True)
                res1 = pd.concat(
                    [res1, df_one[cols[j]].value_counts(normalize=True).sort_index(ascending=False).to_frame().T])
                df_one = DataFrame()
            res2 = res1
            res2.index = Series(sort_clstr)
            if (type(unique_attr_2[1]) == str):
                f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                f.suptitle('Атрибут ' + cols[j], fontsize=22)
                df_all.plot.bar(ax=ax[0], stacked='True', alpha=0.5)
                ax[0].set_title('Заполнение по всем кластерам (' + str(len_clstr) + ')', fontsize=12)
                res2.plot.bar(ax=ax[1], stacked='True', alpha=0.5)
                ax[1].set_title('Кластеры', fontsize=12)
                plt.show()
            else:
                if cols[j] == 'Clusters':
                    pass
                else:
                    f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                    f.suptitle('Атрибут ' + cols[j], fontsize=22)
                    sns.boxplot(ax=ax[0], y=self.pd_data2[cols[j]], data=self.pd_data2, notch=False, linewidth=2.5)
                    ax[0].set_title("Диапазон значений (Y) атрибута " + cols[j] + "(X)", fontsize=12)
                    sns.boxplot(ax=ax[1], x='Clusters', y=cols[j], data=self.pd_data2, notch=False)
                    ax[1].set_title('Рапределение значений атрибута ' + cols[j] + ' (Y) в каждом кластере (X)',
                                    fontsize=12)
                    plt.show()
            res1 = DataFrame()


    def clstr_characteristics(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        sort_clstr = sorted(unique_clstr)
        df_comp = DataFrame()
        df_comp_all = DataFrame()
        sort_clstr_2 = []
        sort_clstr_2.insert(0, (int(self.col4_cb.get())))
        for i in sort_clstr_2:
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == sort_clstr[i]:
                    df_comp = pd.concat([df_comp, row.to_frame().T], ignore_index=True)
            del df_comp['Clusters']
            col = df_comp.columns
            for j in col:
                unique_amount = round(df_comp[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_all = (pd.concat([df_comp_all, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
            # Draw plot
            fig, ax = plt.subplots(figsize=(16, 18), facecolor='white', dpi=80)
            ax.vlines(x=df_comp_all.index, ymin=0, ymax=df_comp_all.prob, color='firebrick', alpha=0.7, linewidth=50)
            # Annotate Text
            for k, prob in enumerate(df_comp_all.prob):
                ax.text(k, prob + 0.5, round(prob, 1), horizontalalignment='center')
            # Title, Label, Ticks and Ylim
            ax.set_title('Влияние значений атрибутов на кластер ' + str(sort_clstr[i]), fontdict={'size': 22})
            ax.set(ylabel='Probability', ylim=(0, 100))
            plt.xticks(df_comp_all.index, rotation=60, horizontalalignment='right', fontsize=12)
            plt.show()
            # clear df
            df_comp = DataFrame()
            df_comp_all = DataFrame()

    def clstr_compare(self):
        df_comp_clstrs = DataFrame()
        df_comp_two_clstrs = DataFrame()
        df_comp_two_clstrs_all = DataFrame()
        two_clstr = []
        two_clstr.insert(0, (int(self.col5_cb.get())))
        two_clstr.insert(1, (int(self.col6_cb.get())))
        # one = self.col5_cb.get()
        # two = self.col6_cb.get()
        # two_clstr = []
        # two_clstr = [one, two]
        for i in range(2):
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == two_clstr[i]:
                    df_comp_clstrs = pd.concat([df_comp_clstrs, row.to_frame().T], ignore_index=True)
            del df_comp_clstrs['Clusters']
            col = df_comp_clstrs.columns
            for j in col:
                unique_amount = round(df_comp_clstrs[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = '(' + str(two_clstr[i]) + ') ' + j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_two_clstrs = (
                    pd.concat([df_comp_two_clstrs, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
                df_comp_two_clstrs['Clusters'] = two_clstr[i]
            df_comp_two_clstrs_all = pd.concat([df_comp_two_clstrs_all, df_comp_two_clstrs]).sort_values(by='prob',
                                                                                                         ascending=False)
            # clear df
            df_comp_clstrs = DataFrame()
            df_comp_two_clstrs = DataFrame()
        # Замена значений пеовго кластера на "-"
        df_comp_two_clstrs_all.loc[
            (df_comp_two_clstrs_all.Clusters == two_clstr[0]), ('prob')] = df_comp_two_clstrs_all.prob * (-1)
        del df_comp_two_clstrs_all['Clusters']
        # Draw plot
        df_comp_two_clstrs_all['colors'] = ['red' if x < 0 else 'green' for x in df_comp_two_clstrs_all['prob']]
        plt.figure(figsize=(18, 10), dpi=80)
        plt.hlines(y=df_comp_two_clstrs_all.index, xmin=0, xmax=df_comp_two_clstrs_all.prob,
                   color=df_comp_two_clstrs_all.colors, alpha=0.4, linewidth=5)
        # Decorations
        plt.gca().set(ylabel='$Атрибуты и их значения$', xlabel='$Вероятность$')
        plt.yticks(df_comp_two_clstrs_all.index, fontsize=12)
        plt.title('Сравнение кластеров ' + str(two_clstr[0]) + ' и ' + str(two_clstr[1]), fontdict={'size': 20})
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()


    def get_elkan_graph(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        inertia = []
        for i in range(1, 11):
            inertia.append(KMeans(n_clusters=i).fit(self.pd_data).inertia_)
        plt.style.use('seaborn-whitegrid')
        plt.figure(1, figsize=(10, 5))
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.95)
        plt.plot(arange(1, 11), inertia, 'o')
        plt.plot(arange(1, 11), inertia, '-', alpha=0.5)
        plt.xlabel('Количество кластеров', {'fontsize': 15})
        plt.ylabel('WCSS', {'fontsize': 15})
        plt.title('Метод локтя для поиска оптимального кол-ва кластеров', {'fontsize': 15})
        plt.show()
        # InertiaView(inertia)


    def save(self):
        pad = {
            'padx': 5,
            'pady': 5
        }
        window = tk.Toplevel(self.master)
        ttk.Label(window, text='Название модели:').grid(row=0, column=0,  **pad)
        name_ent = ttk.Entry(window)
        name_ent.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Label(window, text='Описание модели:').grid(row=1, column=0, **pad)
        desc_text = tk.Text(window, width=30, height=5)
        desc_text.grid(row=1, column=1, **pad)
        ttk.Button(window, text='Сохранить',
                   command=lambda: [save_model(self.entry, self.alg, self.accuracy,
                                                                        name=name_ent.get(),
                                                                        desc=desc_text.get("1.0", "end-1c")),
                                    window.destroy()]).grid(row=2, column=0, columnspan=2, **pad)


#
# class ClusteringView(tk.Tk):
#     def __init__(self, entry):
#         tk.Tk.__init__(self)
#         self.geometry('500x500')
#         self.entry = entry
#         update_entry(self.entry)
#         self.pd_data = deserialize(self.entry.table_file)
#         self.pd_data2 = deserialize(self.entry.table_file)
#
#         upper_frm = tk.LabelFrame(self, text='Выбор алгоритма')
#         upper_frm.pack(fill=tk.X)
#         alg = ['K Means', 'DBSCAN', 'C means', 'Agglomerative']
#         self.alg_cb = ttk.Combobox(upper_frm, values=alg)
#         self.alg_cb.pack(pady=10)
#         self.alg_cb.current(0)
#         ttk.Button(upper_frm, text='Выбрать', command=self.get_alg).pack(pady=5)
#         kmeans_frm = KMeansFrame(self, self.entry, self.pd_data, self.pd_data2)
#         dbscan_frm = DBSCANFrame(self, self.entry, self.pd_data)
#         cmeans_frm = CMeansFrame(self, self.entry, self.pd_data)
#         aggl_frm = AgglFrame(self, self.entry, self.pd_data)
#         self.algorithms = dict(zip(alg, [kmeans_frm, dbscan_frm, cmeans_frm, aggl_frm]))
#         self.current_alg = None
#         self.get_alg()
#
#     def get_alg(self):
#         alg = self.alg_cb.get()
#         self.title(f'Работа с {self.entry.name} с помощью алгоритма "{alg}"')
#         if self.current_alg is not None:
#             self.current_alg.pack_forget()
#         self.algorithms[alg].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#         self.current_alg = self.algorithms[alg]
#
# class KMeansFrame(tk.Frame):
#     def __init__(self, parent, entry, pd_data, pd_data2):
#         tk.Frame.__init__(self, parent)
#         self.entry = entry
#         self.pd_data = pd_data
#         self.pd_data2 = pd_data2
#         self.alg = KMeans
#         #self.df_clstr = df_clstr
#         lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма K Means')
#         lb_frm.pack(fill=tk.BOTH, expand=1)
#
#         elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров.
#         Для этого выберите точку перегиба на графике ниже
#         и занесите это количество в n_clusters."""
#         tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
#         ttk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)
#
#         self.conf_frm = tk.Frame(lb_frm)
#         self.conf_frm.pack()
#
#         self.default_params = {
#             'n_clusters': tk.IntVar(self.conf_frm, value=8),
#             'n_init': tk.StringVar(self.conf_frm, value=10)
#             # 'random_state': tk.StringVar(self.conf_frm, value='None')
#         }
#         opt = {
#             'width': 20,
#             'justify': tk.CENTER
#         }
#         pad = {
#             'padx': 15,
#             'pady': 3
#         }
#         tk.Label(self.conf_frm, text='n_clusters').grid(row=0, column=0, **pad)
#         self.n_clusters_sb = tk.Spinbox(self.conf_frm, textvariable=self.default_params['n_clusters'],
#                                         from_=2, to=15, **opt)
#         self.n_clusters_sb.grid(row=1, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='n_init').grid(row=0, column=1, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['n_init'], **opt)
#         self.n_init_ent.grid(row=1, column=1, **pad)
#
#         # tk.Label(self.conf_frm, text='random_state').grid(row=2, column=0, **pad)
#         # self.random_state_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['random_state'], **opt)
#         # self.random_state_ent.grid(row=3, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='init').grid(row=2, column=0, **pad)
#         self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['k-means++', 'random'], **opt)
#         self.algorithm_cb.grid(row=3, column=0, **pad)
#         self.algorithm_cb.current(0)
#         self.default_params['init'] = self.algorithm_cb
#
#         tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
#         self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'full', 'elkan'], **opt)
#         self.algorithm_cb.grid(row=3, column=1, **pad)
#         self.algorithm_cb.current(0)
#         self.default_params['algorithm'] = self.algorithm_cb
#
#         ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
#         self.isFitted = False
#
#         vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
#         vis_lb_frm.pack(**pad)
#         tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
#                                                                                           **pad)
#         self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col1_cb.grid(row=1, column=0, **pad)
#         self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col2_cb.grid(row=1, column=1, **pad)
#         ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
#                                                                                       **pad)
#
#         # df_clstr = pd_data
#         # self.df_clstr['clstr'] = self.alg.fit_predict(self.df_clstr)
#         # unique_clstr = pd.unique(pd_data['Clusters'])
#         # sort_clstr = sorted(unique_clstr)
#
#
#         self.col3_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col3_cb.grid(row=3, column=0, **pad)
#
#         ttk.Button(vis_lb_frm, text = 'Профили кластеров', command = self.clstr_profile).grid(row=4, column=0,
#                                                                                       **pad)
#
#         # self.col4_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
#         self.col4_cb = ttk.Combobox(vis_lb_frm)
#
#         # self.col4_cb = ttk.Combobox(vis_lb_frm)
#         self.col4_cb.grid(row=3, column=1, **pad)
#
#         ttk.Button(vis_lb_frm, text = 'Характеристики кластера', command = self.clstr_characteristics).grid(row=4, column=1, columnspan=2,
#                                                                                       **pad)
#         # self.col5_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
#         self.col5_cb = ttk.Combobox(vis_lb_frm)
#         self.col5_cb.grid(row=5, column=0, **pad)
#
#         # self.col6_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
#         self.col6_cb = ttk.Combobox(vis_lb_frm)
#         self.col6_cb.grid(row=5, column=1, **pad)
#
#         ttk.Button(vis_lb_frm, text='Сравнение кластеров', command = self.clstr_compare).grid(row=6, column=0, columnspan=2,
#                                                                                       **pad)
#         #
#         # ttk.Button(vis_lb_frm, text='Обновить номера кластеров', command=self.col5_cb).grid(row=7, column=0,
#         #                                                                                     columnspan=2,
#         #                                                                                     **pad)
#
#         ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)
#
#
#     def fit(self):
#         self.pd_data.dropna(axis=0, inplace=True, how='any')
#         str_col_names = []
#         for name in self.pd_data.columns:
#             if (type(self.pd_data[name][1]) == str):
#                 str_col_names.append(name)
#         le = LabelEncoder()
#         for name in str_col_names:
#             self.pd_data[name] = le.fit_transform(self.pd_data[name])
#         self.alg = self.get_alg()
#         self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
#         self.pd_data2['Clusters'] = self.pd_data['Clusters']
#         self.isFitted = True
#         print('Модель обучена')
#         unique_clstr = pd.unique(self.pd_data['Clusters'])
#         sort_clstr = sorted(unique_clstr)
#         print(sort_clstr)
#         self.col4_cb.configure(values=sort_clstr)
#         self.col5_cb.configure(values=sort_clstr)
#         self.col6_cb.configure(values=sort_clstr)
#         # return sort_clstr
#
#     def get_params(self):
#         params = {}
#         for param, obj in self.default_params.items():
#             try:
#                 if obj.get() == 'None':
#                     params[param] = None
#                 else:
#                     params[param] = int(obj.get())
#             except ValueError:
#                 params[param] = obj.get()
#         return params
#
#     def get_alg(self):
#         params = self.get_params()
#         if isinstance(self.alg, KMeans):
#             self.alg = KMeans
#         return self.alg(**params)
#
#     def get_plot(self):
#         if self.isFitted:
#             col1 = self.col1_cb.get()
#             col2 = self.col2_cb.get()
#             centroids = self.alg.cluster_centers_
#             plt.figure(1, figsize=(10, 6))
#             sns.set_theme()
#             sns.set_style('whitegrid')
#             sns.set_context('talk')
#             sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
#             sns.scatterplot(x=centroids[:, self.pd_data.columns.get_loc(col1)],
#                             y=centroids[:, self.pd_data.columns.get_loc(col2)],
#                             color='black', marker='s')
#             plt.show()
#
#     # def get_sort_clstr(self):
#     #     if self.isFitted:
#     #         # sort_clstr = []
#     #         self.alg = self.get_alg()
#     #         self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
#     #         unique_clstr = pd.unique(self.pd_data['Clusters'])
#     #         sort_clstr = sorted(unique_clstr)
#     #         print(sort_clstr)
#     #         print(self.pd_data['Clusters'])
#     #         return sort_clstr
#
#     def clstr_profile(self):
#         unique_clstr = pd.unique(self.pd_data2['Clusters'])
#         len_clstr = len(unique_clstr)
#         sort_clstr = sorted(unique_clstr)
#         cols = []
#         cols.insert(0, (self.col3_cb.get()))
#         df_one = DataFrame()
#         res1 = DataFrame()
#         for j in range(len(cols)):
#             unique_attr_2 = pd.unique(self.pd_data2[cols[j]])
#             unique_amount_norm = self.pd_data2[cols[j]].value_counts().sort_index(ascending=False).to_frame()
#             df_all = unique_amount_norm.T
#             for i in sort_clstr:
#                 for index, row in self.pd_data2.iterrows():
#                     if row['Clusters'] == sort_clstr[i]:
#                         df_one = pd.concat([df_one, row.to_frame().T], ignore_index=True)
#                 res1 = pd.concat(
#                     [res1, df_one[cols[j]].value_counts(normalize=True).sort_index(ascending=False).to_frame().T])
#                 df_one = DataFrame()
#             res2 = res1
#             res2.index = Series(sort_clstr)
#             if (type(unique_attr_2[1]) == str):
#                 f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
#                 f.suptitle('Атрибут ' + cols[j], fontsize=22)
#                 df_all.plot.bar(ax=ax[0], stacked='True', alpha=0.5)
#                 ax[0].set_title('Заполнение по всем кластерам (' + str(len_clstr) + ')', fontsize=12)
#                 res2.plot.bar(ax=ax[1], stacked='True', alpha=0.5)
#                 ax[1].set_title('Кластеры', fontsize=12)
#                 plt.show()
#             else:
#                 if cols[j] == 'Clusters':
#                     pass
#                 else:
#                     f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
#                     f.suptitle('Атрибут ' + cols[j], fontsize=22)
#                     sns.boxplot(ax=ax[0], y=self.pd_data2[cols[j]], data=self.pd_data2, notch=False, linewidth=2.5)
#                     ax[0].set_title("Диапазон значений (Y) атрибута " + cols[j] + "(X)", fontsize=12)
#                     sns.boxplot(ax=ax[1], x='Clusters', y=cols[j], data=self.pd_data2, notch=False)
#                     ax[1].set_title('Рапределение значений атрибута ' + cols[j] + ' (Y) в каждом кластере (X)',
#                                     fontsize=12)
#                     plt.show()
#             res1 = DataFrame()
#
#
#     def clstr_characteristics(self):
#         unique_clstr = pd.unique(self.pd_data2['Clusters'])
#         sort_clstr = sorted(unique_clstr)
#         df_comp = DataFrame()
#         df_comp_all = DataFrame()
#         sort_clstr_2 = []
#         sort_clstr_2.insert(0, (int(self.col4_cb.get())))
#         for i in sort_clstr_2:
#             for index, row in self.pd_data2.iterrows():
#                 if row['Clusters'] == sort_clstr[i]:
#                     df_comp = pd.concat([df_comp, row.to_frame().T], ignore_index=True)
#             del df_comp['Clusters']
#             col = df_comp.columns
#             for j in col:
#                 unique_amount = round(df_comp[j].value_counts(normalize=True).to_frame() * 100, 1)
#                 unique_amount.index = j + ' = ' + unique_amount.index.astype(str)
#                 unique_amount = unique_amount.rename(columns={j: 'prob'})
#                 df_comp_all = (pd.concat([df_comp_all, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
#             # Draw plot
#             fig, ax = plt.subplots(figsize=(16, 18), facecolor='white', dpi=80)
#             ax.vlines(x=df_comp_all.index, ymin=0, ymax=df_comp_all.prob, color='firebrick', alpha=0.7, linewidth=50)
#             # Annotate Text
#             for k, prob in enumerate(df_comp_all.prob):
#                 ax.text(k, prob + 0.5, round(prob, 1), horizontalalignment='center')
#             # Title, Label, Ticks and Ylim
#             ax.set_title('Влияние значений атрибутов на кластер ' + str(sort_clstr[i]), fontdict={'size': 22})
#             ax.set(ylabel='Probability', ylim=(0, 100))
#             plt.xticks(df_comp_all.index, rotation=60, horizontalalignment='right', fontsize=12)
#             plt.show()
#             # clear df
#             df_comp = DataFrame()
#             df_comp_all = DataFrame()
#
#     def clstr_compare(self):
#         df_comp_clstrs = DataFrame()
#         df_comp_two_clstrs = DataFrame()
#         df_comp_two_clstrs_all = DataFrame()
#         two_clstr = []
#         two_clstr.insert(0, (int(self.col5_cb.get())))
#         two_clstr.insert(1, (int(self.col6_cb.get())))
#         # one = self.col5_cb.get()
#         # two = self.col6_cb.get()
#         # two_clstr = []
#         # two_clstr = [one, two]
#         for i in range(2):
#             for index, row in self.pd_data2.iterrows():
#                 if row['Clusters'] == two_clstr[i]:
#                     df_comp_clstrs = pd.concat([df_comp_clstrs, row.to_frame().T], ignore_index=True)
#             del df_comp_clstrs['Clusters']
#             col = df_comp_clstrs.columns
#             for j in col:
#                 unique_amount = round(df_comp_clstrs[j].value_counts(normalize=True).to_frame() * 100, 1)
#                 unique_amount.index = '(' + str(two_clstr[i]) + ') ' + j + ' = ' + unique_amount.index.astype(str)
#                 unique_amount = unique_amount.rename(columns={j: 'prob'})
#                 df_comp_two_clstrs = (
#                     pd.concat([df_comp_two_clstrs, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
#                 df_comp_two_clstrs['Clusters'] = two_clstr[i]
#             df_comp_two_clstrs_all = pd.concat([df_comp_two_clstrs_all, df_comp_two_clstrs]).sort_values(by='prob',
#                                                                                                          ascending=False)
#             # clear df
#             df_comp_clstrs = DataFrame()
#             df_comp_two_clstrs = DataFrame()
#         # Замена значений пеовго кластера на "-"
#         df_comp_two_clstrs_all.loc[
#             (df_comp_two_clstrs_all.Clusters == two_clstr[0]), ('prob')] = df_comp_two_clstrs_all.prob * (-1)
#         del df_comp_two_clstrs_all['Clusters']
#         # Draw plot
#         df_comp_two_clstrs_all['colors'] = ['red' if x < 0 else 'green' for x in df_comp_two_clstrs_all['prob']]
#         plt.figure(figsize=(18, 10), dpi=80)
#         plt.hlines(y=df_comp_two_clstrs_all.index, xmin=0, xmax=df_comp_two_clstrs_all.prob,
#                    color=df_comp_two_clstrs_all.colors, alpha=0.4, linewidth=5)
#         # Decorations
#         plt.gca().set(ylabel='$Атрибуты и их значения$', xlabel='$Вероятность$')
#         plt.yticks(df_comp_two_clstrs_all.index, fontsize=12)
#         plt.title('Сравнение кластеров ' + str(two_clstr[0]) + ' и ' + str(two_clstr[1]), fontdict={'size': 20})
#         plt.grid(linestyle='--', alpha=0.5)
#         plt.show()
#
#
#     def get_elkan_graph(self):
#         inertia = []
#         for i in range(1, 11):
#             inertia.append(KMeans(n_clusters=i).fit(self.pd_data).inertia_)
#         plt.style.use('seaborn-whitegrid')
#         plt.figure(1, figsize=(10, 5))
#         plt.subplots_adjust(left=0.05, bottom=0.1, right=0.98, top=0.95)
#         plt.plot(arange(1, 11), inertia, 'o')
#         plt.plot(arange(1, 11), inertia, '-', alpha=0.5)
#         plt.xlabel('Количество кластеров', {'fontsize': 15})
#         plt.ylabel('WCSS', {'fontsize': 15})
#         plt.title('Метод локтя для поиска оптимального кол-ва кластеров', {'fontsize': 15})
#         plt.show()
#         # InertiaView(inertia)
#
#     def save(self):
#         save_model(self.entry, self.alg)
#         print('Модель сохранена!')
#

class DBSCANFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data, pd_data2, pd_data3):
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.pd_data3 = pd_data3
        self.alg = DBSCAN
        self.lb_frm = ttk.Labelframe(self, text='Конфигурация алгоритма DBSCAN')
        self.lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.clf_conf_frm = tk.Frame(self.lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)

        self.conf_frm = tk.Frame(self.lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'eps': tk.StringVar(self.conf_frm, value=0.5),
            'min_samples': tk.IntVar(self.conf_frm, value=5),
            'leaf_size': tk.IntVar(self.conf_frm, value=30)
        }
        self.params = {}

        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        ttk.Label(self.clf_conf_frm, text='eps').grid(row=2, column=0, columnspan=1, **pad)
        self.n_clusters_sb = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['eps'], **ent_options)
        self.n_clusters_sb.grid(row=3, column=0, columnspan=1, **pad)
        CreateToolTip(self.n_clusters_sb, text='Радиус окружности, с помощью которой\n'
                                               'ведется поиск ближайших точек')

        ttk.Label(self.clf_conf_frm, text='min_samples').grid(row=2, column=1, columnspan=1, **pad)
        self.n_init_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['min_samples'], **ent_options)
        self.n_init_ent.grid(row=3, column=1, columnspan=1, **pad)
        CreateToolTip(self.n_init_ent, text='Количество точек, коорое должно попадать\n'
                                            'в радиус, чтобы точка, от которой построен\n'
                                            'радиус считалась основной')

        ttk.Label(self.clf_conf_frm, text='leaf_size').grid(row=4, column=0, columnspan=1, **pad)
        self.n_init_ent = tk.Entry(self.clf_conf_frm, textvariable=self.default_params['leaf_size'], **ent_options)
        self.n_init_ent.grid(row=5, column=0, columnspan=1, **pad)
        CreateToolTip(self.n_init_ent, text='Размер листьев, передаваемый в BallTree или KDTree.\n'
                                            'Может влиять на скорость построения')

        ttk.Label(self.clf_conf_frm, text='algorithm').grid(row=4, column=1, columnspan=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.clf_conf_frm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], **ent_options)
        self.algorithm_cb.grid(row=5, column=1, columnspan=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb
        CreateToolTip(self.algorithm_cb, text='Алгоритм, который будет использоваться\n'
                                              'модулем Nearest Neighbors для вычисления\n'
                                              'расстояний между точками и поиска соседей')

        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }

        ttk.Button(self.clf_conf_frm, text='Подтвердить', command=self.fit).grid(columnspan=2, padx=15, pady=8)
        self.isFitted = False

        self.accuracy = '...'  # Средняя точность
        self.acc_lbl = ttk.Label(self.clf_conf_frm, text=f'Точность модели: '+str(self.accuracy))
        self.acc_lbl.grid(columnspan=2, **pad)

        self.vis_lb_frm = ttk.LabelFrame(self.clf_conf_frm, text='Визуализация кластеров')
        self.vis_lb_frm.grid(columnspan=2)
        tk.Label(self.vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(self.vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                       **pad)
        self.col3_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col3_cb.grid(row=3, column=0, **pad)

        ttk.Button(self.vis_lb_frm, text='Профили кластеров', command=self.clstr_profile).grid(row=4, column=0, **pad)

        self.col4_cb = ttk.Combobox(self.vis_lb_frm)
        self.col4_cb.grid(row=3, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Характеристики кластера', command=self.clstr_characteristics).grid(row=4, column=1, columnspan=2, **pad)

        tk.Label(self.vis_lb_frm, text='Для сравнения кластеров выберите два кластера:').grid(row=5, column=0, columnspan=2, **pad)

        self.col5_cb = ttk.Combobox(self.vis_lb_frm)
        self.col5_cb.grid(row=6, column=0, **pad)

        self.col6_cb = ttk.Combobox(self.vis_lb_frm)
        self.col6_cb.grid(row=6, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Сравнение кластеров', command=self.clstr_compare).grid(row=7, column=0, columnspan=2, **pad)

        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).grid(**pad)

    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        self.pd_data3.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        # for pd_data3
        for name in self.pd_data3.columns:
            if (type(self.pd_data3[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data3[name] = le.fit_transform(self.pd_data3[name])
        #
        self.alg = self.get_alg()
        # if self.alg.fit_predict(self.pd_data) != -1:
        self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
        self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        unique_clstr = pd.unique(self.pd_data['Clusters'])
        sort_clstr = sorted(unique_clstr)
        # print(sort_clstr)
        # accuracy
        if len(sort_clstr) > 1:
            self.accuracy = metrics.silhouette_score(self.pd_data3, self.alg.fit_predict(self.pd_data))
        self.acc_lbl.configure(text=f'Точность модели: '+str(self.accuracy))
        self.col4_cb.configure(values=sort_clstr)
        self.col5_cb.configure(values=sort_clstr)
        self.col6_cb.configure(values=sort_clstr)


    def get_params(self):
        params = {}
        for param, obj in self.default_params.items():
            try:
                if obj.get() == 'None':
                    params[param] = None
                elif param == 'eps':
                    params[param] = float(obj.get())
                else:
                    params[param] = int(obj.get())
            except ValueError:
                params[param] = obj.get()
        return params

    def get_alg(self):
        params = self.get_params()
        if isinstance(self.alg, DBSCAN):
            self.alg = DBSCAN
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            plt.show()

    def clstr_profile(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        len_clstr = len(unique_clstr)
        sort_clstr = sorted(unique_clstr)
        cols = []
        cols.insert(0, (self.col3_cb.get()))
        df_one = DataFrame()
        res1 = DataFrame()
        for j in range(len(cols)):
            unique_attr_2 = pd.unique(self.pd_data2[cols[j]])
            unique_amount_norm = self.pd_data2[cols[j]].value_counts().sort_index(ascending=False).to_frame()
            df_all = unique_amount_norm.T
            for i in sort_clstr:
                for index, row in self.pd_data2.iterrows():
                    if row['Clusters'] == sort_clstr[i]:
                        df_one = pd.concat([df_one, row.to_frame().T], ignore_index=True)
                res1 = pd.concat(
                    [res1, df_one[cols[j]].value_counts(normalize=True).sort_index(ascending=False).to_frame().T])
                df_one = DataFrame()
            res2 = res1
            res2.index = Series(sort_clstr)
            if (type(unique_attr_2[1]) == str):
                f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                f.suptitle('Атрибут ' + cols[j], fontsize=22)
                df_all.plot.bar(ax=ax[0], stacked='True', alpha=0.5)
                ax[0].set_title('Заполнение по всем кластерам (' + str(len_clstr) + ')', fontsize=12)
                res2.plot.bar(ax=ax[1], stacked='True', alpha=0.5)
                ax[1].set_title('Кластеры', fontsize=12)
                plt.show()
            else:
                if cols[j] == 'Clusters':
                    pass
                else:
                    f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                    f.suptitle('Атрибут ' + cols[j], fontsize=22)
                    sns.boxplot(ax=ax[0], y=self.pd_data2[cols[j]], data=self.pd_data2, notch=False, linewidth=2.5)
                    ax[0].set_title("Диапазон значений (Y) атрибута " + cols[j] + "(X)", fontsize=12)
                    sns.boxplot(ax=ax[1], x='Clusters', y=cols[j], data=self.pd_data2, notch=False)
                    ax[1].set_title('Рапределение значений атрибута ' + cols[j] + ' (Y) в каждом кластере (X)',
                                    fontsize=12)
                    plt.show()
            res1 = DataFrame()


    def clstr_characteristics(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        sort_clstr = sorted(unique_clstr)
        df_comp = DataFrame()
        df_comp_all = DataFrame()
        sort_clstr_2 = []
        sort_clstr_2.insert(0, (int(self.col4_cb.get())))
        for i in sort_clstr_2:
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == sort_clstr[i]:
                    df_comp = pd.concat([df_comp, row.to_frame().T], ignore_index=True)
            del df_comp['Clusters']
            col = df_comp.columns
            for j in col:
                unique_amount = round(df_comp[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_all = (pd.concat([df_comp_all, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
            # Draw plot
            fig, ax = plt.subplots(figsize=(16, 18), facecolor='white', dpi=80)
            ax.vlines(x=df_comp_all.index, ymin=0, ymax=df_comp_all.prob, color='firebrick', alpha=0.7, linewidth=50)
            # Annotate Text
            for k, prob in enumerate(df_comp_all.prob):
                ax.text(k, prob + 0.5, round(prob, 1), horizontalalignment='center')
            # Title, Label, Ticks and Ylim
            ax.set_title('Влияние значений атрибутов на кластер ' + str(sort_clstr[i]), fontdict={'size': 22})
            ax.set(ylabel='Probability', ylim=(0, 100))
            plt.xticks(df_comp_all.index, rotation=60, horizontalalignment='right', fontsize=12)
            plt.show()
            # clear df
            df_comp = DataFrame()
            df_comp_all = DataFrame()

    def clstr_compare(self):
        df_comp_clstrs = DataFrame()
        df_comp_two_clstrs = DataFrame()
        df_comp_two_clstrs_all = DataFrame()
        two_clstr = []
        two_clstr.insert(0, (int(self.col5_cb.get())))
        two_clstr.insert(1, (int(self.col6_cb.get())))
        for i in range(2):
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == two_clstr[i]:
                    df_comp_clstrs = pd.concat([df_comp_clstrs, row.to_frame().T], ignore_index=True)
            del df_comp_clstrs['Clusters']
            col = df_comp_clstrs.columns
            for j in col:
                unique_amount = round(df_comp_clstrs[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = '(' + str(two_clstr[i]) + ') ' + j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_two_clstrs = (
                    pd.concat([df_comp_two_clstrs, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
                df_comp_two_clstrs['Clusters'] = two_clstr[i]
            df_comp_two_clstrs_all = pd.concat([df_comp_two_clstrs_all, df_comp_two_clstrs]).sort_values(by='prob',
                                                                                                         ascending=False)
            # clear df
            df_comp_clstrs = DataFrame()
            df_comp_two_clstrs = DataFrame()
        # Замена значений пеовго кластера на "-"
        df_comp_two_clstrs_all.loc[
            (df_comp_two_clstrs_all.Clusters == two_clstr[0]), ('prob')] = df_comp_two_clstrs_all.prob * (-1)
        del df_comp_two_clstrs_all['Clusters']
        # Draw plot
        df_comp_two_clstrs_all['colors'] = ['red' if x < 0 else 'green' for x in df_comp_two_clstrs_all['prob']]
        plt.figure(figsize=(18, 10), dpi=80)
        plt.hlines(y=df_comp_two_clstrs_all.index, xmin=0, xmax=df_comp_two_clstrs_all.prob,
                   color=df_comp_two_clstrs_all.colors, alpha=0.4, linewidth=5)
        # Decorations
        plt.gca().set(ylabel='$Атрибуты и их значения$', xlabel='$Вероятность$')
        plt.yticks(df_comp_two_clstrs_all.index, fontsize=12)
        plt.title('Сравнение кластеров ' + str(two_clstr[0]) + ' и ' + str(two_clstr[1]), fontdict={'size': 20})
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()

    def save(self):
        pad = {
            'padx': 5,
            'pady': 5
        }
        window = tk.Toplevel(self.master)
        ttk.Label(window, text='Название модели:').grid(row=0, column=0,  **pad)
        name_ent = ttk.Entry(window)
        name_ent.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Label(window, text='Описание модели:').grid(row=1, column=0, **pad)
        desc_text = tk.Text(window, width=30, height=5)
        desc_text.grid(row=1, column=1, **pad)
        ttk.Button(window, text='Сохранить',
                   command=lambda: [save_model(self.entry, self.alg,
                                               # self.accuracy,
                                                                        name=name_ent.get(),
                                                                        desc=desc_text.get("1.0", "end-1c")),
                                    window.destroy()]).grid(row=2, column=0, columnspan=2, **pad)



# class DBSCANFrame(tk.Frame):
#     def __init__(self, parent, entry, pd_data):
#         tk.Frame.__init__(self, parent)
#         self.entry = entry
#         self.pd_data = pd_data
#         self.alg = DBSCAN
#         lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма DBSCAN')
#         lb_frm.pack(fill=tk.BOTH, expand=1)
#
#         self.conf_frm = tk.Frame(lb_frm)
#         self.conf_frm.pack()
#
#         self.default_params = {
#             'eps': tk.StringVar(self.conf_frm, value=0.5),
#             'min_samples': tk.IntVar(self.conf_frm, value=5),
#             'leaf_size': tk.IntVar(self.conf_frm, value=30)
#         }
#         opt = {
#             'width': 20,
#             'justify': tk.CENTER
#         }
#         pad = {
#             'padx': 15,
#             'pady': 3
#         }
#         # tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
#         # self.n_clusters_sb = tk.Spinbox(self.conf_frm, textvariable=self.default_params['n_clusters'],
#         #                                 from_=2, to=15, **opt)
#         # self.n_clusters_sb.grid(row=1, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['eps'], **opt)
#         self.n_init_ent.grid(row=1, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='min_samples').grid(row=0, column=1, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['min_samples'], **opt)
#         self.n_init_ent.grid(row=1, column=1, **pad)
#
#         tk.Label(self.conf_frm, text='leaf_size').grid(row=2, column=0, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['leaf_size'], **opt)
#         self.n_init_ent.grid(row=3, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
#         self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], **opt)
#         self.algorithm_cb.grid(row=3, column=1, **pad)
#         self.algorithm_cb.current(0)
#         self.default_params['algorithm'] = self.algorithm_cb
#
#         ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
#         self.isFitted = False
#
#         vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
#         vis_lb_frm.pack(**pad)
#         tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
#                                                                                           **pad)
#         self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col1_cb.grid(row=1, column=0, **pad)
#         self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col2_cb.grid(row=1, column=1, **pad)
#         ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
#                                                                                       **pad)
#         ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)
#
#     def fit(self):
#         self.alg = self.get_alg()
#         self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
#         self.isFitted = True
#         print('Модель обучена')
#
#     def get_params(self):
#         params = {}
#         for param, obj in self.default_params.items():
#             try:
#                 if obj.get() == 'None':
#                     params[param] = None
#                 elif param == 'eps':
#                     params[param] = float(obj.get())
#                 else:
#                     params[param] = int(obj.get())
#             except ValueError:
#                 params[param] = obj.get()
#         return params
#
#     def get_alg(self):
#         params = self.get_params()
#         if isinstance(self.alg, DBSCAN):
#             self.alg = DBSCAN
#         return self.alg(**params)
#
#     def get_plot(self):
#         if self.isFitted:
#             col1 = self.col1_cb.get()
#             col2 = self.col2_cb.get()
#             # centroids = self.alg.cluster_centers_
#             plt.figure(1, figsize=(10, 6))
#             sns.set_theme()
#             sns.set_style('whitegrid')
#             sns.set_context('talk')
#             sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
#             # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
#             #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
#             #                 color='black', marker='s')
#             plt.show()
#
#     def save(self):
#         save_model(self.entry, self.alg)
#         print('Модель сохранена!')


class CMeansFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data, pd_data2, pd_data3):
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.pd_data3 = pd_data3
        self.alg = FCM
        # self.fcm = FCM
        self.lb_frm = ttk.Labelframe(self, text='Конфигурация алгоритма C-means')
        self.lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.clf_conf_frm = tk.Frame(self.lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)

        self.conf_frm = tk.Frame(self.lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=2)
        }
        self.params = {}

        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        ttk.Label(self.clf_conf_frm, text='n_clusters').grid(row=2, column=0, columnspan=2, **pad)
        # , padx=5)
        self.n_clusters_sb = ttk.Spinbox(self.clf_conf_frm, textvariable=self.default_params['n_clusters'], from_=2, to=15, **ent_options)
        self.n_clusters_sb.grid(row=3, column=0, columnspan=2, **pad)
        CreateToolTip(self.n_clusters_sb, text='Количество кластеров')

        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }

        ttk.Button(self.clf_conf_frm, text='Подтвердить', command=self.fit).grid(columnspan=2, padx=15, pady=8)
        self.isFitted = False

        self.accuracy = '...'  # Средняя точность
        self.acc_lbl = ttk.Label(self.clf_conf_frm, text=f'Точность модели: '+str(self.accuracy))
        self.acc_lbl.grid(columnspan=2, **pad)

        self.vis_lb_frm = ttk.LabelFrame(self.clf_conf_frm, text='Визуализация кластеров')
        self.vis_lb_frm.grid(columnspan=2)
        tk.Label(self.vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(self.vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                       **pad)
        self.col3_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col3_cb.grid(row=3, column=0, **pad)

        ttk.Button(self.vis_lb_frm, text='Профили кластеров', command=self.clstr_profile).grid(row=4, column=0, **pad)

        self.col4_cb = ttk.Combobox(self.vis_lb_frm)
        self.col4_cb.grid(row=3, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Характеристики кластера', command=self.clstr_characteristics).grid(row=4, column=1, columnspan=2, **pad)

        tk.Label(self.vis_lb_frm, text='Для сравнения кластеров выберите два кластера:').grid(row=5, column=0, columnspan=2, **pad)

        self.col5_cb = ttk.Combobox(self.vis_lb_frm)
        self.col5_cb.grid(row=6, column=0, **pad)

        self.col6_cb = ttk.Combobox(self.vis_lb_frm)
        self.col6_cb.grid(row=6, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Сравнение кластеров', command=self.clstr_compare).grid(row=7, column=0, columnspan=2, **pad)

        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).grid(**pad)

    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        self.pd_data3.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        # for pd_data3
        for name in self.pd_data3.columns:
            if (type(self.pd_data3[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data3[name] = le.fit_transform(self.pd_data3[name])
        #
        self.alg = self.get_alg()
        # self.fcm = FCM(self.get_alg())
        print(self.alg)
        # self.fcm = FCM(self.get_alg())
        # fcm = FCM.self.get_alg()
        # fcm.fit(self.pd_data)
        # self.pd_data['Clusters'] = self.fcm.soft_predict(self.pd_data)
        # self.pd_data['Clusters'] = self.alg.fit(self.pd_data)
        # self.X = self.pd_data.iloc[:].values
        self.alg.fit(self.pd_data.iloc[:].values)
        # print(labels)
        # self.pd_data['Clusters'] = self.alg.soft_predict(self.pd_data.iloc[:].values)
        # labels = self.alg.soft_predict(self.pd_data.iloc[:].values)
        labels = self.alg.predict(self.pd_data.iloc[:].values)
        # self.pd_data['Clusters'] = labels
        # print(self.pd_data['Clusters'])
        # print(labels)
        self.pd_data['Clusters'] = self.alg.predict(self.pd_data.iloc[:].values)
        self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        unique_clstr = pd.unique(self.pd_data['Clusters'])
        sort_clstr = sorted(unique_clstr)
        print(sort_clstr)
        # accuracy
        if len(sort_clstr) > 1:
            self.accuracy = metrics.silhouette_score(self.pd_data3, self.alg.predict(self.pd_data.iloc[:].values))
        self.acc_lbl.configure(text=f'Точность модели: '+str(self.accuracy))
        self.col4_cb.configure(values=sort_clstr)
        self.col5_cb.configure(values=sort_clstr)
        self.col6_cb.configure(values=sort_clstr)
        return labels


    def get_params(self):
        params = {}
        for param, obj in self.default_params.items():
            try:
                if obj.get() == 'None':
                    params[param] = None
                else:
                    params[param] = int(obj.get())
            except ValueError:
                params[param] = obj.get()
        return params

    def get_alg(self):
        params = self.get_params()
        if isinstance(self.alg, FCM):
            self.alg = FCM
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            # plt.figure(1, figsize=(10, 6))
            # sns.set_theme()
            # sns.set_style('whitegrid')
            # sns.set_context('talk')
            fcm_labels = self.fit()
            # print(fcm_labels)
            # FCM.soft_predict(self.pd_data)
            # fcm_centers = FCM.centers

            f, axes = plt.subplots(1,2, figsize=(11,5))
            # self.pd_data.columns.get_loc(col1), self.pd_data.columns.get_loc(col2)
            # axes[0].scatter(x=col1, y=col2, alpha=.1)
            # axes[1].scatter(x=col1, y=col2, c=fcm_labels, alpha=.1)
            print(fcm_labels)
            print(self.pd_data.iloc[:,1].values)
            print(self.pd_data[col1].iloc[:].values)
            print(self.pd_data[col2].iloc[:].values)
            axes[0].scatter(self.pd_data[col1].iloc[:].values, self.pd_data[col2].iloc[:].values)
            # , alpha=.1)
            axes[1].scatter(self.pd_data[col1].iloc[:].values, self.pd_data[col2].iloc[:].values, c=fcm_labels)
            # , alpha=.1)

            # pd_data.iloc[:].values
            # axes[1].scatter(fcm_centers)
            # sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            plt.show()

            # sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            # sns.scatterplot(x=centroids[:, self.pd_data.columns.get_loc(col1)],
            #                 y=centroids[:, self.pd_data.columns.get_loc(col2)],
            #                 color='black', marker='s')



    def clstr_profile(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        len_clstr = len(unique_clstr)
        sort_clstr = sorted(unique_clstr)
        cols = []
        cols.insert(0, (self.col3_cb.get()))
        df_one = DataFrame()
        res1 = DataFrame()
        for j in range(len(cols)):
            unique_attr_2 = pd.unique(self.pd_data2[cols[j]])
            unique_amount_norm = self.pd_data2[cols[j]].value_counts().sort_index(ascending=False).to_frame()
            df_all = unique_amount_norm.T
            for i in sort_clstr:
                for index, row in self.pd_data2.iterrows():
                    if row['Clusters'] == sort_clstr[i]:
                        df_one = pd.concat([df_one, row.to_frame().T], ignore_index=True)
                res1 = pd.concat(
                    [res1, df_one[cols[j]].value_counts(normalize=True).sort_index(ascending=False).to_frame().T])
                df_one = DataFrame()
            res2 = res1
            res2.index = Series(sort_clstr)
            if (type(unique_attr_2[1]) == str):
                f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                f.suptitle('Атрибут ' + cols[j], fontsize=22)
                df_all.plot.bar(ax=ax[0], stacked='True', alpha=0.5)
                ax[0].set_title('Заполнение по всем кластерам (' + str(len_clstr) + ')', fontsize=12)
                res2.plot.bar(ax=ax[1], stacked='True', alpha=0.5)
                ax[1].set_title('Кластеры', fontsize=12)
                plt.show()
            else:
                if cols[j] == 'Clusters':
                    pass
                else:
                    f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                    f.suptitle('Атрибут ' + cols[j], fontsize=22)
                    sns.boxplot(ax=ax[0], y=self.pd_data2[cols[j]], data=self.pd_data2, notch=False, linewidth=2.5)
                    ax[0].set_title("Диапазон значений (Y) атрибута " + cols[j] + "(X)", fontsize=12)
                    sns.boxplot(ax=ax[1], x='Clusters', y=cols[j], data=self.pd_data2, notch=False)
                    ax[1].set_title('Рапределение значений атрибута ' + cols[j] + ' (Y) в каждом кластере (X)',
                                    fontsize=12)
                    plt.show()
            res1 = DataFrame()


    def clstr_characteristics(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        sort_clstr = sorted(unique_clstr)
        df_comp = DataFrame()
        df_comp_all = DataFrame()
        sort_clstr_2 = []
        sort_clstr_2.insert(0, (int(self.col4_cb.get())))
        for i in sort_clstr_2:
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == sort_clstr[i]:
                    df_comp = pd.concat([df_comp, row.to_frame().T], ignore_index=True)
            del df_comp['Clusters']
            col = df_comp.columns
            for j in col:
                unique_amount = round(df_comp[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_all = (pd.concat([df_comp_all, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
            # Draw plot
            fig, ax = plt.subplots(figsize=(16, 18), facecolor='white', dpi=80)
            ax.vlines(x=df_comp_all.index, ymin=0, ymax=df_comp_all.prob, color='firebrick', alpha=0.7, linewidth=50)
            # Annotate Text
            for k, prob in enumerate(df_comp_all.prob):
                ax.text(k, prob + 0.5, round(prob, 1), horizontalalignment='center')
            # Title, Label, Ticks and Ylim
            ax.set_title('Влияние значений атрибутов на кластер ' + str(sort_clstr[i]), fontdict={'size': 22})
            ax.set(ylabel='Probability', ylim=(0, 100))
            plt.xticks(df_comp_all.index, rotation=60, horizontalalignment='right', fontsize=12)
            plt.show()
            # clear df
            df_comp = DataFrame()
            df_comp_all = DataFrame()

    def clstr_compare(self):
        df_comp_clstrs = DataFrame()
        df_comp_two_clstrs = DataFrame()
        df_comp_two_clstrs_all = DataFrame()
        two_clstr = []
        two_clstr.insert(0, (int(self.col5_cb.get())))
        two_clstr.insert(1, (int(self.col6_cb.get())))
        for i in range(2):
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == two_clstr[i]:
                    df_comp_clstrs = pd.concat([df_comp_clstrs, row.to_frame().T], ignore_index=True)
            del df_comp_clstrs['Clusters']
            col = df_comp_clstrs.columns
            for j in col:
                unique_amount = round(df_comp_clstrs[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = '(' + str(two_clstr[i]) + ') ' + j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_two_clstrs = (
                    pd.concat([df_comp_two_clstrs, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
                df_comp_two_clstrs['Clusters'] = two_clstr[i]
            df_comp_two_clstrs_all = pd.concat([df_comp_two_clstrs_all, df_comp_two_clstrs]).sort_values(by='prob',
                                                                                                         ascending=False)
            # clear df
            df_comp_clstrs = DataFrame()
            df_comp_two_clstrs = DataFrame()
        # Замена значений пеовго кластера на "-"
        df_comp_two_clstrs_all.loc[
            (df_comp_two_clstrs_all.Clusters == two_clstr[0]), ('prob')] = df_comp_two_clstrs_all.prob * (-1)
        del df_comp_two_clstrs_all['Clusters']
        # Draw plot
        df_comp_two_clstrs_all['colors'] = ['red' if x < 0 else 'green' for x in df_comp_two_clstrs_all['prob']]
        plt.figure(figsize=(18, 10), dpi=80)
        plt.hlines(y=df_comp_two_clstrs_all.index, xmin=0, xmax=df_comp_two_clstrs_all.prob,
                   color=df_comp_two_clstrs_all.colors, alpha=0.4, linewidth=5)
        # Decorations
        plt.gca().set(ylabel='$Атрибуты и их значения$', xlabel='$Вероятность$')
        plt.yticks(df_comp_two_clstrs_all.index, fontsize=12)
        plt.title('Сравнение кластеров ' + str(two_clstr[0]) + ' и ' + str(two_clstr[1]), fontdict={'size': 20})
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()

    def save(self):
        pad = {
            'padx': 5,
            'pady': 5
        }
        window = tk.Toplevel(self.master)
        ttk.Label(window, text='Название модели:').grid(row=0, column=0,  **pad)
        name_ent = ttk.Entry(window)
        name_ent.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Label(window, text='Описание модели:').grid(row=1, column=0, **pad)
        desc_text = tk.Text(window, width=30, height=5)
        desc_text.grid(row=1, column=1, **pad)
        ttk.Button(window, text='Сохранить',
                   command=lambda: [save_model(self.entry, self.alg,
                                               # self.accuracy,
                                                                        name=name_ent.get(),
                                                                        desc=desc_text.get("1.0", "end-1c")),
                                    window.destroy()]).grid(row=2, column=0, columnspan=2, **pad)





# class CMeansFrame(tk.Frame):
#     def __init__(self, parent, entry, pd_data):
#         tk.Frame.__init__(self, parent)
#         self.entry = entry
#         self.pd_data = pd_data
#         self.alg = FCM
#         lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма C-Means')
#         lb_frm.pack(fill=tk.BOTH, expand=1)
#
#         self.conf_frm = tk.Frame(lb_frm)
#         self.conf_frm.pack()
#
#         self.default_params = {
#             'n_clusters': tk.IntVar(self.conf_frm, value=2)
#         }
#         opt = {
#             'width': 20,
#             'justify': tk.CENTER
#         }
#         pad = {
#             'padx': 15,
#             'pady': 3
#         }
#
#         tk.Label(self.conf_frm, text='n_clusters').grid(row=0, column=0, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['n_clusters'], **opt)
#         self.n_init_ent.grid(row=1, column=0, **pad)
#
#         ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
#         self.isFitted = False
#
#         vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
#         vis_lb_frm.pack(**pad)
#         tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
#                                                                                           **pad)
#         self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col1_cb.grid(row=1, column=0, **pad)
#         self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col2_cb.grid(row=1, column=1, **pad)
#         ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
#                                                                                       **pad)
#         ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)
#
#     def fit(self):
#         self.alg = self.get_alg()
#         self.pd_data['Clusters'] = self.alg.fit(self.pd_data)
#         self.isFitted = True
#         print('Модель обучена')
#
#     def get_params(self):
#         params = {}
#         for param, obj in self.default_params.items():
#             try:
#                 if obj.get() == 'None':
#                     params[param] = None
#                 else:
#                     params[param] = int(obj.get())
#             except ValueError:
#                 params[param] = obj.get()
#         return params
#
#     def get_alg(self):
#         params = self.get_params()
#         if isinstance(self.alg, FCM):
#             self.alg = FCM
#         return self.alg(**params)
#
#     def get_plot(self):
#         if self.isFitted:
#             col1 = self.col1_cb.get()
#             col2 = self.col2_cb.get()
#             # centroids = self.alg.cluster_centers_
#             plt.figure(1, figsize=(10, 6))
#             sns.set_theme()
#             sns.set_style('whitegrid')
#             sns.set_context('talk')
#             sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
#             # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
#             #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
#             #                 color='black', marker='s')
#             plt.show()
#
#     def save(self):
#         save_model(self.entry, self.alg)
#         print('Модель сохранена!')


class AgglFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data, pd_data2, pd_data3):
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.pd_data3 = pd_data3
        self.alg = AgglomerativeClustering
        self.lb_frm = ttk.Labelframe(self, text='Конфигурация алгоритма иерархической кластеризации')
        self.lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.clf_conf_frm = tk.Frame(self.lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)

        self.conf_frm = tk.Frame(self.lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=2),
            'affinity': tk.StringVar(self.conf_frm, value='euclidean'),
            'linkage': tk.StringVar(self.conf_frm, value='ward')
        }
        self.params = {}

        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        ttk.Label(self.clf_conf_frm, text='n_clusters').grid(row=2, column=0, columnspan=1, **pad)
        self.n_clusters_sb = ttk.Spinbox(self.clf_conf_frm, textvariable=self.default_params['n_clusters'], from_=2, to=15, **ent_options)
        self.n_clusters_sb.grid(row=3, column=0, columnspan=1, **pad)
        CreateToolTip(self.n_clusters_sb, text='Количество кластеров')

        ttk.Label(self.clf_conf_frm, text='affinity').grid(row=2, column=1, columnspan=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.clf_conf_frm, values=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], **ent_options)
        self.algorithm_cb.grid(row=3, column=1, columnspan=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['affinity'] = self.algorithm_cb
        CreateToolTip(self.algorithm_cb, text='Метрика для вычисления расстояния между объектами.\n'
                                              'Если linkage = "ward", то affinity должно принимать\n'
                                              'значение "Euclidean"')

        ttk.Label(self.clf_conf_frm, text='linkage').grid(row=4, columnspan=2, **pad)
        self.algorithm_cb = ttk.Combobox(self.clf_conf_frm, values=['ward', 'complete', 'average', 'single'],
                                         **ent_options)
        self.algorithm_cb.grid(row=5, columnspan=2, **pad)
        self.algorithm_cb.current(0)
        self.default_params['linkage'] = self.algorithm_cb
        CreateToolTip(self.algorithm_cb, text='Метрика для вычисления расстояния между кластерами')

        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }

        ttk.Button(self.clf_conf_frm, text='Подтвердить', command=self.fit).grid(columnspan=2, padx=15, pady=8)
        self.isFitted = False

        self.accuracy = '...'  # Средняя точность
        self.acc_lbl = ttk.Label(self.clf_conf_frm, text=f'Точность модели: '+str(self.accuracy))
        self.acc_lbl.grid(columnspan=2, **pad)

        self.vis_lb_frm = ttk.LabelFrame(self.clf_conf_frm, text='Визуализация кластеров')
        self.vis_lb_frm.grid(columnspan=2)
        tk.Label(self.vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(self.vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                       **pad)
        self.col3_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col3_cb.grid(row=3, column=0, **pad)

        ttk.Button(self.vis_lb_frm, text='Профили кластеров', command=self.clstr_profile).grid(row=4, column=0, **pad)

        self.col4_cb = ttk.Combobox(self.vis_lb_frm)
        self.col4_cb.grid(row=3, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Характеристики кластера', command=self.clstr_characteristics).grid(row=4, column=1, columnspan=2, **pad)

        tk.Label(self.vis_lb_frm, text='Для сравнения кластеров выберите два кластера:').grid(row=5, column=0, columnspan=2, **pad)

        self.col5_cb = ttk.Combobox(self.vis_lb_frm)
        self.col5_cb.grid(row=6, column=0, **pad)

        self.col6_cb = ttk.Combobox(self.vis_lb_frm)
        self.col6_cb.grid(row=6, column=1, **pad)

        ttk.Button(self.vis_lb_frm, text='Сравнение кластеров', command=self.clstr_compare).grid(row=7, column=0, columnspan=2, **pad)

        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).grid(**pad)

    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        self.pd_data3.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        # for pd_data3
        for name in self.pd_data3.columns:
            if (type(self.pd_data3[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data3[name] = le.fit_transform(self.pd_data3[name])
        #
        self.alg = self.get_alg()
        self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
        self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        unique_clstr = pd.unique(self.pd_data['Clusters'])
        sort_clstr = sorted(unique_clstr)
        # print(sort_clstr)
        # accuracy
        if len(sort_clstr) > 1:
            self.accuracy = metrics.silhouette_score(self.pd_data3, self.alg.fit_predict(self.pd_data))
        self.acc_lbl.configure(text=f'Точность модели: '+str(self.accuracy))
        self.col4_cb.configure(values=sort_clstr)
        self.col5_cb.configure(values=sort_clstr)
        self.col6_cb.configure(values=sort_clstr)


    def get_params(self):
        params = {}
        for param, obj in self.default_params.items():
            try:
                if obj.get() == 'None':
                    params[param] = None
                elif param == 'eps':
                    params[param] = float(obj.get())
                else:
                    params[param] = int(obj.get())
            except ValueError:
                params[param] = obj.get()
        return params

    def get_alg(self):
        params = self.get_params()
        if isinstance(self.alg, AgglomerativeClustering):
            self.alg = AgglomerativeClustering
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            plt.show()

    def clstr_profile(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        len_clstr = len(unique_clstr)
        sort_clstr = sorted(unique_clstr)
        cols = []
        cols.insert(0, (self.col3_cb.get()))
        df_one = DataFrame()
        res1 = DataFrame()
        for j in range(len(cols)):
            unique_attr_2 = pd.unique(self.pd_data2[cols[j]])
            unique_amount_norm = self.pd_data2[cols[j]].value_counts().sort_index(ascending=False).to_frame()
            df_all = unique_amount_norm.T
            for i in sort_clstr:
                for index, row in self.pd_data2.iterrows():
                    if row['Clusters'] == sort_clstr[i]:
                        df_one = pd.concat([df_one, row.to_frame().T], ignore_index=True)
                res1 = pd.concat(
                    [res1, df_one[cols[j]].value_counts(normalize=True).sort_index(ascending=False).to_frame().T])
                df_one = DataFrame()
            res2 = res1
            res2.index = Series(sort_clstr)
            if (type(unique_attr_2[1]) == str):
                f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                f.suptitle('Атрибут ' + cols[j], fontsize=22)
                df_all.plot.bar(ax=ax[0], stacked='True', alpha=0.5)
                ax[0].set_title('Заполнение по всем кластерам (' + str(len_clstr) + ')', fontsize=12)
                res2.plot.bar(ax=ax[1], stacked='True', alpha=0.5)
                ax[1].set_title('Кластеры', fontsize=12)
                plt.show()
            else:
                if cols[j] == 'Clusters':
                    pass
                else:
                    f, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=80)
                    f.suptitle('Атрибут ' + cols[j], fontsize=22)
                    sns.boxplot(ax=ax[0], y=self.pd_data2[cols[j]], data=self.pd_data2, notch=False, linewidth=2.5)
                    ax[0].set_title("Диапазон значений (Y) атрибута " + cols[j] + "(X)", fontsize=12)
                    sns.boxplot(ax=ax[1], x='Clusters', y=cols[j], data=self.pd_data2, notch=False)
                    ax[1].set_title('Рапределение значений атрибута ' + cols[j] + ' (Y) в каждом кластере (X)',
                                    fontsize=12)
                    plt.show()
            res1 = DataFrame()


    def clstr_characteristics(self):
        unique_clstr = pd.unique(self.pd_data2['Clusters'])
        sort_clstr = sorted(unique_clstr)
        df_comp = DataFrame()
        df_comp_all = DataFrame()
        sort_clstr_2 = []
        sort_clstr_2.insert(0, (int(self.col4_cb.get())))
        for i in sort_clstr_2:
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == sort_clstr[i]:
                    df_comp = pd.concat([df_comp, row.to_frame().T], ignore_index=True)
            del df_comp['Clusters']
            col = df_comp.columns
            for j in col:
                unique_amount = round(df_comp[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_all = (pd.concat([df_comp_all, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
            # Draw plot
            fig, ax = plt.subplots(figsize=(16, 18), facecolor='white', dpi=80)
            ax.vlines(x=df_comp_all.index, ymin=0, ymax=df_comp_all.prob, color='firebrick', alpha=0.7, linewidth=50)
            # Annotate Text
            for k, prob in enumerate(df_comp_all.prob):
                ax.text(k, prob + 0.5, round(prob, 1), horizontalalignment='center')
            # Title, Label, Ticks and Ylim
            ax.set_title('Влияние значений атрибутов на кластер ' + str(sort_clstr[i]), fontdict={'size': 22})
            ax.set(ylabel='Probability', ylim=(0, 100))
            plt.xticks(df_comp_all.index, rotation=60, horizontalalignment='right', fontsize=12)
            plt.show()
            # clear df
            df_comp = DataFrame()
            df_comp_all = DataFrame()

    def clstr_compare(self):
        df_comp_clstrs = DataFrame()
        df_comp_two_clstrs = DataFrame()
        df_comp_two_clstrs_all = DataFrame()
        two_clstr = []
        two_clstr.insert(0, (int(self.col5_cb.get())))
        two_clstr.insert(1, (int(self.col6_cb.get())))
        for i in range(2):
            for index, row in self.pd_data2.iterrows():
                if row['Clusters'] == two_clstr[i]:
                    df_comp_clstrs = pd.concat([df_comp_clstrs, row.to_frame().T], ignore_index=True)
            del df_comp_clstrs['Clusters']
            col = df_comp_clstrs.columns
            for j in col:
                unique_amount = round(df_comp_clstrs[j].value_counts(normalize=True).to_frame() * 100, 1)
                unique_amount.index = '(' + str(two_clstr[i]) + ') ' + j + ' = ' + unique_amount.index.astype(str)
                unique_amount = unique_amount.rename(columns={j: 'prob'})
                df_comp_two_clstrs = (
                    pd.concat([df_comp_two_clstrs, unique_amount]).sort_values(by='prob', ascending=False)).head(10)
                df_comp_two_clstrs['Clusters'] = two_clstr[i]
            df_comp_two_clstrs_all = pd.concat([df_comp_two_clstrs_all, df_comp_two_clstrs]).sort_values(by='prob',
                                                                                                         ascending=False)
            # clear df
            df_comp_clstrs = DataFrame()
            df_comp_two_clstrs = DataFrame()
        # Замена значений пеовго кластера на "-"
        df_comp_two_clstrs_all.loc[
            (df_comp_two_clstrs_all.Clusters == two_clstr[0]), ('prob')] = df_comp_two_clstrs_all.prob * (-1)
        del df_comp_two_clstrs_all['Clusters']
        # Draw plot
        df_comp_two_clstrs_all['colors'] = ['red' if x < 0 else 'green' for x in df_comp_two_clstrs_all['prob']]
        plt.figure(figsize=(18, 10), dpi=80)
        plt.hlines(y=df_comp_two_clstrs_all.index, xmin=0, xmax=df_comp_two_clstrs_all.prob,
                   color=df_comp_two_clstrs_all.colors, alpha=0.4, linewidth=5)
        # Decorations
        plt.gca().set(ylabel='$Атрибуты и их значения$', xlabel='$Вероятность$')
        plt.yticks(df_comp_two_clstrs_all.index, fontsize=12)
        plt.title('Сравнение кластеров ' + str(two_clstr[0]) + ' и ' + str(two_clstr[1]), fontdict={'size': 20})
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()

    def save(self):
        pad = {
            'padx': 5,
            'pady': 5
        }
        window = tk.Toplevel(self.master)
        ttk.Label(window, text='Название модели:').grid(row=0, column=0,  **pad)
        name_ent = ttk.Entry(window)
        name_ent.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Label(window, text='Описание модели:').grid(row=1, column=0, **pad)
        desc_text = tk.Text(window, width=30, height=5)
        desc_text.grid(row=1, column=1, **pad)
        ttk.Button(window, text='Сохранить',
                   command=lambda: [save_model(self.entry, self.alg,
                                               # self.accuracy,
                                                                        name=name_ent.get(),
                                                                        desc=desc_text.get("1.0", "end-1c")),
                                    window.destroy()]).grid(row=2, column=0, columnspan=2, **pad)

# class AgglFrame(tk.Frame):
#     def __init__(self, parent, entry, pd_data):
#         tk.Frame.__init__(self, parent)
#         self.entry = entry
#         self.pd_data = pd_data
#         self.alg = DBSCAN
#         lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма DBSCAN')
#         lb_frm.pack(fill=tk.BOTH, expand=1)
#
#         # elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров.
#         # Для этого выберите точку перегиба на графике ниже
#         # и занесите это количество в n_clusters."""
#         # tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
#         # ttk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)
#
#         self.conf_frm = tk.Frame(lb_frm)
#         self.conf_frm.pack()
#
#         self.default_params = {
#             # 'n_clusters': tk.IntVar(self.conf_frm, value=8),
#             # 'n_init': tk.StringVar(self.conf_frm, value=10)
#             # 'random_state': tk.StringVar(self.conf_frm, value='None')
#             'eps': tk.StringVar(self.conf_frm, value=0.5),
#             'min_samples': tk.IntVar(self.conf_frm, value=5),
#             'leaf_size': tk.IntVar(self.conf_frm, value=30)
#         }
#         opt = {
#             'width': 20,
#             'justify': tk.CENTER
#         }
#         pad = {
#             'padx': 15,
#             'pady': 3
#         }
#
#         tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['eps'], **opt)
#         self.n_init_ent.grid(row=1, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='min_samples').grid(row=0, column=1, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['min_samples'], **opt)
#         self.n_init_ent.grid(row=1, column=1, **pad)
#
#         tk.Label(self.conf_frm, text='leaf_size').grid(row=2, column=0, **pad)
#         self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['leaf_size'], **opt)
#         self.n_init_ent.grid(row=3, column=0, **pad)
#
#         tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
#         self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], **opt)
#         self.algorithm_cb.grid(row=3, column=1, **pad)
#         self.algorithm_cb.current(0)
#         self.default_params['algorithm'] = self.algorithm_cb
#
#         ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
#         self.isFitted = False
#
#         vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
#         vis_lb_frm.pack(**pad)
#         tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
#                                                                                           **pad)
#         self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col1_cb.grid(row=1, column=0, **pad)
#         self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
#         self.col2_cb.grid(row=1, column=1, **pad)
#         ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
#                                                                                       **pad)
#         ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)
#
#     def fit(self):
#         self.alg = self.get_alg()
#         self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
#         self.isFitted = True
#         print('Модель обучена')
#
#     def get_params(self):
#         params = {}
#         for param, obj in self.default_params.items():
#             try:
#                 if obj.get() == 'None':
#                     params[param] = None
#                 elif param == 'eps':
#                     params[param] = float(obj.get())
#                 else:
#                     params[param] = int(obj.get())
#             except ValueError:
#                 params[param] = obj.get()
#         return params
#
#     def get_alg(self):
#         params = self.get_params()
#         if isinstance(self.alg, DBSCAN):
#             self.alg = DBSCAN
#         return self.alg(**params)
#
#     def get_plot(self):
#         if self.isFitted:
#             col1 = self.col1_cb.get()
#             col2 = self.col2_cb.get()
#             # centroids = self.alg.cluster_centers_
#             plt.figure(1, figsize=(10, 6))
#             sns.set_theme()
#             sns.set_style('whitegrid')
#             sns.set_context('talk')
#             sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
#             # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
#             #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
#             #                 color='black', marker='s')
#             plt.show()
#
#     def save(self):
#         save_model(self.entry, self.alg)
#         print('Модель сохранена!')



class CMeansSoftFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data, pd_data2, pd_data3):
        tk.Frame.__init__(self, parent)
        self.master = parent
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.pd_data3 = pd_data3
        self.alg = FCM
        # self.fcm = FCM
        self.lb_frm = ttk.Labelframe(self, text='Конфигурация алгоритма C-means (soft_predict)')
        self.lb_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=5, pady=5)
        self.clf_conf_frm = tk.Frame(self.lb_frm)  # фрейм для установления объектов по центру
        self.clf_conf_frm.pack()
        self.btn_frm = tk.Frame(self)
        self.btn_frm.pack(side=tk.TOP)

        self.conf_frm = tk.Frame(self.lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=2)
        }
        self.params = {}

        ent_options = {
            'width': 15,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        ttk.Label(self.clf_conf_frm, text='Кол-во кластеров').grid(row=2, column=0, columnspan=2, **pad)
        self.n_clusters_sb = ttk.Spinbox(self.clf_conf_frm, textvariable=self.default_params['n_clusters'], from_=2, to=15, **ent_options)
        self.n_clusters_sb.grid(row=3, column=0, columnspan=2, **pad)
        CreateToolTip(self.n_clusters_sb, text='Количество кластеров')

        btn_pack = {
            'side': tk.LEFT,
            'padx': 10,
            'pady': 5
        }

        ttk.Button(self.clf_conf_frm, text='Подтвердить', command=self.fit).grid(columnspan=2, padx=15, pady=8)
        self.isFitted = False

        self.accuracy = '...'  # Средняя точность
        self.acc_lbl = ttk.Label(self.clf_conf_frm, text=f'Точность модели: '+str(self.accuracy))
        self.acc_lbl.grid(columnspan=2, **pad)

        self.vis_lb_frm = ttk.LabelFrame(self.clf_conf_frm, text='Визуализация кластеров')
        self.vis_lb_frm.grid(columnspan=2)
        tk.Label(self.vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(self.vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                       **pad)
        # self.col3_cb = ttk.Combobox(self.vis_lb_frm, values=list(self.pd_data.columns))
        # self.col3_cb.grid(row=3, column=0, **pad)

        # ttk.Button(self.vis_lb_frm, text='Профили кластеров', command=self.clstr_profile).grid(row=4, column=0, **pad)
        #
        # self.col4_cb = ttk.Combobox(self.vis_lb_frm)
        # self.col4_cb.grid(row=3, column=1, **pad)
        #
        # ttk.Button(self.vis_lb_frm, text='Характеристики кластера', command=self.clstr_characteristics).grid(row=4, column=1, columnspan=2, **pad)
        #
        # tk.Label(self.vis_lb_frm, text='Для сравнения кластеров выберите два кластера:').grid(row=5, column=0, columnspan=2, **pad)

        # self.col5_cb = ttk.Combobox(self.vis_lb_frm)
        # self.col5_cb.grid(row=6, column=0, **pad)
        #
        # self.col6_cb = ttk.Combobox(self.vis_lb_frm)
        # self.col6_cb.grid(row=6, column=1, **pad)

        # ttk.Button(self.vis_lb_frm, text='Сравнение кластеров', command=self.clstr_compare).grid(row=7, column=0, columnspan=2, **pad)

        ttk.Button(self.btn_frm, text='Сохранить модель', command=self.save).grid(**pad)

    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        self.pd_data3.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        # for pd_data3
        # for name in self.pd_data3.columns:
        #     if (type(self.pd_data3[name][1]) == str):
        #         str_col_names.append(name)
        # le = LabelEncoder()
        # for name in str_col_names:
        #     self.pd_data3[name] = le.fit_transform(self.pd_data3[name])
        #
        self.alg = self.get_alg()
        # self.fcm = FCM(self.get_alg())
        print(self.alg)
        # self.fcm = FCM(self.get_alg())
        # fcm = FCM.self.get_alg()
        # fcm.fit(self.pd_data)
        # self.pd_data['Clusters'] = self.fcm.soft_predict(self.pd_data)
        # self.pd_data['Clusters'] = self.alg.fit(self.pd_data)
        # self.X = self.pd_data.iloc[:].values
        self.alg.fit(self.pd_data.iloc[:].values)
        # print(labels)
        # self.pd_data['Clusters'] = self.alg.soft_predict(self.pd_data.iloc[:].values)
        labels = self.alg.soft_predict(self.pd_data.iloc[:].values)
        # labels = self.alg.predict(self.pd_data.iloc[:].values)
        # self.pd_data['Clusters'] = labels
        # print(self.pd_data['Clusters'])
        # print(labels)
        # self.pd_data['Clusters'] = self.alg(self.get_alg()).soft_predict(self.pd_data)
        # self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        # unique_clstr = pd.unique(self.pd_data['Clusters'])
        # sort_clstr = sorted(unique_clstr)
        # print(sort_clstr)
        # accuracy
        # if len(sort_clstr) > 1:
            # self.accuracy = metrics.silhouette_score(self.pd_data3, self.alg.fit_predict(self.pd_data))
        self.acc_lbl.configure(text=f'Точность модели: '+str(self.accuracy))
        # self.col4_cb.configure(values=sort_clstr)
        # self.col5_cb.configure(values=sort_clstr)
        # self.col6_cb.configure(values=sort_clstr)
        return labels


    def get_params(self):
        params = {}
        for param, obj in self.default_params.items():
            try:
                if obj.get() == 'None':
                    params[param] = None
                else:
                    params[param] = int(obj.get())
            except ValueError:
                params[param] = obj.get()
        return params

    def get_alg(self):
        params = self.get_params()
        if isinstance(self.alg, FCM):
            self.alg = FCM
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            # plt.figure(1, figsize=(10, 6))
            # sns.set_theme()
            # sns.set_style('whitegrid')
            # sns.set_context('talk')
            fcm_labels = self.fit()
            # print(fcm_labels)
            # FCM.soft_predict(self.pd_data)
            # fcm_centers = FCM.centers

            f, axes = plt.subplots(1,2, figsize=(11,5))
            # self.pd_data.columns.get_loc(col1), self.pd_data.columns.get_loc(col2)
            # axes[0].scatter(x=col1, y=col2, alpha=.1)
            # axes[1].scatter(x=col1, y=col2, c=fcm_labels, alpha=.1)
            print(fcm_labels)
            print(self.pd_data.iloc[:,1].values)
            print(self.pd_data[col1].iloc[:].values)
            print(self.pd_data[col2].iloc[:].values)
            axes[0].scatter(self.pd_data[col1].iloc[:].values, self.pd_data[col2].iloc[:].values, alpha=.1)
            axes[1].scatter(self.pd_data[col1].iloc[:].values, self.pd_data[col2].iloc[:].values, c=fcm_labels, alpha=.1)
            # pd_data.iloc[:].values
            # axes[1].scatter(fcm_centers)
            # sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            plt.show()

            # sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            # sns.scatterplot(x=centroids[:, self.pd_data.columns.get_loc(col1)],
            #                 y=centroids[:, self.pd_data.columns.get_loc(col2)],
            #                 color='black', marker='s')


    def save(self):
        pad = {
            'padx': 5,
            'pady': 5
        }
        window = tk.Toplevel(self.master)
        ttk.Label(window, text='Название модели:').grid(row=0, column=0,  **pad)
        name_ent = ttk.Entry(window)
        name_ent.grid(row=0, column=1, sticky=tk.W, **pad)
        ttk.Label(window, text='Описание модели:').grid(row=1, column=0, **pad)
        desc_text = tk.Text(window, width=30, height=5)
        desc_text.grid(row=1, column=1, **pad)
        ttk.Button(window, text='Сохранить',
                   command=lambda: [save_model(self.entry, self.alg,
                                               # self.accuracy,
                                                                        name=name_ent.get(),
                                                                        desc=desc_text.get("1.0", "end-1c")),
                                    window.destroy()]).grid(row=2, column=0, columnspan=2, **pad)


