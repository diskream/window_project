import tkinter as tk
from tkinter import ttk
from numpy import arange
from tools.functions import serialize, deserialize, update_entry, save_model
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


class ClusteringView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.geometry('500x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)

        upper_frm = tk.LabelFrame(self, text='Выбор алгоритма')
        upper_frm.pack(fill=tk.X)
        alg = ['K Means', 'DBSCAN']
        self.alg_cb = ttk.Combobox(upper_frm, values=alg)
        self.alg_cb.pack(pady=10)
        self.alg_cb.current(0)
        tk.Button(upper_frm, text='Выбрать', command=self.get_alg).pack(pady=5)
        kmeans_frm = KMeansFrame(self, self.entry, self.pd_data)
        dbscan_frm = DBSCANFrame(self, self.entry, self.pd_data)
        self.algorithms = dict(zip(alg, [kmeans_frm, dbscan_frm]))
        self.current_alg = None
        self.get_alg()

    def get_alg(self):
        alg = self.alg_cb.get()
        self.title(f'Работа с {self.entry.name} с помощью алгоритма "{alg}"')
        if self.current_alg is not None:
            self.current_alg.pack_forget()
        self.algorithms[alg].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.current_alg = self.algorithms[alg]


class KMeansFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.alg = KMeans
        lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма K Means')
        lb_frm.pack(fill=tk.BOTH, expand=1)

        elkan_info = """    При выборе algorithm elkan (метод локтя) для точного построения кластеров 
        необходимо задать их правильное количество. Для этого выберите точку
        перегиба на графике ниже и занесите это количество в n_clusters."""
        tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
        tk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)

        self.conf_frm = tk.Frame(lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=8),
            'n_init': tk.StringVar(self.conf_frm, value=10),
            'random_state': tk.StringVar(self.conf_frm, value='None')
        }
        opt = {
            'width': 20,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }
        tk.Label(self.conf_frm, text='n_clusters').grid(row=0, column=0, **pad)
        self.n_clusters_sb = tk.Spinbox(self.conf_frm, textvariable=self.default_params['n_clusters'],
                                        from_=2, to=15, **opt)
        self.n_clusters_sb.grid(row=1, column=0, **pad)

        tk.Label(self.conf_frm, text='n_init').grid(row=0, column=1, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['n_init'], **opt)
        self.n_init_ent.grid(row=1, column=1, **pad)

        tk.Label(self.conf_frm, text='random_state').grid(row=2, column=0, **pad)
        self.random_state_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['random_state'], **opt)
        self.random_state_ent.grid(row=3, column=0, **pad)

        tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'full', 'elkan'], **opt)
        self.algorithm_cb.grid(row=3, column=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb

        tk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
        self.isFitted = False

        vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
        vis_lb_frm.pack(**pad)
        tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        tk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                      **pad)
        tk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)

    def fit(self):
        self.alg = self.get_alg()
        self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
        self.isFitted = True
        print('Модель обучена')

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
            plt.figure(1, figsize=(15, 8))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            sns.scatterplot(x=centroids[:, self.pd_data.columns.get_loc(col1)],
                            y=centroids[:, self.pd_data.columns.get_loc(col2)],
                            color='black', marker='s')
            plt.show()

    def get_elkan_graph(self):
        inertia = []
        for i in range(1, 11):
            inertia.append(KMeans(n_clusters=i).fit(self.pd_data).inertia_)
        plt.style.use('seaborn-whitegrid')
        plt.figure(1, figsize=(15, 8))
        plt.subplots_adjust(left=0.045, bottom=0.06, right=0.98, top=0.95)
        plt.plot(arange(1, 11), inertia, 'o')
        plt.plot(arange(1, 11), inertia, '-', alpha=0.5)
        plt.xlabel('Количество кластеров', {'fontsize': 15})
        plt.ylabel('Инерция', {'fontsize': 15})
        plt.title('График инерции от количества кластеров.', {'fontsize': 20})
        plt.show()
        # InertiaView(inertia)

    def save(self):
        save_model(self.entry, self.alg)
        print('Модель сохранена!')


class DBSCANFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
