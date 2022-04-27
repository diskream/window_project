import tkinter as tk
from tkinter import ttk
from numpy import arange
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

from tools.functions import deserialize, update_entry, save_model
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from fcmeans import FCM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class ClusteringView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.geometry('500x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        self.pd_data2 = deserialize(self.entry.table_file)

        upper_frm = tk.LabelFrame(self, text='Выбор алгоритма')
        upper_frm.pack(fill=tk.X)
        alg = ['K Means', 'DBSCAN', 'C means', 'Agglomerative']
        self.alg_cb = ttk.Combobox(upper_frm, values=alg)
        self.alg_cb.pack(pady=10)
        self.alg_cb.current(0)
        ttk.Button(upper_frm, text='Выбрать', command=self.get_alg).pack(pady=5)
        kmeans_frm = KMeansFrame(self, self.entry, self.pd_data, self.pd_data2)
        dbscan_frm = DBSCANFrame(self, self.entry, self.pd_data)
        cmeans_frm = CMeansFrame(self, self.entry, self.pd_data)
        aggl_frm = AgglFrame(self, self.entry, self.pd_data)
        self.algorithms = dict(zip(alg, [kmeans_frm, dbscan_frm, cmeans_frm, aggl_frm]))
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
    def __init__(self, parent, entry, pd_data, pd_data2):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.pd_data2 = pd_data2
        self.alg = KMeans
        #self.df_clstr = df_clstr
        lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма K Means')
        lb_frm.pack(fill=tk.BOTH, expand=1)

        elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров. 
        Для этого выберите точку перегиба на графике ниже 
        и занесите это количество в n_clusters."""
        tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
        ttk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)

        self.conf_frm = tk.Frame(lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=8),
            'n_init': tk.StringVar(self.conf_frm, value=10)
            # 'random_state': tk.StringVar(self.conf_frm, value='None')
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

        # tk.Label(self.conf_frm, text='random_state').grid(row=2, column=0, **pad)
        # self.random_state_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['random_state'], **opt)
        # self.random_state_ent.grid(row=3, column=0, **pad)

        tk.Label(self.conf_frm, text='init').grid(row=2, column=0, **pad)
        self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['k-means++', 'random'], **opt)
        self.algorithm_cb.grid(row=3, column=0, **pad)
        self.algorithm_cb.current(0)
        self.default_params['init'] = self.algorithm_cb

        tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'full', 'elkan'], **opt)
        self.algorithm_cb.grid(row=3, column=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb

        ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
        self.isFitted = False

        vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
        vis_lb_frm.pack(**pad)
        tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                      **pad)

        # df_clstr = pd_data
        # self.df_clstr['clstr'] = self.alg.fit_predict(self.df_clstr)
        # unique_clstr = pd.unique(pd_data['Clusters'])
        # sort_clstr = sorted(unique_clstr)


        self.col3_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col3_cb.grid(row=3, column=0, **pad)

        ttk.Button(vis_lb_frm, text = 'Профили кластеров', command = self.clstr_profile).grid(row=4, column=0,
                                                                                      **pad)

        # self.col4_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
        self.col4_cb = ttk.Combobox(vis_lb_frm)

        # self.col4_cb = ttk.Combobox(vis_lb_frm)
        self.col4_cb.grid(row=3, column=1, **pad)

        ttk.Button(vis_lb_frm, text = 'Характеристики кластера', command = self.clstr_characteristics).grid(row=4, column=1, columnspan=2,
                                                                                      **pad)
        # self.col5_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
        self.col5_cb = ttk.Combobox(vis_lb_frm)
        self.col5_cb.grid(row=5, column=0, **pad)

        # self.col6_cb = ttk.Combobox(vis_lb_frm, values=self.fit())
        self.col6_cb = ttk.Combobox(vis_lb_frm)
        self.col6_cb.grid(row=5, column=1, **pad)

        ttk.Button(vis_lb_frm, text='Сравнение кластеров', command = self.clstr_compare).grid(row=6, column=0, columnspan=2,
                                                                                      **pad)
        #
        # ttk.Button(vis_lb_frm, text='Обновить номера кластеров', command=self.col5_cb).grid(row=7, column=0,
        #                                                                                     columnspan=2,
        #                                                                                     **pad)

        ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)


    def fit(self):
        self.pd_data.dropna(axis=0, inplace=True, how='any')
        str_col_names = []
        for name in self.pd_data.columns:
            if (type(self.pd_data[name][1]) == str):
                str_col_names.append(name)
        le = LabelEncoder()
        for name in str_col_names:
            self.pd_data[name] = le.fit_transform(self.pd_data[name])
        self.alg = self.get_alg()
        self.pd_data['Clusters'] = self.alg.fit_predict(self.pd_data)
        self.pd_data2['Clusters'] = self.pd_data['Clusters']
        self.isFitted = True
        print('Модель обучена')
        unique_clstr = pd.unique(self.pd_data['Clusters'])
        sort_clstr = sorted(unique_clstr)
        print(sort_clstr)
        self.col4_cb.configure(values=sort_clstr)
        self.col5_cb.configure(values=sort_clstr)
        self.col6_cb.configure(values=sort_clstr)
        # return sort_clstr

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
        cols.insert(0,(self.col3_cb.get()))
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
        save_model(self.entry, self.alg)
        print('Модель сохранена!')


class DBSCANFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.alg = DBSCAN
        lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма DBSCAN')
        lb_frm.pack(fill=tk.BOTH, expand=1)

        # elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров.
        # Для этого выберите точку перегиба на графике ниже
        # и занесите это количество в n_clusters."""
        # tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
        # ttk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)

        self.conf_frm = tk.Frame(lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            # 'n_clusters': tk.IntVar(self.conf_frm, value=8),
            # 'n_init': tk.StringVar(self.conf_frm, value=10)
            # 'random_state': tk.StringVar(self.conf_frm, value='None')
            'eps': tk.StringVar(self.conf_frm, value=0.5),
            'min_samples': tk.IntVar(self.conf_frm, value=5),
            'leaf_size': tk.IntVar(self.conf_frm, value=30)
        }
        opt = {
            'width': 20,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }
        # tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
        # self.n_clusters_sb = tk.Spinbox(self.conf_frm, textvariable=self.default_params['n_clusters'],
        #                                 from_=2, to=15, **opt)
        # self.n_clusters_sb.grid(row=1, column=0, **pad)

        tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['eps'], **opt)
        self.n_init_ent.grid(row=1, column=0, **pad)

        tk.Label(self.conf_frm, text='min_samples').grid(row=0, column=1, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['min_samples'], **opt)
        self.n_init_ent.grid(row=1, column=1, **pad)

        tk.Label(self.conf_frm, text='leaf_size').grid(row=2, column=0, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['leaf_size'], **opt)
        self.n_init_ent.grid(row=3, column=0, **pad)

        tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], **opt)
        self.algorithm_cb.grid(row=3, column=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb

        ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
        self.isFitted = False

        vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
        vis_lb_frm.pack(**pad)
        tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                      **pad)
        ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)

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
            # centroids = self.alg.cluster_centers_
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
            #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
            #                 color='black', marker='s')
            plt.show()

    def save(self):
        save_model(self.entry, self.alg)
        print('Модель сохранена!')


class CMeansFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.alg = FCM
        lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма C-Means')
        lb_frm.pack(fill=tk.BOTH, expand=1)

        self.conf_frm = tk.Frame(lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            'n_clusters': tk.IntVar(self.conf_frm, value=2)
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
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['n_clusters'], **opt)
        self.n_init_ent.grid(row=1, column=0, **pad)

        ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
        self.isFitted = False

        vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
        vis_lb_frm.pack(**pad)
        tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                      **pad)
        ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)

    def fit(self):
        self.alg = self.get_alg()
        self.pd_data['Clusters'] = self.alg.fit(self.pd_data)
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
        if isinstance(self.alg, FCM):
            self.alg = FCM
        return self.alg(**params)

    def get_plot(self):
        if self.isFitted:
            col1 = self.col1_cb.get()
            col2 = self.col2_cb.get()
            # centroids = self.alg.cluster_centers_
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
            #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
            #                 color='black', marker='s')
            plt.show()

    def save(self):
        save_model(self.entry, self.alg)
        print('Модель сохранена!')



class AgglFrame(tk.Frame):
    def __init__(self, parent, entry, pd_data):
        tk.Frame.__init__(self, parent)
        self.entry = entry
        self.pd_data = pd_data
        self.alg = DBSCAN
        lb_frm = tk.LabelFrame(self, text='Конфигурация алгоритма DBSCAN')
        lb_frm.pack(fill=tk.BOTH, expand=1)

        # elkan_info = """        Для работы алгоритма K means необходимо задать количество кластеров.
        # Для этого выберите точку перегиба на графике ниже
        # и занесите это количество в n_clusters."""
        # tk.Label(lb_frm, text=elkan_info, justify=tk.LEFT).pack(pady=2)
        # ttk.Button(lb_frm, text='Выбрать количество кластеров', command=self.get_elkan_graph).pack(pady=5)

        self.conf_frm = tk.Frame(lb_frm)
        self.conf_frm.pack()

        self.default_params = {
            # 'n_clusters': tk.IntVar(self.conf_frm, value=8),
            # 'n_init': tk.StringVar(self.conf_frm, value=10)
            # 'random_state': tk.StringVar(self.conf_frm, value='None')
            'eps': tk.StringVar(self.conf_frm, value=0.5),
            'min_samples': tk.IntVar(self.conf_frm, value=5),
            'leaf_size': tk.IntVar(self.conf_frm, value=30)
        }
        opt = {
            'width': 20,
            'justify': tk.CENTER
        }
        pad = {
            'padx': 15,
            'pady': 3
        }

        tk.Label(self.conf_frm, text='eps').grid(row=0, column=0, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['eps'], **opt)
        self.n_init_ent.grid(row=1, column=0, **pad)

        tk.Label(self.conf_frm, text='min_samples').grid(row=0, column=1, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['min_samples'], **opt)
        self.n_init_ent.grid(row=1, column=1, **pad)

        tk.Label(self.conf_frm, text='leaf_size').grid(row=2, column=0, **pad)
        self.n_init_ent = tk.Entry(self.conf_frm, textvariable=self.default_params['leaf_size'], **opt)
        self.n_init_ent.grid(row=3, column=0, **pad)

        tk.Label(self.conf_frm, text='algorithm').grid(row=2, column=1, **pad)
        self.algorithm_cb = ttk.Combobox(self.conf_frm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], **opt)
        self.algorithm_cb.grid(row=3, column=1, **pad)
        self.algorithm_cb.current(0)
        self.default_params['algorithm'] = self.algorithm_cb

        ttk.Button(lb_frm, text='Подтвердить', command=self.fit).pack(**pad)
        self.isFitted = False

        vis_lb_frm = tk.LabelFrame(lb_frm, text='Визуализация кластеров')
        vis_lb_frm.pack(**pad)
        tk.Label(vis_lb_frm, text='Для отображения кластеров выберите две колонки:').grid(row=0, column=0, columnspan=2,
                                                                                          **pad)
        self.col1_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col1_cb.grid(row=1, column=0, **pad)
        self.col2_cb = ttk.Combobox(vis_lb_frm, values=list(self.pd_data.columns))
        self.col2_cb.grid(row=1, column=1, **pad)
        ttk.Button(vis_lb_frm, text='Отобразить кластеры', command=self.get_plot).grid(row=2, column=0, columnspan=2,
                                                                                      **pad)
        ttk.Button(lb_frm, text='Сохранить модель', command=self.save).pack(**pad)

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
            # centroids = self.alg.cluster_centers_
            plt.figure(1, figsize=(10, 6))
            sns.set_theme()
            sns.set_style('whitegrid')
            sns.set_context('talk')
            sns.scatterplot(x=col1, y=col2, hue='Clusters', data=self.pd_data, palette='bright')
            # sns.scatterplot(x=col1[:, self.pd_data.columns.get_loc(col1)],
            #                 y=col2[:, self.pd_data.columns.get_loc(col2)],
            #                 color='black', marker='s')
            plt.show()

    def save(self):
        save_model(self.entry, self.alg)
        print('Модель сохранена!')

