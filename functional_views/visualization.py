import tkinter as tk
from tkinter import ttk

import pandas as pd

from tools.functions import update_entry, deserialize
import random
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


def graph_forget(fn):
    def wrapped(self):
        self.graph.pack_forget()
        self.figure = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(1, 1, 1)
        try:
            fn(self)
        except Exception as _ex:
            sns.lineplot(x=[1, 2, 3, 4], y=[3, 1, 4, 4], ax=self.ax)
            print(_ex)

        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph = self.graph.get_tk_widget()
        self.graph.pack(side=tk.TOP)

    return wrapped


class VisualisationView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.entry = entry
        self.geometry('1280x700')
        self.figsize = (9, 5)
        self.dpi = 100
        self.pad = {
            'padx': 5,
            'pady': 2
        }
        update_entry(self.entry)
        title = self.entry.name if not isinstance(self.entry.name, tuple) else self.entry.name[0]

        self.title('Визуализация таблицы ' + title)
        self.pd_data = deserialize(self.entry.table_file)
        cols = list(self.pd_data.columns)
        # self.graph_frm = tk.Frame(self, bg='blue')
        # self.graph_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.action_lb_frm = tk.LabelFrame(self, text='Работа с графиками')
        self.action_lb_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.action_frm = tk.Frame(self.action_lb_frm)
        self.action_frm.pack()

        self.figure = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(1, 1, 1)
        sns.lineplot(x=[1, 2, 3, 4], y=[3, 1, 4, 4], ax=self.ax)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph = self.graph.get_tk_widget()
        self.graph.pack(side=tk.TOP)

        _pad = {
            'padx': 5,
            'pady': 5
        }

        linear_lb_frm = tk.LabelFrame(self.action_frm, text='LinePlot')
        linear_lb_frm.grid(row=1, column=0, **_pad)
        bar_lb_frm = tk.LabelFrame(self.action_frm, text='BarPlot')
        bar_lb_frm.grid(row=1, column=1, **_pad)
        heat_lb_frm = tk.LabelFrame(self.action_frm, text='HeatMapPlot')
        heat_lb_frm.grid(row=1, column=2, **_pad)
        count_lb_frm = tk.LabelFrame(self.action_frm, text='CountPlot')
        count_lb_frm.grid(row=1, column=3, **_pad)
        reg_lb_frm = tk.LabelFrame(self.action_frm, text='regPlot')
        reg_lb_frm.grid(row=1, column=4, **_pad)
        scatter_lb_frm = tk.LabelFrame(self.action_frm, text='ScatterPlot')
        scatter_lb_frm.grid(row=1, column=5, **_pad)

        # Line Plot
        tk.Label(linear_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_linear_cb = ttk.Combobox(linear_lb_frm, values=cols)
        self.x_linear_cb.grid(row=0, column=1, **_pad)
        tk.Label(linear_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_linear_cb = ttk.Combobox(linear_lb_frm, values=cols)
        self.y_linear_cb.grid(row=1, column=1, **_pad)
        tk.Label(linear_lb_frm, text='Hue :').grid(row=2, column=0, **_pad)
        self.hue_linear_cb = ttk.Combobox(linear_lb_frm, values=cols)
        self.hue_linear_cb.grid(row=2, column=1, **_pad)
        tk.Button(linear_lb_frm, text='Отобразить', command=self.get_line_plot).grid(row=3, column=0, columnspan=2,
                                                                                     **_pad)

        # Bar Plot
        tk.Label(bar_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_bar_cb = ttk.Combobox(bar_lb_frm, values=cols)
        self.x_bar_cb.grid(row=0, column=1, **_pad)
        tk.Label(bar_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_bar_cb = ttk.Combobox(bar_lb_frm, values=cols)
        self.y_bar_cb.grid(row=1, column=1, **_pad)
        tk.Label(bar_lb_frm, text='Hue :').grid(row=2, column=0, **_pad)
        self.hue_bar_cb = ttk.Combobox(bar_lb_frm, values=cols)
        self.hue_bar_cb.grid(row=2, column=1, **_pad)
        tk.Button(bar_lb_frm, text='Отобразить', command=self.get_bar_plot).grid(row=3, column=0, columnspan=2, **_pad)

        # HeatMap
        tk.Label(heat_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_hm_cb = ttk.Combobox(heat_lb_frm, values=cols)
        self.x_hm_cb.grid(row=0, column=1, **_pad)
        tk.Label(heat_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_hm_cb = ttk.Combobox(heat_lb_frm, values=cols)
        self.y_hm_cb.grid(row=1, column=1, **_pad)
        tk.Label(heat_lb_frm, text='Corr :').grid(row=2, column=0, **_pad)
        self.corr_hm_cb = ttk.Combobox(heat_lb_frm, values=[True, False])
        self.corr_hm_cb.grid(row=2, column=1, **_pad)
        self.corr_hm_cb.current(0)
        tk.Button(heat_lb_frm, text='Отобразить', command=self.get_hm_plot).grid(row=3, column=0, columnspan=2, **_pad)

        # Count Plot
        tk.Label(count_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_count_cb = ttk.Combobox(count_lb_frm, values=cols)
        self.x_count_cb.grid(row=0, column=1, **_pad)
        tk.Label(count_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_count_cb = ttk.Combobox(count_lb_frm, values=cols)
        self.y_count_cb.grid(row=1, column=1, **_pad)
        tk.Label(count_lb_frm, text='Hue :').grid(row=2, column=0, **_pad)
        self.hue_count_cb = ttk.Combobox(count_lb_frm, values=cols)
        self.hue_count_cb.grid(row=2, column=1, **_pad)
        tk.Button(count_lb_frm, text='Отобразить', command=self.get_count_plot).grid(row=3, column=0, columnspan=2,
                                                                                     **_pad)

        # reg Plot
        tk.Label(reg_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_reg_cb = ttk.Combobox(reg_lb_frm, values=cols)
        self.x_reg_cb.grid(row=0, column=1, **_pad)
        tk.Label(reg_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_reg_cb = ttk.Combobox(reg_lb_frm, values=cols)
        self.y_reg_cb.grid(row=1, column=1, **_pad)
        tk.Button(reg_lb_frm, text='Отобразить', command=self.get_reg_plot).grid(row=3, column=0, columnspan=2,
                                                                                 **_pad)

        # Scatter Plot
        tk.Label(scatter_lb_frm, text='X :').grid(row=0, column=0, **_pad)
        self.x_scatter_cb = ttk.Combobox(scatter_lb_frm, values=cols)
        self.x_scatter_cb.grid(row=0, column=1, **_pad)
        tk.Label(scatter_lb_frm, text='Y :').grid(row=1, column=0, **_pad)
        self.y_scatter_cb = ttk.Combobox(scatter_lb_frm, values=cols)
        self.y_scatter_cb.grid(row=1, column=1, **_pad)
        tk.Label(scatter_lb_frm, text='Hue :').grid(row=2, column=0, **_pad)
        self.hue_scatter_cb = ttk.Combobox(scatter_lb_frm, values=cols)
        self.hue_scatter_cb.grid(row=2, column=1, **_pad)
        tk.Button(scatter_lb_frm, text='Отобразить', command=self.get_scatter_plot).grid(row=3, column=0, columnspan=2,
                                                                                         **_pad)

    @graph_forget
    def get_line_plot(self):
        x = self.x_linear_cb.get()
        y = self.y_linear_cb.get()
        hue = self.hue_linear_cb.get()
        if hue == '':
            hue = None
        sns.lineplot(x=x, y=y, hue=hue, data=self.pd_data, ax=self.ax)

    @graph_forget
    def get_bar_plot(self):
        x = self.x_bar_cb.get()
        y = self.y_bar_cb.get()
        hue = self.hue_bar_cb.get()
        if hue == '':
            hue = None
        sns.barplot(x=x, y=y, hue=hue, data=self.pd_data, ax=self.ax)
        if len(self.pd_data[x]) > 15 or len(self.pd_data[y]) > 15:
            self.ax.tick_params(rotation=90)

    @graph_forget
    def get_hm_plot(self):
        x = self.x_hm_cb.get()
        y = self.y_hm_cb.get()
        is_corr = bool(self.corr_hm_cb.get())
        if is_corr:
            corr = self.pd_data.corr()
            np.fill_diagonal(corr.values, np.nan)
            sns.heatmap(corr, annot=True, ax=self.ax)
            self.ax.tick_params(rotation=45)
        else:
            data = pd.DataFrame()
            data[x] = self.pd_data[x]
            data[y] = self.pd_data[y]
            sns.heatmap(data, ax=self.ax)

    @graph_forget
    def get_count_plot(self):
        x = self.x_count_cb.get()
        y = self.y_count_cb.get()
        hue = self.hue_count_cb.get()
        if hue == '':
            hue = None
        if x == '':
            x = None
        y = None if x is not None else y
        sns.countplot(x=x, y=y, hue=hue, data=self.pd_data, ax=self.ax)
        if len(self.pd_data[x]) > 15 or len(self.pd_data[y]) > 15:
            self.ax.tick_params(rotation=90)

    @graph_forget
    def get_reg_plot(self):
        x = self.x_reg_cb.get()
        y = self.y_reg_cb.get()
        sns.regplot(x=x, y=y, fit_reg=True, data=self.pd_data, ax=self.ax)

    @graph_forget
    def get_scatter_plot(self):
        x = self.x_scatter_cb.get()
        y = self.y_scatter_cb.get()
        hue = self.hue_scatter_cb.get()
        if hue == '':
            hue = None
        sns.scatterplot(x=x, y=y, hue=hue, data=self.pd_data, ax=self.ax)

