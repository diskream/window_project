import tkinter as tk
from tools.functions import update_entry, deserialize
import random
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class VisualisationView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.entry = entry
        self.geometry('1120x800')
        update_entry(self.entry)
        title = self.entry.name if not isinstance(self.entry.name, tuple) else self.entry.name[0]

        self.title('Визуализация таблицы ' + title)
        self.pd_data = deserialize(self.entry.table_file)
        # self.graph_frm = tk.Frame(self, bg='blue')
        # self.graph_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.action_lb_frm = tk.LabelFrame(self, text='Работа с графиками')
        self.action_lb_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        self.figure = Figure(figsize=(6,4), dpi=100)
        self.ax = self.figure.add_subplot(1, 1, 1)
        sns.distplot(x=random.randrange(0, 10), ax=self.ax)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph = self.graph.get_tk_widget()
        self.graph.pack(side=tk.TOP)
        tk.Button(self.action_lb_frm, text='asd', command=self.btn).pack()

    def btn(self):
        self.graph.pack_forget()
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(1, 1, 1)
        sns.distplot(x=random.randrange(0, 10), ax=self.ax)
        self.graph.pack()

