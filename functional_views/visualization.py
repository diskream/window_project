import tkinter as tk
from tkinter import ttk
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
        self.figsize = (7, 4)
        self.dpi = 100
        self.pad = {
            'padx': 5,
            'pady': 2
        }
        update_entry(self.entry)
        title = self.entry.name if not isinstance(self.entry.name, tuple) else self.entry.name[0]

        self.title('Визуализация таблицы ' + title)
        self.pd_data = deserialize(self.entry.table_file)
        # self.graph_frm = tk.Frame(self, bg='blue')
        # self.graph_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.action_lb_frm = tk.LabelFrame(self, text='Работа с графиками')
        self.action_lb_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        self.figure = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(1, 1, 1)
        sns.distplot(x=random.randrange(0, 10), ax=self.ax)
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph = self.graph.get_tk_widget()
        self.graph.pack(side=tk.TOP)

        # Создание словаря с графиками
        graph_lst = ['Linear', 'Bar', 'Heatmap', 'Count', 'Relplot', 'Scatter']
        plots = [LinePlot, BarPlot, HeatMap, CountPlot, RelPlot, ScatterPlot]
        self.graph_dict = dict(zip(graph_lst, plots))
        tk.Label(self.action_lb_frm, text='Выберите тип графика:').pack(side=tk.TOP, **self.pad)
        self.graph_cb = ttk.Combobox(self.action_lb_frm, values=graph_lst)
        self.graph_cb.pack(side=tk.TOP, **self.pad)
        self.graph_cb.current(0)
        tk.Button(self.action_lb_frm, text='Выбрать', command=self.choose_plot).pack(side=tk.TOP, **self.pad)
        self.current_plt = None
        self.choose_plot()

    def choose_plot(self):
        new_frame = self.graph_dict[self.graph_cb.get()](self.action_lb_frm)
        if self.current_plt is not None:
            self.current_plt.destroy()
        self.current_plt = new_frame
        self.current_plt.pack()


class LinePlot(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='LinePlot')
        tk.Label(self, text='Line plot text').pack(side=tk.TOP)


class BarPlot(tk.LabelFrame):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='BarPlot')
        tk.Label(self, text='Bar plot text').pack(side=tk.TOP)


class HeatMap(LinePlot):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='HeatMap')
        tk.Label(self, text='Heatmap plot text').pack(side=tk.TOP)


class CountPlot(LinePlot):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='CountPlot')
        tk.Label(self, text='Count plot text').pack(side=tk.TOP)


class RelPlot(LinePlot):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='RelPlot')
        tk.Label(self, text='Rel plot text').pack(side=tk.TOP)


class ScatterPlot(LinePlot):
    def __init__(self, parent):
        tk.LabelFrame.__init__(self, parent)
        self.configure(text='ScatterPlot')
        tk.Label(self, text='Scatter plot text').pack(side=tk.TOP)
