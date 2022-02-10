import tkinter as tk
from tkinter import ttk
from functions import show_table


class TableView:
    def __init__(self, master, geometry, data, table=None, *args, **kwargs):
        self.master = master
        self.master.geometry(geometry)
        if table is not None:
            self.table = table[0]
        else:
            self.table = table
        self.data = data
        tk.Label(self.master, text="{} table".format(self.data)).pack(side=tk.TOP)
        self.tv = ttk.Treeview(self.master, show='headings')
        show_table(self)

        # self.configuration_tv()
        # self.cols = self.get_cols()
