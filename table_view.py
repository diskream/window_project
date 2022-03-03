import sqlite3
import tkinter as tk
from tkinter import ttk
from functions import show_table, update_entry, deserialize


class TableView(tk.Tk):
    def __init__(self, db: bool, db_table=None, entry=None):
        tk.Tk.__init__(self)
        self.geometry('1000x600')
        self.table_frm = tk.LabelFrame(self)
        self.table_frm.pack(fill=tk.BOTH, expand=True)
        self.tv1_frm = tk.Frame(self.table_frm)
        self.tv1_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.tv2_frm = tk.Frame(self.table_frm)
        self.tv2_frm.pack(side=tk.TOP, fill=tk.X)
        self.tv = ttk.Treeview(self.tv1_frm, show='headings')
        if not db:
            self.entry = entry
            self.table = entry.table
            if isinstance(self.entry.name, tuple):
                self.entry.name = self.entry.name[0]
            self.data = self.entry.name
            update_entry(self.entry)
            self.pd_data = deserialize(self.entry.table_file)
            show_table(self)
        else:
            self.db_table = db_table
            self.show_db_table()

    def get_db_data(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            data = cur.execute(f'SELECT * FROM {self.db_table}').fetchall()
            cur.execute(f'SELECT * FROM {self.db_table} WHERE 1=0')
            columns = [d[0] for d in cur.description]
            return data, columns
        finally:
            conn.close()

    def show_db_table(self):
        data, columns = self.get_db_data()
        col_width, i = 40, 0
        self.tv['columns'] = columns
        for column in self.tv['columns']:
            self.tv.heading(column, text=column)
            self.tv.column(column, width=col_width)
        for row in data:
            if i % 2 == 0:
                self.tv.insert('', tk.END, values=row, tag='even')
            else:
                self.tv.insert('', tk.END, values=row, tag='odd')
            i += 1
        self.tv.tag_configure('even', background='#E8E8E8')
        self.tv.tag_configure('odd', background='#DFDFDF')
        ysb = ttk.Scrollbar(self.tv1_frm, orient=tk.VERTICAL, command=self.tv.yview)
        xsb = ttk.Scrollbar(self.tv2_frm, orient=tk.HORIZONTAL, command=self.tv.xview)
        self.tv.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)


