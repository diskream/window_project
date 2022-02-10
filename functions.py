import sqlite3
import pickle
import tkinter as tk
import pandas as pd
from tkinter import ttk


def treeview_sort_column(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    l.sort(key=lambda t: float(t[0]), reverse=reverse)

    # rearrange items in sorted positions
    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

    # reverse sort next time
    tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))


def get_data(method):
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        if method.table is not None:
            data = deserialize(cur.execute(f'SELECT table_file FROM {method.table} '
                                           f'WHERE name = "{method.data}"').fetchall()[0][0])
            return data
        else:
            data = cur.execute(f'SELECT * FROM {method.data}').fetchall()
            cur.execute('SELECT * FROM {} WHERE 1=0'.format(method.data))
            columns = [d[0] for d in method.cur.description]
            return data, columns
    finally:
        conn.close()


def show_table(method, sb_place=None):
    '''
    Отображает таблицу
    :param method: используется для ссылки на self
    :return:
    '''
    column_width = 100
    i = 0
    if method.table is not None:
        data = get_data(method)
        method.tv['columns'] = list(data.columns)
        for column in method.tv['columns']:
            method.tv.heading(column, text=column, command=lambda: treeview_sort_column(method.tv, column, False))
            method.tv.column(column, width=column_width)
        rows = data.to_numpy().tolist()
        for row in rows:
            if i % 2 == 0:
                method.tv.insert('', tk.END, values=row, tag='even')
            else:
                method.tv.insert('', tk.END, values=row, tag='odd')
            i += 1
    else:
        data, columns = get_data(method)
        method.tv['columns'] = columns
        for column in method.tv['columns']:
            method.tv.heading(column, text=column, command=lambda: treeview_sort_column(method.tv, column, False))
            method.tv.column(column, width=column_width)
        for row in data:
            if i % 2 == 0:
                method.tv.insert('', tk.END, values=row, tag='even')
            else:
                method.tv.insert('', tk.END, values=row, tag='odd')
            i += 1

    method.tv.tag_configure('even', background='#E8E8E8')
    method.tv.tag_configure('odd', background='#DFDFDF')
    if sb_place is None:
        ysb = ttk.Scrollbar(method.tv1_frm, orient=tk.VERTICAL, command=method.tv.yview)
        xsb = ttk.Scrollbar(method.tv2_frm, orient=tk.HORIZONTAL, command=method.tv.xview)
    method.tv.configure(yscroll=ysb.set, xscroll=xsb.set)
    ysb.pack(side=tk.RIGHT, fill=tk.Y)
    method.tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    xsb.pack(side=tk.BOTTOM, fill=tk.X)

    # def get_cols(self):
    #     # getting columns from table
    #     self.cur.execute('SELECT * FROM {} WHERE 1=0'.format(self.table))
    #     return [d[0] for d in self.cur.description]
    #
    # def configuration_tv(self):
    #     # configuring headings
    #     n, m = 1, 0
    #     for col in self.cols:
    #         self.tv.heading('#{}'.format(n), text=col)
    #         self.tv.column(m, width=80)
    #         n += 1
    #         m += 1


def serialize(file):
    return pickle.dumps(file)


def deserialize(file):
    return pickle.loads(file)