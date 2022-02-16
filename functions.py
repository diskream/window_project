import sqlite3
import pickle
import tkinter as tk
from tkinter import ttk
from models import *


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
    """
    Отображает таблицу
    :param method: используется для ссылки на self
    :param sb_place: место расположения скроллбара
    :return:
    """
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


def upload_data(table, **data):
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        # Тут костыль: if работает по сути только 1 раз - при обработке
        # новой таблицы из Tasks. variant_id в таком случае всегда устанавливается
        # 1. Сделать проверку на уже существующие измененные таблицы...
        if 'variant_id' not in data:
            data['variant_id'] = 1
        if data['variant_id'] is None:
            variant = cur.execute(f'SELECT MAX(variant_id) FROM {table} '
                                  f'WHERE task_id = {data["task_id"]}').fetchone()[0]
            data['variant_id'] = variant + 1
        data['name'] += f'_{data["variant_id"]}'
        placeholders = ', '.join('?' for _ in data.values())
        cols = ', '.join(data.keys())
        _sql = f'INSERT INTO {table} ({cols}) VALUES ({placeholders})'
        cur.execute(_sql, tuple(data.values()))

    finally:
        conn.commit()
        conn.close()
        return data


def get_db():
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    classes = {'Tasks': Task, 'Task_variant': Variant, 'Models': Model}
    db = {}
    try:
        table_list = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()[1:]
        for table in table_list:
            temp_columns = cur.execute(f'PRAGMA table_info({table[0]})').fetchall()
            columns = []
            for col in temp_columns:
                columns.append(col[1])
            try:
                columns.remove('table_file')
            except ValueError:
                columns.remove('bin_file')
            query = cur.execute(f'SELECT {", ".join(columns)} FROM {table[0]}').fetchall()
            temp_lst = []
            for que in query:
                temp_lst.append(classes[table[0]](*que))
            db[table[0]] = temp_lst
        return db
    finally:
        conn.close()


def update_entry(entry):
    if entry.table_file is not None:
        return
    else:
        sql = f'SELECT table_file FROM {entry.table} WHERE task_id = {entry.task_id} '
        if hasattr(entry, 'variant_id'):
            sql += f'AND variant_id = {entry.variant_id} '
        print(sql)
        with sqlite3.connect('main.sqlite3') as conn:
            entry.table_file = conn.cursor().execute(sql).fetchall()[0][0]


def get_entry(table, **data):
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        if isinstance(table, tuple):
            table = table[0]
        query = cur.execute(f'SELECT * FROM {table} WHERE task_id = {data["task_id"]}'
                            f' AND variant_id = {data["variant_id"]}').fetchall()[0]
        return Variant(*query)
    finally:
        conn.close()
