import tkinter as tk
import pandas as pd
from tkinter import ttk
import sqlite3
from table_view import TableView
from tkinter.filedialog import askopenfile
from classification import MLView
from data_views import DataView
from functions import serialize


class TopFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.master = master

        # initializing ListBox and putting values into it
        # self.tables_list = self.get_tables_list()
        # self.list_box = tk.Listbox(self)
        # self.tables_list = sorted([table[0] for table in self.tables_list])
        # for index, table in enumerate(self.tables_list):
        #     self.list_box.insert(index, table)
        # self.list_box.pack(anchor=tk.W)

        tk.Label(self, text='Выберите данные из вложенного списка:').pack(side=tk.TOP, ipady=10)
        # Creating hierarchical treeview
        self.tv_hier = ttk.Treeview(self, height=13, show='tree')
        self.tv_hier.pack(side=tk.TOP)
        self.insert_tv()

        # Buttons for the new windows
        self.ml_window_btn = tk.Button(self, text='Открыть окно классификации', command=self.open_ml)
        self.ml_window_btn.pack(side=tk.BOTTOM)
        self.table_open_btn = tk.Button(self, text='Открыть окно обзора данных', command=self.open_table)
        self.table_open_btn.pack(side=tk.BOTTOM)
        self.data_btn = tk.Button(self, text='Открыть окно редактирования данных', command=self.open_data)
        self.data_btn.pack(side=tk.BOTTOM)

        self.warn_var = tk.StringVar()
        self.warn_lbl = tk.Label(self, textvariable=self.warn_var)
        self.warn_lbl.pack(side=tk.BOTTOM)

    def insert_tv(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            n = 0
            tables_list = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()
            if len(tables_list) >= 4:
                tables_list = tables_list[1:]
            for table in tables_list:
                self.tv_hier.insert('', '0', 'item{}'.format(n), text=table, tags='table')
                names = cur.execute('SELECT name FROM {}'.format(table[0])).fetchall()
                for name in names:
                    self.tv_hier.insert(f'item{n}', tk.END, text=name, tags=f'{table[0]}')
                n += 1
        finally:
            cur.close()
            conn.close()

    def open_table(self):
        print(self.tv_hier.item(self.tv_hier.selection()))
        table = self.tv_hier.item(self.tv_hier.selection())
        if table['text'] == '':
            self.warn_var.set('Данные не выбраны. Пожалуйста, выберетите данные из списка выше.')
            return
        elif table['tags'][0] == 'table':
            TableView(tk.Toplevel(self), geometry='1000x700', data=table['text'])
        else:
            TableView(tk.Toplevel(self), geometry='1000x700', data=table['text'], table=table['tags'])

    def open_ml(self):
        table = self.tv_hier.item(self.tv_hier.selection())
        if self.table_check(table):
            MLView(data=table['text'], table=table['tags'])

    def open_data(self):
        table = self.tv_hier.item(self.tv_hier.selection())
        if self.table_check(table):
            DataView(self.master.HEIGHT, data=table['text'], table=table['tags'])

    def table_check(self, table):
        if table['text'] == '':
            self.warn_var.set('Данные не выбраны. Пожалуйста, выберетите данные из списка выше.')
            return False
        elif table['tags'][0] == 'table':
            self.warn_var.set('Пожалуйста, выберите данные, а не таблицу.')
            return False
        else:
            return True

    @staticmethod
    def get_tables_list():
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            tables_list = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()
            # print(tables_list)
            return tables_list
        finally:
            cur.close()
            conn.close()


class BottomFrame(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.file_path = ''

        self.file_path_lbl = tk.Label(self, text='File path: \n' + self.file_path)
        self.file_path_lbl.pack(anchor=tk.N)

        self.file_open_button = tk.Button(self, text='File Upload',
                                          command=self.open_file)
        self.file_open_button.pack(anchor=tk.SE)

        self.db_create_button = tk.Button(self, text='Add table to Tasks',
                                          command=self.input_table)
        self.db_create_button.pack(anchor=tk.SE)

        self.db_create_info = tk.StringVar()
        self.db_create_info.set('Please, enter table name:')
        tk.Label(self, textvariable=self.db_create_info).pack(anchor=tk.S)

        self.table_name = tk.StringVar()
        self.table_name_entry = tk.Entry(self, textvariable=self.table_name)
        self.table_name_entry.pack(anchor=tk.S)

        # self.table_box = ttk.Combobox(self, values=self.get_table_list())
        # self.table_box.pack(side=tk.BOTTOM)

    def get_table_list(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            table_list = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()[1:]
            return [table[0] for table in table_list]
        finally:
            conn.close()

    def open_file(self):
        self.file_path = askopenfile(mode='r').name
        self.file_path_lbl['text'] = 'File path: \n' + self.file_path

    def input_table(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            name = self.table_name.get()
            if name == '':
                raise ValueError
            else:
                file = serialize(pd.read_csv(str(self.file_path)))
                cur.execute('INSERT INTO Task_variant (task_id, variant_id, name, table_file) VALUES (2, 1, ?, ?)',
                            (name, file))
                conn.commit()
        except ValueError:
            self.db_create_info.set('Enter table name first!')
        except FileNotFoundError:
            self.db_create_info.set('Upload the file first!')
        finally:
            cur.close()
            conn.close()

    # def create_table(self):
    #     conn = sqlite3.connect('main.sqlite3')
    #     try:
    #         name = self.table_name.get()
    #         if name is None:
    #             raise ValueError
    #         else:
    #             pd.read_csv(str(self.file_path)).to_sql(name, conn, if_exists='replace')
    #             self.db_create_info.set('Table created successfully')
    #             App.update_list(self)
    #     except ValueError:
    #         self.db_create_info.set('Enter table name first!')
    #     except FileNotFoundError:
    #         self.db_create_info.set('Upload the file first!')
    #     finally:
    #         conn.close()
