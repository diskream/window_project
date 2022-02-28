import tkinter as tk
import pandas as pd
import json
from tkinter import ttk
import sqlite3
from table_view import TableView
from tkinter.filedialog import askopenfile
from classification import MLView
from data_views import DataView
from functions import serialize, get_db



class TopFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.master = master
        tk.Label(self, text='Выберите данные из вложенного списка:').pack(side=tk.TOP, ipady=10)
        # Creating hierarchical treeview
        self.tv_hier = ttk.Treeview(self, height=13, show='tree')
        self.tv_hier.pack(side=tk.TOP)
        self.db = get_db()
        self.insert_tv(self.db)

        tk.Button(self, text='Обновить таблицу', command=self.update_table).pack(side=tk.TOP, pady=10)
        self.btn_frm = tk.LabelFrame(self, text='Выбор действия')
        self.btn_frm.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        # Buttons for the new windows
        self.ml_window_btn = tk.Button(self.btn_frm, text='Классификация', command=self.open_ml, width=18)
        self.ml_window_btn.pack(side=tk.RIGHT, padx=10, pady=10, anchor=tk.W)
        self.table_open_btn = tk.Button(self.btn_frm, text='Обзор данных', command=self.open_table, width=18)
        self.table_open_btn.pack(side=tk.RIGHT, padx=10, pady=10)
        self.data_btn = tk.Button(self.btn_frm, text='Редактирование', command=self.open_data, width=18)
        self.data_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.warn_var = tk.StringVar()
        self.warn_lbl = tk.Label(self, textvariable=self.warn_var)
        self.warn_lbl.pack(side=tk.BOTTOM)

    def insert_tv(self, db):
        """
        Вставляет в таблицу записи. При вставке данных использует тэги.
        Тэги нужны для определения родителя. Тэг "table" соответствует главной родительской таблице -
        таблице из БД.
        :param db:
        :return:
        """
        for table, entries in db.items():
            self.tv_hier.insert('', '0', table, text=table, tags='table')
            if table != 'Task_variant':
                for entry in entries:
                    self.tv_hier.insert(table, tk.END, text=entry.name, tags=entry)
            else:
                # Находим количество родительских данных
                parents = []
                for entry in entries:
                    parents.append(entry.parent_name)
                # Создаем для каждого родителя новую ветку:
                for parent in set(parents):
                    self.tv_hier.insert(table, '1', parent, text=parent, tags='Tasks')
                for entry in entries:
                    self.tv_hier.insert(entry.parent_name, '2', text=entry.name, tags=entry)

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
            s = '{' + table['tags'][0].replace('(', '').replace(',)', '').replace("'", '"').replace('None', '0') + '}'
            entry = json.loads(s)
            DataView(self.master.HEIGHT, entry=self.get_entry(entry['table'], entry['task_id'],
                                                              entry['variant_id'] if 'variant_id' in entry else None))

    def get_entry(self, table, task_id, variant_id=None):
        for value in self.db[table]:
            if value.task_id == task_id and (value.variant_id if hasattr(value, 'variant_id') else None) == variant_id:
                return value

    def table_check(self, table):
        if table['text'] == '':
            self.warn_var.set('Данные не выбраны. Пожалуйста, выберетите данные из списка выше.')
            return False
        elif table['tags'][0] == 'table':
            self.warn_var.set('Пожалуйста, выберите данные, а не таблицу.')
            return False
        else:
            return True

    def update_table(self):
        self.tv_hier.delete(*self.tv_hier.get_children())
        self.insert_tv(get_db())
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
        self.configure(bg='#abcdef', text='Загрузка данных')

        self.btn_frm = tk.Frame(self, bg='#abcdef')
        self.btn_frm.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_path_lbl = tk.Label(self, text='Путь к файлу: \n' + self.file_path, bg='#abcdef')
        self.file_path_lbl.pack(anchor=tk.N)

        self.file_open_button = tk.Button(self.btn_frm, text='Загрузить файл', command=self.open_file, width=20)
        self.file_open_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.db_create_button = tk.Button(self.btn_frm, text='Добавить в таблицу Tasks', command=self.input_table,
                                          width=20)
        self.db_create_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.db_create_info = tk.StringVar()
        self.db_create_info.set('Введите название данных:')
        tk.Label(self, textvariable=self.db_create_info, bg='#abcdef').pack(anchor=tk.S)

        self.table_name = tk.StringVar()
        self.table_name_entry = tk.Entry(self, textvariable=self.table_name, width=40)
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
        self.file_path_lbl['text'] = 'Путь к файлу: \n' + self.file_path

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
            self.db_create_info.set('Введите название таблицы!')
        except FileNotFoundError:
            self.db_create_info.set('Сначала загрузите файл!')
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
