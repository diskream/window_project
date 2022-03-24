import tkinter as tk
import pandas as pd
import json
from tkinter import ttk
import sqlite3
from functional_views.table_view import TableView
from tkinter.filedialog import askopenfile
from machine_learning.classification import ClassificationView
from machine_learning.clustering import ClusteringView
from machine_learning.neural_network import NNView
from functional_views.data_views import DataView
from functional_views.visualization import VisualisationView
from tools.functions import serialize, get_db


class TopFrame(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        ttk.Frame.__init__(self, master)
        self.master = master
        ttk.Label(self, text='Выберите данные из вложенного списка:').pack(side=tk.TOP, ipady=10)
        # Creating hierarchical treeview
        self.tv_hierarchy = ttk.Treeview(self, height=13, show='tree')
        self.tv_hierarchy.pack(side=tk.TOP)
        self.db = get_db()
        self.insert_tv(self.db)

        ttk.Button(self, text='Обновить таблицу', command=self.update_table).pack(side=tk.TOP, pady=10)
        self.btn_lb_frm = ttk.LabelFrame(self, text='Выбор действия')
        self.btn_lb_frm.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)
        self.btn_frm = ttk.Frame(self.btn_lb_frm)
        self.btn_frm.pack()
        # Buttons for the new windows
        btn_pad = {
            'padx': 10,
            'pady': 5
        }
        ttk.Button(self.btn_frm, text='Классификация', command=self.open_ml).grid(row=1, column=0, **btn_pad)
        ttk.Button(self.btn_frm, text='Обзор данных', command=self.open_table).grid(row=0, column=0, **btn_pad)
        ttk.Button(self.btn_frm, text='Редактирование', command=self.open_data).grid(row=0, column=1,
                                                                                              **btn_pad)
        ttk.Button(self.btn_frm, text='Визуализация', command=self.open_visual).grid(row=0, column=2,
                                                                                              **btn_pad)
        ttk.Button(self.btn_frm, text='Кластеризация', command=self.open_cluster).grid(row=1, column=1,
                                                                                                **btn_pad)
        ttk.Button(self.btn_frm, text='Нейронные сети', command=self.open_neural).grid(row=1, column=2,
                                                                                                **btn_pad)

        self.warn_var = tk.StringVar()
        self.warn_lbl = ttk.Label(self, textvariable=self.warn_var)
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
            self.tv_hierarchy.insert('', '0', table, text=table, tags='table')
            if table != 'Task_variant':
                for entry in entries:
                    self.tv_hierarchy.insert(table, tk.END, text=entry.name, tags=entry)
            else:
                # Находим количество родительских данных
                parents = []
                for entry in entries:
                    parents.append(entry.parent_name)
                # Создаем для каждого родителя новую ветку:
                for parent in set(parents):
                    self.tv_hierarchy.insert(table, '1', parent, text=parent, tags='Tasks')
                for entry in entries:
                    self.tv_hierarchy.insert(entry.parent_name, '2', text=entry.name, tags=entry)

    def open_table(self):
        print(self.tv_hierarchy.item(self.tv_hierarchy.selection()))
        table = self.tv_hierarchy.item(self.tv_hierarchy.selection())
        if table['text'] == '':
            self.warn_var.set('Данные не выбраны. Пожалуйста, выберетите данные из списка выше.')
            return
        elif table['tags'][0] == 'table':
            TableView(True, table['text'])
        else:
            s = '{' + table['tags'][0].replace('(', '').replace(',)', '').replace("'", '"').replace('None', '0') + '}'
            entry = json.loads(s)
            TableView(False, entry=self.get_entry(entry['table'], entry['task_id'],
                                                  entry['variant_id'] if 'variant_id' in entry else None))

    def open_ml(self):
        entry = self.get_dict_entry()
        ClassificationView(self.get_entry(entry['table'], entry['task_id'],
                                          entry['variant_id'] if 'variant_id' in entry else None))

    def open_data(self):
        entry = self.get_dict_entry()
        DataView(self.master.HEIGHT, entry=self.get_entry(entry['table'], entry['task_id'],
                                                          entry['variant_id'] if 'variant_id' in entry else None))

    def open_visual(self):
        entry = self.get_dict_entry()
        VisualisationView(self.get_entry(entry['table'], entry['task_id'],
                                         entry['variant_id'] if 'variant_id' in entry else None))

    def open_cluster(self):
        entry = self.get_dict_entry()
        ClusteringView(self.get_entry(entry['table'], entry['task_id'],
                                      entry['variant_id'] if 'variant_id' in entry else None))

    def open_neural(self):
        entry = self.get_dict_entry()
        NNView(self.get_entry(entry['table'], entry['task_id'],
                              entry['variant_id'] if 'variant_id' in entry else None))

    def get_dict_entry(self):
        table = self.tv_hierarchy.item(self.tv_hierarchy.selection())
        if self.table_check(table):
            s = '{' + table['tags'][0].replace('(', '').replace(',)', '').replace("'", '"').replace('None', '0') + '}'
            entry = json.loads(s)
            return entry

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
        self.tv_hierarchy.delete(*self.tv_hierarchy.get_children())
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


class BottomFrame(ttk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        ttk.LabelFrame.__init__(self, parent)
        self.parent = parent
        self.file_path = ''
        self.configure(text='Загрузка данных')

        self.btn_frm = ttk.Frame(self)
        self.btn_frm.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_path_lbl = ttk.Label(self, text='Путь к файлу: \n' + self.file_path)
        self.file_path_lbl.pack(anchor=tk.N)

        self.file_open_button = ttk.Button(self.btn_frm, text='Загрузить файл', command=self.open_file, width=20)
        self.file_open_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.db_create_button = ttk.Button(self.btn_frm, text='Добавить в таблицу Tasks', command=self.input_table,
                                          width=20)
        self.db_create_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.db_create_info = tk.StringVar()
        self.db_create_info.set('Введите название данных:')
        ttk.Label(self, textvariable=self.db_create_info).pack(anchor=tk.S)

        self.table_name = tk.StringVar()
        self.table_name_entry = ttk.Entry(self, textvariable=self.table_name, width=40)
        self.table_name_entry.pack(anchor=tk.S)

        # self.table_box = ttk.Combobox(self, values=self.get_table_list())
        # self.table_box.pack(side=tk.BOTTOM)

    @staticmethod
    def get_table_list():
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
                _sql = 'SELECT MAX(task_id) FROM Tasks'
                task_id = cur.execute(_sql).fetchone()[0] + 1
                cur.execute('INSERT INTO Tasks (task_id, name, table_file) VALUES (?, ?, ?)',
                            (task_id, name, file))
                conn.commit()
                self.db_create_info.set(f'Файл {name} загружен успешно.')
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
