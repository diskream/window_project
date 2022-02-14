import tkinter as tk
from tkinter import ttk
from functions import get_data, show_table
import pandas as pd


class DataView(tk.Tk):
    def __init__(self, geo, table, data, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Treeview.Heading', background='#42aaff')

        # Set 16:9 window geometry
        self.WIDTH = 16 * geo // 9
        self.HEIGHT = geo
        self.geometry(f'{self.WIDTH}x{self.HEIGHT}')

        self.table = table[0]
        self.data = data
        self.title('Редактирование таблицы ' + str(self.data))
        self.pd_data = get_data(self)

        # Создание фреймов для корректного распределения элементов по окну
        self.table_frm = tk.LabelFrame(self, height=self.HEIGHT * 0.77)
        self.table_frm.pack(fill=tk.BOTH, expand=True)
        self.action_frm = tk.Frame(self)
        self.action_frm.pack(fill=tk.BOTH, expand=True)
        self.tv1_frm = tk.Frame(self.table_frm)
        self.tv1_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.tv2_frm = tk.Frame(self.table_frm, height=10)
        self.tv2_frm.pack(side=tk.TOP, fill=tk.X)
        self.af1_frm = tk.Frame(self.action_frm)
        self.af1_frm.pack(side=tk.BOTTOM)

        self.tv = ttk.Treeview(self.tv1_frm, show='headings', style='Treeview')
        show_table(self)

        self.warn_var = tk.StringVar()
        self.warn_lbl = tk.Label(self.action_frm, textvariable=self.warn_var)
        self.warn_lbl.pack(anchor=tk.N)
        self.isnull_var = tk.StringVar()

        self.isnull_lbl = tk.Label(self.action_frm, text=self.check_empty())
        self.isnull_lbl.pack(anchor=tk.NW)

        # Buttons
        tk.Button(self.af1_frm, text='Преобразование данных', command=self.str_to_num).pack(side=tk.LEFT, pady=20,
                                                                                            padx=20)
        tk.Button(self.af1_frm, text='Удаление колонки', command=self.del_column).pack(side=tk.LEFT, pady=20, padx=20)
        tk.Button(self.af1_frm, text='Добавление колонки', command=self.add_column).pack(side=tk.LEFT, pady=20, padx=20)
        tk.Button(self.af1_frm, text='Обработка пустых значений', command=self.empty_data).pack(side=tk.LEFT, pady=20,
                                                                                                padx=20)
        print(self.tasks_dict)
    def str_to_num(self):
        pass

    def del_column(self):
        DeleteWindow('Удаление колонки', self.table, self.data, self.pd_data)

    def add_column(self):
        pass

    def empty_data(self):
        pass

    def check_empty(self):
        if self.pd_data.isnull().sum().nunique() == 1:
            return 'Пропущенных значений не обнаружено'
        else:
            isnull_cols = list()
            for col in self.pd_data.columns:
                ser = pd.isnull(self.pd_data[col])
                if ser.nunique() == 2:
                    isnull_cols.append(col)
            if len(isnull_cols) <= 3:
                return 'Пропущены значения в следующих столбцах:\n' + '\n'.join(isnull_cols)
            else:
                return 'Пропущены значения в следующем количестве столбцов: ' + str(len(isnull_cols))


class DeleteWindow(tk.Tk):
    def __init__(self, title, table, data, pd_data, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry('500x500')
        self.title(f'{title}')

        self.table = table,
        self.data = data,
        self.pd_data = pd_data

        lb_frm = tk.Frame(self, bg='blue')
        self.action_frm = tk.Frame(self, bg='red')
        confirm_frm = tk.Frame(self.action_frm)
        self.columns_lb = tk.Listbox(lb_frm, selectmode=tk.EXTENDED)
        for col in self.pd_data.columns:
            self.columns_lb.insert(tk.END, col)
        lb_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.action_frm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        confirm_frm.pack(side=tk.BOTTOM)
        tk.Label(lb_frm, text='Пожалуйста, выберете одну или\nнесколько колонок для удаления').pack()
        self.columns_lb.pack(fill=tk.BOTH, expand=1)

        tk.Button(confirm_frm, text='Закрыть').pack(side=tk.RIGHT, padx=20, pady=5)
        tk.Button(confirm_frm, text='Сохранить').pack(side=tk.RIGHT, padx=20, pady=5)

        self.warn_lbl = tk.Label(self.action_frm, text='', pady=20)
        self.warn_lbl.pack(side=tk.TOP)
        tk.Button(self.action_frm, text='Удалить', command=self.del_col, width=15, pady=35).pack(side=tk.TOP)

    def del_col(self):
        cols = []
        columns = self.columns_lb.curselection()
        for column in columns:
            cols.append(self.columns_lb.get(column))
        if len(cols) == 0:
            self.warn_lbl.configure(text='Не выбрана ни одна колонка для удаления.')
        else:
            self.warn_lbl.configure(text='')
            self.pd_data.drop(columns=cols)
