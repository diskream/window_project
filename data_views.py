import tkinter as tk
from tkinter import ttk
from functions import serialize, deserialize, show_table, update_entry, upload_data, get_entry
import pandas as pd
import io


class DataView(tk.Tk):
    """
    Главное окно редактирования данных
    """

    def __init__(self, geo, entry):
        tk.Tk.__init__(self)

        self.style = ttk.Style()
        self.style.configure('Treeview.Heading', background='#42aaff')

        # Установка соотношения сторон 16:9
        self.WIDTH = 16 * geo // 9
        self.HEIGHT = geo
        self.geometry(f'{self.WIDTH}x{self.HEIGHT}')

        # Entry - класс записи в базе данных
        self.entry = entry
        self.table = entry.table
        # Проверка, является ли название данных котрежем
        # Это связано с тем, что данные выдаются в запросе кортежем
        if isinstance(self.entry.name, tuple):
            self.entry.name = self.entry.name[0]
        self.data = self.entry.name
        update_entry(self.entry)

        self.title('Редактирование таблицы ' + self.entry.name)
        self.pd_data = deserialize(self.entry.table_file)

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
        # Создание таблицы
        self.tv = ttk.Treeview(self.tv1_frm, show='headings', style='Treeview')
        show_table(self)
        # lbl - область текста, var - текстовые переменные, в которых хранятся сообщения
        self.warn_var = tk.StringVar()
        self.warn_lbl = tk.Label(self.action_frm, textvariable=self.warn_var)
        self.warn_lbl.pack(anchor=tk.N)
        self.null_columns = None
        # Лейбл, отображающий количество пустых значений
        self.isnull_lbl = tk.Label(self.action_frm, text=self.check_empty())
        self.isnull_lbl.pack(anchor=tk.NW)

        # Кнопки действий
        tk.Button(self.af1_frm, text='Преобразование данных',
                  command=self.data_preparation).pack(side=tk.LEFT, pady=20, padx=20)
        tk.Button(self.af1_frm, text='Удаление колонки', command=self.del_column).pack(side=tk.LEFT, pady=20, padx=20)
        tk.Button(self.af1_frm, text='Добавление колонки', command=self.add_column).pack(side=tk.LEFT, pady=20, padx=20)
        tk.Button(self.af1_frm, text='Обработка пустых значений', command=self.empty_data).pack(side=tk.LEFT, pady=20,
                                                                                                padx=20)
        tk.Button(self.af1_frm, text='Информация по данным', command=self.description).pack(side=tk.LEFT, pady=20,
                                                                                            padx=20)

    def data_preparation(self):
        DataPreparation(self.entry, self.pd_data, self, self.WIDTH, self.HEIGHT)

    def del_column(self):
        """
        Открывает окно с действиями для удаления колонок
        :return:
        """
        DeleteWindow(self.entry, self.pd_data, self, self.HEIGHT)

    def add_column(self):
        AddWindow(self.entry, self.pd_data, self, self.WIDTH, self.HEIGHT)

    def empty_data(self):
        pass

    def description(self):
        DescriptionWindow(self.pd_data, self.WIDTH, self.HEIGHT)

    def check_empty(self):
        """
         Главная функция проверки пропущенных значений.
         Ищет по всему Dataframe пропущенные значения, агригирует функцией sum
         количество пропущенных в каждом столбце.
         Если количество уникальных значений равно 1 (в каждой колонке 0 пропущенных значений),
         выводит соответствуещее уведомление. В противном случае заносит в список все колонки
         с пропущенными значениями и возвращает их.
        :return:
        """
        if self.pd_data.isnull().sum().nunique() == 1:
            return 'Пропущенных значений не обнаружено'
        else:
            isnull_cols = list()
            for col in self.pd_data.columns:
                ser = pd.isnull(self.pd_data[col])
                if ser.nunique() == 2:
                    isnull_cols.append(col)
            self.null_columns = isnull_cols
            if len(isnull_cols) <= 3:
                return 'Пропущены значения в следующих столбцах:\n' + '\n'.join(isnull_cols)
            else:
                return 'Пропущены значения в следующем количестве столбцов: ' + str(len(isnull_cols))

    def close(self):
        self.destroy()


class DeleteWindow(tk.Tk):
    """
    Окно с действиями для удаления колонок
    """

    def __init__(self, entry, pd_data, parent, height):
        tk.Tk.__init__(self)

        self.geometry('500x500')
        self.parent = parent
        self.HEIGHT = height
        self.entry = entry
        self.pd_data = pd_data  # данные в формате Dataframe
        self.title(f'{self.entry.name}')
        self.columns_to_delete = list()  # список выбранных колонок для удаления
        # создание фреймов для удобного размещения элементов
        lb_frm = tk.Frame(self)
        self.action_frm = tk.Frame(self)
        confirm_frm = tk.Frame(self.action_frm)
        # создание listbox со всеми колонками в данных. Параметр EXTENDED позволяет через
        # ctrl или shift выделять несколько значений
        self.columns_lb = tk.Listbox(lb_frm, selectmode=tk.EXTENDED)
        for col in self.pd_data.columns:  # заполнение данными
            self.columns_lb.insert(tk.END, col)
        # размещение фреймов
        lb_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.action_frm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        confirm_frm.pack(side=tk.BOTTOM)
        tk.Label(lb_frm, text='Пожалуйста, выберете одну или\nнесколько колонок для удаления').pack()
        self.columns_lb.pack(fill=tk.BOTH, expand=1)

        tk.Button(confirm_frm, text='Отмена', command=self.cancel).pack(side=tk.RIGHT, padx=20, pady=5)
        tk.Button(confirm_frm, text='Сохранить', command=self.save).pack(side=tk.RIGHT, padx=20, pady=5)

        self.warn_lbl = tk.Label(self.action_frm, text='', pady=20)
        self.warn_lbl.pack(side=tk.TOP)
        tk.Button(self.action_frm, text='Удалить', command=self.del_col, pady=35,
                  bg='#abcdef', activebackground='#a6caf0').pack(side=tk.TOP, fill=tk.X)

    def del_col(self):
        """
        Получает информацию из выделенных элементов listbox
        Ничего не возвращает, заносит информацию в аргументы класса
        :return:
        """
        cols = []  # Названия колонок для удаления
        columns = self.columns_lb.curselection()  # Индексы колонок для удаления
        for column in columns:
            cols.append(self.columns_lb.get(column))
        if len(cols) == 0:
            self.warn_lbl.configure(text='Не выбрана ни одна колонка для удаления.')
        else:
            self.warn_lbl.configure(text='')
            for i in columns[::-1]:  # удаляет только выбранные элементы
                self.columns_lb.delete(i)
            self.columns_to_delete = cols

    def save(self):
        """
        Отвечает за сохранение результатов в БД.
        Конструкция if - else нужна для корректной записи в БД в правильную таблицу.
        data - список данных, которые будут занесены в БД.
        out - словарь, ключи которого являются названиями колонок, а значения - данными.
        :return:
        """
        self.pd_data = self.pd_data.drop(self.columns_to_delete, axis=1)
        DataView.destroy(self.parent)
        save_to_db(self)

    def cancel(self):
        """
        Закрывает окно удаления и возвращает окно редактирования
        :return:
        """
        self.destroy()


class DescriptionWindow(tk.Tk):
    """
    Окно с описательной статистикой
    """

    def __init__(self, pd_data, width, height):
        tk.Tk.__init__(self)

        self.geometry(f'{int(width // 1.5)}x{height // 2}')
        self.title('Просмотр описатльной статистики')
        self.tv = ttk.Treeview(self, show='headings')
        self.show_table(pd_data)
        self.tv.pack(fill=tk.BOTH, expand=1)

    def show_table(self, pd_data):
        """
        Преобразовывает и заносит данные в таблицу.
        Примечание по переменной description:
        Dataframe описательной статистики имеет сложный индекс. Объект Treeview не поддерживает
        вертикальные заголовки, поэтому необходимо удалить сложный индекс.
        :param pd_data: pd.Dataframe данных, для которых используется описательная статистика
        :return:
        """
        description = pd_data.describe().reset_index().rename(columns={'index': ''})
        self.tv['columns'] = list(description.columns)
        for col in self.tv['columns']:
            self.tv.heading(col, text=col)
            self.tv.column(col, width=50)
        rows = description.to_numpy().tolist()
        for row in rows:
            self.tv.insert('', tk.END, values=row)


class AddWindow(tk.Tk):
    """
    Возможные варианты добавления:
    1. Перевод timestamp -> datetime
    2. Datetime -> date / time
    3. Колонка1 = / >= / <= / > / < Колонка2
    4. Колонка3 = Колонка1 * / "/" / - / + Колонка2
    5. Колонка2 = % от Колонки1
    6. Колонка1 * / "/" / - / + Число
    Функции не ловят испключения. Если будет время - добавить проверку исключений.
    """

    def __init__(self, entry, pd_data, parent, width, height):
        tk.Tk.__init__(self)

        self.geometry(f'{int(width * 1.2)}x{int(height * 1.2)}')
        self.entry = entry
        self.title('Добавление колонки в данные ' + self.entry.name)
        self.pd_data = pd_data
        self.columns = list(self.pd_data.columns)
        self.parent = parent
        # Разделение окна на таблицу и область действий
        self.table_frm = tk.Frame(self)
        self.table_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.buttons_frm = tk.Frame(self, bg='#abcdef')
        self.buttons_frm.pack(side=tk.BOTTOM, fill=tk.X)
        self.action_frm = tk.Frame(self, bg='#abcdef')
        self.action_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        # Добавление областей для скроллбаров
        self.tv1_frm = tk.Frame(self.table_frm)
        self.tv1_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.tv2_frm = tk.Frame(self.table_frm, height=10)
        self.tv2_frm.pack(side=tk.TOP, fill=tk.X)
        # Инициализация таблицы
        self.tv = ttk.Treeview(self.tv1_frm, show='headings')
        # Переменные для функции show_table()
        self.table = entry.table
        if isinstance(self.entry.name, tuple):
            self.entry.name = self.entry.name[0]
        self.data = self.entry.name
        # Отрисовка таблицы
        show_table(self)
        # Добавление 6 фреймов
        self.add_frm_1 = tk.LabelFrame(self.action_frm, text='Перевод Timestamp', bg='#abcdef')
        self.add_frm_2 = tk.LabelFrame(self.action_frm, text='Разбиение DateTime', bg='#abcdef')
        self.add_frm_3 = tk.LabelFrame(self.action_frm, text='Сравнение 2 колонок', bg='#abcdef')
        self.add_frm_4 = tk.LabelFrame(self.action_frm, text='Вычисляемая колонка из 2', bg='#abcdef')
        self.add_frm_5 = tk.LabelFrame(self.action_frm, text='Вычисляемая колонка как %', bg='#abcdef')
        self.add_frm_6 = tk.LabelFrame(self.action_frm, text='Вычисляемая колонка из числа', bg='#abcdef')
        # Расположение фреймов
        self.add_frm_1.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=1)
        self.add_frm_2.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=1)
        self.add_frm_3.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=1)
        self.add_frm_4.pack(side=tk.LEFT, anchor=tk.S, fill=tk.BOTH, expand=1)
        self.add_frm_5.pack(side=tk.LEFT, anchor=tk.S, fill=tk.BOTH, expand=1)
        self.add_frm_6.pack(side=tk.LEFT, anchor=tk.S, fill=tk.BOTH, expand=1)
        tk.Button(self.buttons_frm, text='Отмена', command=self.cancel, width=10).pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Button(self.buttons_frm, text='Сохранить', command=self.save, width=10).pack(side=tk.RIGHT, padx=10, pady=5)
        # Добавление элементов в фрейм 1
        for frm in [self.add_frm_1, self.add_frm_2, self.add_frm_3, self.add_frm_4, self.add_frm_5, self.add_frm_6]:
            tk.Label(frm, text='Введите название новой колонки:', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W,
                                                                                     padx=5, pady=5)
        self.ent_1 = tk.Entry(self.add_frm_1, width=30)
        self.ent_2 = tk.Entry(self.add_frm_2, width=30)
        self.ent_3 = tk.Entry(self.add_frm_3, width=30)
        self.ent_4 = tk.Entry(self.add_frm_4, width=30)
        self.ent_5 = tk.Entry(self.add_frm_5, width=30)
        self.ent_6 = tk.Entry(self.add_frm_6, width=30)
        # расположение элементов
        self.ent_1.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.ent_2.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.ent_3.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.ent_4.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.ent_5.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.ent_6.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        # Кнопки сохранения
        tk.Button(self.add_frm_1, text='Добавить', command=self.comm1).pack(side=tk.BOTTOM, padx=5, pady=5)
        tk.Button(self.add_frm_2, text='Добавить', command=self.comm2).pack(side=tk.BOTTOM, padx=5, pady=5)
        tk.Button(self.add_frm_3, text='Добавить', command=self.comm3).pack(side=tk.BOTTOM, padx=5, pady=5)
        tk.Button(self.add_frm_4, text='Добавить', command=self.comm4).pack(side=tk.BOTTOM, padx=5, pady=5)
        tk.Button(self.add_frm_5, text='Добавить', command=self.comm5).pack(side=tk.BOTTOM, padx=5, pady=5)
        tk.Button(self.add_frm_6, text='Добавить', command=self.comm6).pack(side=tk.BOTTOM, padx=5, pady=5)
        # При создании новой колонки используется как минимум одна старая
        tk.Label(self.add_frm_1, text='Выберите колонку с Timestamp', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W,
                                                                                         padx=5)
        tk.Label(self.add_frm_2, text='Выберите колонку с DateTime', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W,
                                                                                        padx=5)
        tk.Label(self.add_frm_3, text='Выберите первую колонку', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W, padx=5)
        tk.Label(self.add_frm_4, text='Выберите первую колонку', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W, padx=5)
        tk.Label(self.add_frm_5, text='Выберите колонку', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W, padx=5, )
        tk.Label(self.add_frm_6, text='Выберите колонку', bg='#abcdef').pack(side=tk.TOP, anchor=tk.W, padx=5)
        # Combobox
        self.cb_1 = ttk.Combobox(self.add_frm_1, values=self.columns)
        self.cb_2 = ttk.Combobox(self.add_frm_2, values=self.columns)
        self.cb_3 = ttk.Combobox(self.add_frm_3, values=self.columns)
        self.cb_4 = ttk.Combobox(self.add_frm_4, values=self.columns)
        self.cb_5 = ttk.Combobox(self.add_frm_5, values=self.columns)
        self.cb_6 = ttk.Combobox(self.add_frm_6, values=self.columns)
        # Расположение Combobox
        self.cb_1.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.cb_2.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.cb_3.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.cb_4.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.cb_5.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        self.cb_6.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        # Работа с Datetime
        tk.Label(self.add_frm_2, text='Выберите тип новой колонки:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                                anchor=tk.W)
        self.dt_cb = ttk.Combobox(self.add_frm_2, values=['Год', 'Месяц', 'День'])
        self.dt_cb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        # Сравнение колонок
        tk.Label(self.add_frm_3, text='Выберите вторую колонку:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                             anchor=tk.W)
        self.cb_3_2 = ttk.Combobox(self.add_frm_3, values=self.columns)
        self.cb_3_2.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        tk.Label(self.add_frm_3, text='Выберите оператор сравнения:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                                 anchor=tk.W)
        self.comp_cb = ttk.Combobox(self.add_frm_3, values=['==', '>', '<', '>=', '<='])
        self.comp_cb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        # Вычисление из двух колонок
        tk.Label(self.add_frm_4, text='Выберите вторую колонку:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                             anchor=tk.W)
        self.cb_4_2 = ttk.Combobox(self.add_frm_4, values=self.columns)
        self.cb_4_2.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        tk.Label(self.add_frm_4, text='Выберите оператор:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                                 anchor=tk.W)
        self.oper_cb = ttk.Combobox(self.add_frm_4, values=['+', '-', '*', '/', '%'])
        self.oper_cb.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        # Вычисление процента
        tk.Label(self.add_frm_5, text='Выберите вторую колонку:', bg='#abcdef', padx=5).pack(side=tk.TOP,
                                                                                             anchor=tk.W)
        self.cb_5_2 = ttk.Combobox(self.add_frm_5, values=self.columns)
        self.cb_5_2.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)

    def comm1(self):
        self.pd_data[self.ent_1.get()] = pd.to_datetime(self.pd_data[self.cb_1], unit='s')

    def comm2(self):
        name = self.ent_2.get()
        column = self.cb_2.get()
        dt = self.dt_cb.get()
        # Завернуть объекты Datetime в словарь и по ключу брать значения не получилось -
        # Придется писать if'ами...
        try:
            if dt == 'Год':
                self.pd_data[name] = self.pd_data[column].dt.year
            elif dt == 'Месяц':
                self.pd_data[name] = self.pd_data[column].dt.month
            elif dt == 'Год':
                self.pd_data[name] = self.pd_data[column].dt.day
        except AttributeError:
            self.pd_data[column].to_datetime()

    def comm3(self):
        name = self.ent_3.get()
        col1 = self.cb_3.get()
        col2 = self.cb_3_2.get()
        op = self.comp_cb.get()
        if op == '==':
            self.pd_data[name] = col1 == col2
        elif op == '>':
            self.pd_data[name] = col1 > col2
        elif op == '<':
            self.pd_data[name] = col1 < col2
        elif op == '>=':
            self.pd_data[name] = col1 >= col2
        elif op == '<=':
            self.pd_data[name] = col1 <= col2

    def comm4(self):
        name = self.ent_4.get()
        col1 = self.cb_4.get()
        col2 = self.cb_4_2.get()
        op = self.oper_cb.get()
        if op == '+':
            self.pd_data[name] = col1 + col2
        elif op == '-':
            self.pd_data[name] = col1 - col2
        elif op == '*':
            self.pd_data[name] = col1 * col2
        elif op == '/':
            self.pd_data[name] = col1 / col2
        elif op == '%':
            self.pd_data[name] = col1 % col2

    def comm5(self):
        name = self.ent_5.get()
        col1 = self.cb_5.get()
        col2 = self.cb_5_2.get()
        try:
            col1 = int(col1)
            col2 = int(col2)
            greater = col1 if col1 > col2 else col2
            less = col1 if col1 < col2 else col2
            self.pd_data[name] = (greater / less) * 100
        except ValueError as _ex:
            print(_ex)

    def comm6(self):
        print(self.ent_6.get())

    def save(self):
        pass

    def cancel(self):
        pass


class DataPreparation(tk.Tk):
    def __init__(self, entry, pd_data, parent, width=None, height=None):
        tk.Tk.__init__(self)

        self.HEIGHT = height
        self.parent = parent
        # self.geometry(f'{width}x{height}')
        self.geometry('500x500')
        self.entry = entry
        self.title('Преобразование данных ' + self.entry.name)
        self.is_edited = False

        self.pd_data = pd_data
        self.action_frm = tk.LabelFrame(self, text='Проеобразование данных в тип int', bg='#abcdef')
        self.action_frm.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.lbl_frm = tk.LabelFrame(self, text='Информация о данных')
        self.lbl_frm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.lb_frm = tk.LabelFrame(self, text='Выбор колонок для преобразования')
        self.lb_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.info_lbl = tk.Label(self.lbl_frm, text='', justify=tk.LEFT)
        self.info_lbl.pack(side=tk.LEFT)
        self.columns_lb = tk.Listbox(self.lb_frm, selectmode=tk.EXTENDED)
        self.n_unique = {}
        self.get_lb()
        self.columns_lb.pack(fill=tk.BOTH, expand=1)

        tk.Button(self.action_frm, text='Отмена', command=self.cancel, width=10).pack(side=tk.RIGHT, anchor=tk.S,
                                                                                      padx=10, pady=10)
        tk.Button(self.action_frm, text='Сохранить', command=self.save, width=10).pack(side=tk.RIGHT, anchor=tk.S,
                                                                                       padx=10, pady=10)
        tk.Button(self.action_frm, text='Преобразовать', command=self.convert_data).pack(side=tk.TOP, anchor=tk.W,
                                                                                         padx=10, pady=10)
        self.get_info()

    def convert_data(self):
        cols = []  # названия колонок для преобразования
        columns = self.columns_lb.curselection()  # индексы колонок для преобразования
        for column in columns:
            cols.append(self.columns_lb.get(column))
        if len(cols) == 0:
            pass  # TODO: добавить уведомление, что не выбрана ни 1 колонка
        else:
            self.pd_data = obj_to_int(self.pd_data, cols=cols)
            self.columns_lb.delete(0, tk.END)
            self.get_lb()
            self.get_info()
            self.is_edited = True

    def get_info(self):
        buf = io.StringIO()
        self.pd_data.info(buf=buf)
        s = '\nКоличество уникальных значений\n'
        for key, value in self.n_unique.items():
            s += (str(key) + ': ' + str(value) + '\n')
        self.info_lbl.configure(text=buf.getvalue() + s[:-2])

    def get_lb(self):
        self.n_unique = {}
        for col in self.pd_data.columns:
            if self.pd_data[col].dtype not in ['int64', 'float64']:
                self.columns_lb.insert(tk.END, col)
                self.n_unique[col] = self.pd_data[col].nunique()

    def save(self):
        if self.is_edited:
            DataView.destroy(self.parent)
            save_to_db(self)
        else:
            self.cancel()

    def cancel(self):
        self.destroy()


# получаем словарь со всеми колонками типа object и их уникальными значениями
def obj_to_int(df, length=100, cols=None):
    if cols is None:
        cols = []
    if len(cols) == 0:
        columns = df.columns
    else:
        columns = cols
    obj_dict = dict()
    for col in columns:
        if df[col].dtype == 'O':
            obj_dict[col] = df[col].unique().tolist()
    # получаем словарь типа object:int64
    # length - максимальное число изменений (изменять каждый раз для конктретного случая)
    for key, value in obj_dict.items():
        temp_dict = {}
        if len(value) < length:
            for i in range(len(value)):
                temp_dict[value[i]] = i
            obj_dict[key] = temp_dict
    for key, value in obj_dict.items():
        df[key] = df[key].replace(value)
    return df


def save_to_db(method):
    if method.entry.table == 'Task_variant':
        data = [method.entry.task_id, None, method.entry.name, serialize(method.pd_data)]
    else:
        data = [method.entry.task_id, method.entry.name, serialize(method.pd_data)]
    out = dict(zip(method.entry.columns, data))
    DataView(method.HEIGHT, get_entry('Task_variant', **upload_data('Task_variant', **out)))
    method.destroy()
