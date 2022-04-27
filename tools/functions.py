import pickle
import tkinter as tk
from tkinter import ttk
from tools.models import *
from keras.engine.sequential import Sequential
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object


# Примечание: в функциях есть повторяющиейся строки с созданием соединения и курсора.
# Сделать либо деякоратор, либо функцию, которая будет возвращать соединение.

def treeview_sort_column(tv, col, reverse):
    """
    Функция сортировки по возрастанию и убыванию колонки.
    Примечание: функция работает некорректно и сортирует только последний столбец.
    :param tv: фрейм таблицы
    :param col: список колонок
    :param reverse: bool. Отвечает за изменение направления сортировки
    :return:
    """
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    l.sort(key=lambda t: float(t[0]), reverse=reverse)

    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

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


def show_table(method, sb_configure=True, pd_data=None):
    """
    Отображает таблицу
    :param method: используется для ссылки на self. Нужно для использования разными классами
    :param pd_data: если нужно обновить таблицу из DataFrame
    :return:
    """
    column_width = 100
    i = 0  # итератор для чередования цветов записей в таблице
    if method.table is not None:
        data = get_data(method) if pd_data is None else pd_data
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
    if sb_configure:
        ysb = ttk.Scrollbar(method.tv1_frm, orient=tk.VERTICAL, command=method.tv.yview)
        xsb = ttk.Scrollbar(method.tv2_frm, orient=tk.HORIZONTAL, command=method.tv.xview)
        method.tv.configure(yscroll=ysb.set, xscroll=xsb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
    method.tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

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
    """
    Переводит поданные на вход данные в последовательность байтов.
    :param file: данные для преобразование
    :return:
    """
    if isinstance(file, Sequential):
        return pickle.dumps(serialize_keras_object(file))
    return pickle.dumps(file)


def deserialize(file):
    """
    Функция, обратная serialize.
    Переводит последовательность байтов обратно в обхект.
    :param file: последовательность байтов для преобразования.
    :return:
    """
    return pickle.loads(file)


def upload_data(table, **data):
    """
    Выгружает новый вариант данных в БД в таблицу Task_variant.
    Обрабатывает словарь с данными для корректной записи в БД.
    Получает последний variant_id для семейства данных для инкремента и занесения новых данных
    с корректным ключом.
    :param table: название таблицы
    :param data: словарь, в октором ключи - названия колонок, а значения - данные
    :return:
    """
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        if 'variant_id' not in data:
            try:
                data['variant_id'] = cur.execute(f'SELECT MAX(variant_id) FROM {table} '
                                                 f'WHERE task_id = {data["task_id"]}').fetchone()[0] + 1
            except Exception as _ex:
                print(_ex)
                data['variant_id'] = 1
        if data['variant_id'] is None:
            variant = cur.execute(f'SELECT MAX(variant_id) FROM {table} '
                                  f'WHERE task_id = {data["task_id"]}').fetchone()[0]
            data['variant_id'] = variant + 1
        data['name'] = f'{data["name"][:-2]}_{data["variant_id"]}'
        placeholders = ', '.join('?' for _ in data.values())  # форматирование данных для запроса
        cols = ', '.join(data.keys())  # форматирование названия колонок для запроса
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
    """
    Обновляет класс, соответствующий записи в БД, занося закодированный файл.
    Это нужно для того, чтобы при инициализации проекта и создании классов для каждой записи
    не забивать память.
    Конструкция if - else нужна для проверки, был ли файл помещен в класс ранее.
    :param entry: класс записи в БД
    :return:
    """
    if entry.table_file is not None:
        return
    else:
        sql = f'SELECT table_file FROM {entry.table} WHERE task_id = {entry.task_id} '
        if hasattr(entry, 'variant_id'):
            sql += f'AND variant_id = {entry.variant_id} '
        with sqlite3.connect('main.sqlite3') as conn:
            entry.table_file = conn.cursor().execute(sql).fetchall()[0][0]


def get_entry(table, **data):
    """
    Создает объект записи из БД.
    Запросом получает информацию о записи в виде списка.
    :param table:
    :param data:
    :return: класс Variant
    """
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        if isinstance(table, tuple):
            table = table[0]
        query = cur.execute(f'SELECT * FROM {table} WHERE task_id = {data["task_id"]}'
                            f' AND variant_id = {data["variant_id"]}').fetchall()[0]
        return Variant(*query)  # * - распаковка списка
    finally:
        conn.close()


def save_model(entry, clf, accuracy=None, name:str=None, desc:str=None, path=False):
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        data = {
            'model_id': None,
            'task_id': entry.task_id,
            'variant_id': entry.variant_id,
            'name': None,
            'accuracy': accuracy if accuracy else None,
            'bin_file': serialize(clf) if not path else None,
            'description': desc if desc else None
        }
        try:
            model_id = cur.execute(f'SELECT MAX(model_id) FROM Models WHERE task_id = {data["task_id"]}'
                                   f' AND variant_id = {data["variant_id"]}').fetchone()[0] + 1
            data['model_id'] = model_id
        except TypeError:
            data['model_id'] = 1
        if name:
            data['name'] = name + f'_{data["model_id"]}'
        else:
            data['name'] = entry.name + f'_m_{data["model_id"]}' if not isinstance(entry.name, tuple) \
                else entry.name[0] + f'_m_{data["model_id"]}'
        if path:
            clf.save(path + data['name'])
            data['bin_file'] = serialize(path + data['name'])
        placeholders = ', '.join('?' for _ in data.values())  # форматирование данных для запроса
        cols = ', '.join(data.keys())  # форматирование названия колонок для запроса
        _sql = f'INSERT INTO Models ({cols}) VALUES ({placeholders})'
        cur.execute(_sql, tuple(data.values()))
    finally:
        conn.commit()
        conn.close()

def get_models_list() -> list:
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        models = []
        for model in cur.execute('SELECT name FROM Models').fetchall():
            models.append(model[0])
        return models
    finally:
        conn.close()