import sqlite3
import pickle
import tkinter as tk
import pandas as pd
from tkinter import ttk
from tkinter.filedialog import askopenfile
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.HEIGHT = 700
        self.WIDTH = (7 * self.HEIGHT) // 9
        self.geometry(f'{self.WIDTH}x{self.HEIGHT}')
        # self.resizable(width=False, height=False)
        self.title('Brand New Window')

        # self.table_list_frame = tk.LabelFrame(self.master, text='Table list')

        self.table_list_frame = TopFrame(self)
        self.table_list_frame.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        self.file_upload_frame = BottomFrame(self)
        self.file_upload_frame.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True)

        # self.table_list_frame.pack(anchor=tk.NW)
        # self.to_tree_frame = TopRightFrame(self, text='Open a table')
        # self.to_tree_frame.place(x=250, y=0, anchor='nw', width=250, height=250)

    def update_list(self):
        self.table_list_frame = TopFrame(self, text='Table list')
        self.table_list_frame.place(x=0, y=0, anchor='nw', width=250, height=250)


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

        self.table_frm = tk.LabelFrame(self, height=self.HEIGHT * 0.60)
        self.table_frm.pack(fill=tk.BOTH, expand=True)
        self.action_frm = tk.Frame(self)
        self.action_frm.pack(fill=tk.BOTH, expand=True)

        self.tv = ttk.Treeview(self.table_frm, show='headings', style='Treeview')
        show_table(self)


class MLView(tk.Tk):
    def __init__(self, table, data, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('500x530')
        self.resizable(width=False, height=False)
        self.table = table[0]
        self.data = data
        self.dataframe = self.get_data()
        self.clf = 'classifier pattern'
        self.accuracy = 0
        tk.Label(self, text='Data: ' + self.data + '\nfrom ' + self.table + ' table.').grid(row=0, column=1)
        tk.Label(self, text='Please, choose ML algorithm:').grid(row=1, column=0)

        # configuring combobox
        self.alg_box = ttk.Combobox(self, values=[
            'Decision Tree',
            'Random Forest',
            'k Nearest Neighbors'
        ], width=15)
        self.alg_box.grid(row=2, column=0)
        self.alg_box.current(0)
        self.update_title()
        self.alg_box_button = tk.Button(self, text='Enter', command=self.alg_configuration
                                        ).grid(row=2, column=1, sticky=tk.W)

    def update_title(self):
        self.title('Working on {} model'.format(self.alg_box.get()))

    def alg_configuration(self):
        alg = self.alg_box.get()
        self.update_title()
        param_frame = tk.LabelFrame(self, text='{} configuration'.format(alg), width=450)
        param_frame.place(height=255, width=450, x=25, y=100)
        clf_frame = tk.LabelFrame(self, text='{} fit and score'.format(alg))
        clf_frame.place(x=25, y=360, width=450, height=150)

        def get_tree_params(outer_self, frame):
            params['criterion'] = criterion.get()
            params['max_depth'] = int(max_depth.get())
            params['min_samples_split'] = int(min_samples_split.get())
            params['min_samples_leaf'] = int(min_samples_leaf.get())
            params['min_weight_fraction_leaf'] = float(min_weight_fraction_leaf.get())
            params['random_state'] = random_state.get()
            if params['random_state'] == 'None':
                params['random_state'] = None

            params_list = ''
            for key, value in params.items():
                if key != 'random_state':
                    params_list += str(key) + ': ' + str(value) + '\n'
                else:
                    params_list += str(key) + ': ' + str(value)
            tk.Label(param_frame, text=params_list, justify=tk.LEFT).grid(row=11, column=0, columnspan=2, sticky=tk.W)
            return outer_self.get_clf(params, frame)

        tk.Button(param_frame, text='Get params', command=lambda: get_tree_params(self, clf_frame), width=14).grid(
            row=10, column=1)
        params = {}
        default_params = {
            'criterion': tk.StringVar(param_frame, value='entropy'),
            'max_depth': tk.StringVar(param_frame, value=15),
            'min_samples_split': tk.StringVar(param_frame, value=2),
            'min_samples_leaf': tk.StringVar(param_frame, value=1),
            'min_weight_fraction_leaf': tk.StringVar(param_frame, value=0.0),
            'random_state': tk.StringVar(param_frame, value='None')
        }
        if alg == 'Decision Tree':
            tk.Label(param_frame, text='criterion').grid(row=0, column=0)
            criterion = tk.Entry(param_frame, width=14, textvariable=default_params['criterion'])
            criterion.grid(row=1, column=0)

            tk.Label(param_frame, text='max_depth').grid(row=0, column=1)
            max_depth = tk.Entry(param_frame, width=14, textvariable=default_params['max_depth'])
            max_depth.grid(row=1, column=1)

            tk.Label(param_frame, text='min_samples_split').grid(row=0, column=2)
            min_samples_split = tk.Entry(param_frame, width=14, textvariable=default_params['min_samples_split'])
            min_samples_split.grid(row=1, column=2)

            tk.Label(param_frame, text='min_samples_leaf').grid(row=2, column=0)
            min_samples_leaf = tk.Entry(param_frame, width=14, textvariable=default_params['min_samples_leaf'])
            min_samples_leaf.grid(row=3, column=0)

            tk.Label(param_frame, text='min_weight_fraction_leaf').grid(row=2, column=1)
            min_weight_fraction_leaf = tk.Entry(param_frame, width=14,
                                                textvariable=default_params['min_weight_fraction_leaf'])
            min_weight_fraction_leaf.grid(row=3, column=1)

            tk.Label(param_frame, text='random_state').grid(row=2, column=2)
            random_state = tk.Entry(param_frame, width=14, textvariable=default_params['random_state'])
            random_state.grid(row=3, column=2)

    def get_clf(self, params, frame):
        self.clf = DecisionTreeClassifier(**params)

        tk.Label(frame, text='Please, set the target variable:').grid(row=0, column=0)
        target = ttk.Combobox(frame, values=list(self.dataframe.columns))
        target.grid(row=1, column=0)
        target.current(0)

        tk.Label(frame, text='Please, set the size of test data.', justify=tk.LEFT).grid(row=2, column=0)
        split = tk.Entry(frame)
        split.grid(row=3, column=0)

        def process_model(cur_frame, out_self, tar, spl):
            df = out_self.dataframe
            X = df.drop(tar, axis=1)
            y = df[tar]
            if spl != '':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(spl))
                out_self.clf.fit(X_train, y_train)
                out_self.accuracy = out_self.clf.score(X_test, y_test)
                tk.Label(cur_frame,
                         text='Model accuracy: ' + str(out_self.accuracy)
                         ).grid(row=0, column=1)

        tk.Button(frame, text='Fit the model',
                  command=lambda: process_model(frame, self, target.get(), split.get())
                  ).grid(row=4, column=0)

        def save_to_db(data, table, model, acc):
            conn = sqlite3.connect('main.sqlite3')
            cur = conn.cursor()
            try:
                query = cur.execute(f'SELECT * FROM {table} WHERE name = "{data}"').fetchall()[0]
                foreign_key = int(query[0])
                key = int(query[1])
                cur.execute('INSERT INTO Models (model_id, variant_id, task_id, name, model_code, accuracy) '
                            'VALUES (?, ?, ?, ?, ?, ?)', (key, key, foreign_key, data + ' model', model, acc))
                conn.commit()
            finally:
                conn.close()

        tk.Button(frame, text='Save model to database',
                  command=lambda: save_to_db(self.data, self.table,
                                             serialize(self.clf),
                                             self.accuracy)).place(x=290, y=90)

    def get_data(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            query = f'SELECT table_file FROM {self.table} WHERE name = "{self.data}"'
            return deserialize(cur.execute(query).fetchall()[0][0])
        finally:
            conn.close()


def show_table(method):
    '''
    Отображает таблицу
    :param method: используется для ссылки на self
    :return:
    '''
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    i = 0
    if method.table is not None:
        data = deserialize(cur.execute(f'SELECT table_file FROM {method.table} '
                                              f'WHERE name = "{method.data}"').fetchall()[0][0])
        method.tv['columns'] = list(data.columns)
        for column in method.tv['columns']:
            method.tv.heading(column, text=column)
            method.tv.column(column, width=60)
        rows = data.to_numpy().tolist()
        for row in rows:
            if i % 2 == 0:
                method.tv.insert('', tk.END, values=row, tag='even')
            else:
                method.tv.insert('', tk.END, values=row, tag='odd')
            i += 1
    else:
        data = cur.execute(f'SELECT * FROM {method.data}').fetchall()
        cur.execute('SELECT * FROM {} WHERE 1=0'.format(method.data))
        columns = [d[0] for d in method.cur.description]
        method.tv['columns'] = columns
        for column in method.tv['columns']:
            method.tv.heading(column, text=column)
            method.tv.column(column, width=60)
        for row in data:
            if i % 2 == 0:
                method.tv.insert('', tk.END, values=row, tag='even')
            else:
                method.tv.insert('', tk.END, values=row, tag='odd')
            i += 1

    method.tv.tag_configure('even', background='#E8E8E8')
    method.tv.tag_configure('odd', background='#DFDFDF')
    method.tv.pack(fill=tk.BOTH, expand=True)  # expand=True, fill=tk.BOTH)
    # ysb = ttk.Scrollbar(method.master, orient=tk.VERTICAL, command=method.tv.yview)
    # xsb = ttk.Scrollbar(method.master, orient=tk.HORIZONTAL, command=method.tv.xview)
    # method.tv.configure(yscroll=ysb.set, xscroll=xsb.set)
    # ysb.pack(side=tk.RIGHT, fill=tk.Y)
    # xsb.pack(side=tk.BOTTOM, fill=tk.X)

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


if __name__ == '__main__':
    app = App()
    app.mainloop()
