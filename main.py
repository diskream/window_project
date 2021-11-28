import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
import pandas as pd
import sqlite3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('500x500')
        self.resizable(width=False, height=False)
        self.title('Brand New Window')

        # self.table_list_frame = tk.LabelFrame(self.master, text='Table list')
        self.table_list_frame = TopFrame(self)
        self.table_list_frame.place(x=0, y=0, anchor='nw', width=500, height=250)

        self.file_upload_frame = BottomFrame(self)
        self.file_upload_frame.place(x=10, y=255, anchor='nw', width=480, height=235)

        # self.table_list_frame.pack(anchor=tk.NW)
        # self.to_tree_frame = TopRightFrame(self, text='Open a table')
        # self.to_tree_frame.place(x=250, y=0, anchor='nw', width=250, height=250)

    def update_list(self):
        self.table_list_frame = TopFrame(self, text='Table list')
        self.table_list_frame.place(x=0, y=0, anchor='nw', width=250, height=250)


class TopFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # initializing ListBox and putting values into it
        # self.tables_list = self.get_tables_list()
        # self.list_box = tk.Listbox(self)
        # self.tables_list = sorted([table[0] for table in self.tables_list])
        # for index, table in enumerate(self.tables_list):
        #     self.list_box.insert(index, table)
        # self.list_box.pack(anchor=tk.W)

        tk.Label(self, text='List of all database tables').pack(side=tk.TOP)

        # creating hierarchical treeview
        self.tv_hier = ttk.Treeview(self.parent, height=3, show='tree')
        self.tv_hier.place(x=150, y=20, height=150)
        self.insert_tv()

        # creating buttons
        self.ml_window_button = tk.Button(self, text='Open ML window', command=self.open_ml).pack(side=tk.BOTTOM)
        self.table_open_button = tk.Button(self, text='Open a TreeView in the new window',
                                           command=self.open_table).pack(side=tk.BOTTOM)

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
            print('No table selected. Please, select the table to continue.')
            return
        elif table['tags'][0] == 'table':
            print('Please, select data file, not DB table.')
            return
        else:
            TableView(tk.Toplevel(self), geometry='1000x700', data=table['text'], table=table['tags'])

    def open_ml(self):
        table = self.tv_hier.item(self.tv_hier.selection())
        if table['text'] == '':
            print('No table selected. Please, select the table to continue.')
            return
        elif table['tags'][0] == 'table':
            print('Please, select data file, not DB table.')
            return
        else:
            MLView(data=table['text'], table=table['tags'])

    @staticmethod
    def get_tables_list(self):
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
        self.file_path_info = tk.StringVar()
        self.file_path_info.set('File path will be there')
        self.file_path = ''
        tk.Label(self, textvariable=self.file_path_info).pack(side=tk.TOP)
        self.file_open_button = tk.Button(self, text='File Upload',
                                          command=self.open_file).pack(anchor=tk.SE, expand=True)
        self.db_create_button = tk.Button(self, text='Add table to Tasks',
                                          command=self.input_table).pack(anchor=tk.SE)
        self.db_create_info = tk.StringVar()
        self.db_create_info.set('Please, enter table name:')
        tk.Label(self, textvariable=self.db_create_info).pack(anchor=tk.CENTER)
        self.table_name = tk.StringVar()
        self.table_name_entry = tk.Entry(self, textvariable=self.table_name).pack(anchor=tk.CENTER)

    def open_file(self):
        self.file_path = askopenfile(mode='r').name
        self.file_path_info.set('File path: ' + self.file_path)

    def input_table(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            name = self.table_name.get()
            if name == '':
                raise ValueError
            else:
                file = serialize(pd.read_csv(str(self.file_path)))
                cur.execute('INSERT INTO Tasks (name, table_file) VALUES (?, ?)', (name, file))
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
    def __init__(self, master, geometry, table, data, *args, **kwargs):
        self.master = master
        self.master.geometry(geometry)
        self.table = table[0]
        self.data = data
        tk.Label(self.master, text="{} table".format(self.table)).pack(side=tk.TOP)
        self.conn = sqlite3.connect('main.sqlite3')
        self.cur = self.conn.cursor()
        self.tv = ttk.Treeview(self.master, show='headings')
        self.show_table()

        # self.configuration_tv()
        # self.cols = self.get_cols()

    def show_table(self):
        # creating DB cursor and configuring table environment
        data = deserialize(self.cur.execute(f'SELECT table_file FROM {self.table} '
                                            f'WHERE name = "{self.data}"').fetchall()[0][0])
        self.tv['columns'] = list(data.columns)
        for column in self.tv['columns']:
            self.tv.heading(column, text=column)
            self.tv.column(column, width=60)
        rows = data.to_numpy().tolist()
        i = 0
        for row in rows:
            if i % 2 == 0:
                self.tv.insert('', tk.END, values=row, tag='even')
            else:
                self.tv.insert('', tk.END, values=row, tag='odd')
            i += 1
        self.tv.tag_configure('even', background='#E8E8E8')
        self.tv.tag_configure('odd', background='#DFDFDF')
        self.tv.place(relheight=1, relwidth=1)  # expand=True, fill=tk.BOTH)
        ysb = ttk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.tv.yview)
        xsb = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.tv.xview)
        self.tv.configure(yscroll=ysb.set, xscroll=xsb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
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

    # configuring scrollbar


class MLView(tk.Tk):
    def __init__(self, table, data, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('500x530')
        self.resizable(width=False, height=False)
        self.table = table[0]
        self.data = data
        self.dataframe = self.get_data()
        self.clf = 'classifier pattern'
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
                                        ).grid(row=2, column=1,sticky=tk.W)

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
                tk.Label(cur_frame,
                         text='Model accuracy: ' + str(out_self.clf.score(X_test, y_test))
                         ).grid(row=0, column=1)

        tk.Button(frame, text='Fit the model',
                  command=lambda: process_model(frame, self, target.get(), split.get())
                  ).grid(row=4, column=0)

    def get_data(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        try:
            query = f'SELECT table_file FROM {self.table} WHERE name = "{self.data}"'
            return deserialize(cur.execute(query).fetchall()[0][0])
        finally:
            conn.close()


def serialize(file):
    return pickle.dumps(file)


def deserialize(file):
    return pickle.loads(file)


if __name__ == '__main__':
    app = App()
    app.mainloop()
