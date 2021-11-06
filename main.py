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
        self.file_upload_frame.place(x=0, y=250, anchor='nw', width=500, height=250)

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
        self.tables_list = self.get_tables_list()
        self.list_box = tk.Listbox(self)
        self.tables_list = sorted([table[0] for table in self.tables_list])
        for index, table in enumerate(self.tables_list):
            self.list_box.insert(index, table)
            print(index, table)
        tk.Label(self, text='List of all database tables').pack(side=tk.TOP)
        self.list_box.pack(anchor=tk.CENTER)

        self.ml_window_button = tk.Button(self, text='Open ML window', command=self.open_ml).pack(side=tk.BOTTOM)
        self.table_open_button = tk.Button(self, text='Open a TreeView in the new window',
                                           command=self.open_table).pack(side=tk.BOTTOM)

    def open_table(self):
        new_window = TableView(tk.Toplevel(self), geometry='500x500')

    def open_ml(self):
        if self.list_box.get(tk.ANCHOR) == '':
            pass
        else:
            print(self.list_box.get(tk.ANCHOR), '123')
            MLView(table=self.list_box.get(tk.ANCHOR))

    def get_tables_list(self):
        conn = sqlite3.connect('main.sqlite3')
        curs = conn.cursor()
        try:
            tables_list = curs.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()
            # print(tables_list)
            return tables_list
        finally:
            conn.close()


class BottomFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.file_path_info = tk.StringVar()
        self.file_path_info.set('File path will be there')
        self.file_path = ''
        tk.Label(self, textvariable=self.file_path_info).pack(side=tk.TOP)
        self.file_open_button = tk.Button(self, text='File Upload',
                                          command=self.open_file).pack(anchor=tk.SE, expand=True)
        self.db_create_button = tk.Button(self, text='Create DB from File',
                                          command=self.create_table).pack(anchor=tk.SE)
        self.db_create_info = tk.StringVar()
        self.db_create_info.set('Please, enter table name:')
        tk.Label(self, textvariable=self.db_create_info).pack(anchor=tk.CENTER)
        self.table_name = tk.StringVar()
        self.table_name_entry = tk.Entry(self, textvariable=self.table_name).pack(anchor=tk.CENTER)

    def open_file(self):
        self.file_path = askopenfile(mode='r').name
        self.file_path_info.set('File path: ' + self.file_path)

    def create_table(self):
        conn = sqlite3.connect('main.sqlite3')
        try:
            name = self.table_name.get()
            if name is None:
                raise ValueError
            else:
                pd.read_csv(str(self.file_path)).to_sql(name, conn, if_exists='replace')
                self.db_create_info.set('Table created successfully')
                App.update_list(self)
        except ValueError:
            self.db_create_info.set('Enter table name first!')
        except FileNotFoundError:
            self.db_create_info.set('Upload the file first!')
        finally:
            conn.close()


class TableView:
    def __init__(self, master, geometry, *args, **kwargs):
        self.master = master
        self.master.geometry(geometry)
        tk.Label(self.master, text="I'm in TableView Frame now!").pack(side=tk.TOP)
        self.get_cols_button = tk.Button(self.master, text='Get columns from mall',
                                         command=self.show_table).pack(side=tk.TOP)
        self.conn = sqlite3.connect('main.sqlite3')
        self.cur = self.conn.cursor()
        self.cols = self.get_cols()
        self.tv = ttk.Treeview(self.master, columns=self.cols, show='headings')
        self.configuration_tv()

    def show_table(self):
        self.cur.execute('SELECT * FROM mall')
        rows = self.cur.fetchall()
        for row in rows:
            self.tv.insert('', tk.END, values=row)
        self.tv.pack(expand=True, fill=tk.BOTH)
        ysb = ttk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.tv.yview)
        self.tv.configure(yscroll=ysb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

    def get_cols(self):
        self.cur.execute('SELECT * FROM mall WHERE 1=0')
        return [d[0] for d in self.cur.description]

    def configuration_tv(self):
        # configuring headings
        n, m = 1, 0
        for col in self.cols:
            self.tv.heading('#{}'.format(n), text=col)
            self.tv.column(m, width=80)
            n += 1
            m += 1

        # configuring scrollbar


class MLView(tk.Tk):
    def __init__(self, table, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('500x500')
        self.table = table
        self.data = ''
        tk.Label(self, text='Chosen table: ' + self.table).pack()
        self.get_pandas_button = tk.Button(self, text='Get data from database', command=self.get_pandas).pack()
        self.train_button = tk.Button(self, text='Fit model', command=self.fit_model)
        self.X_test = ''
        self.y_test = ''
        self.to_db_button = ''
        self.predict_button = tk.Button(self, text='Predict class from X_test', command=self.prediction).\
            pack()

    def get_pandas(self):
        conn = sqlite3.connect('main.sqlite3')
        query = 'SELECT * FROM {}'.format(self.table)
        self.data = pd.read_sql(query, conn, index_col='index')
        print(type(self.data))
        tk.Label(self, text=self.data.head(5)).pack()
        self.train_button.pack()

    def fit_model(self):
        data = self.data
        data = data.rename(columns={'Annual Income (k$)': 'income', 'Spending Score (1-100)': 'spending_score'})
        data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
        X = data.drop('spending_score', axis=1)
        y = data.spending_score
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.25)
        clf = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
        tk.Label(self, text='Model score: ' + str(clf.score(self.X_test, self.y_test))).pack()
        model = pickle.dumps(clf)
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        cur.execute('INSERT INTO models VALUES (?, ?)', ('mall', model))
        # print(cur.execute('SELECT * FROM models').fetchall())
        tk.Label(self, text='Model successfully fitted and imported to models table').pack()
        conn.close()

    def prediction(self):
        conn = sqlite3.connect('main.sqlite3')
        cur = conn.cursor()
        clf = cur.execute('SELECT model FROM models').fetchone()
        print(clf, type(clf))
        clf = pickle.loads(clf[0])
        prediction = clf.predict(self.X_test)
        tk.Label(self, text=str(prediction)).pack()
        conn.close()




if __name__ == '__main__':
    app = App()
    app.mainloop()
