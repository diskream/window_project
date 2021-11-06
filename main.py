import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
import pandas as pd
import sqlite3


class App:
    def __init__(self, master, geometry):
        self.master = master
        self.master.geometry(geometry)
        self.master.resizable(width=False, height=False)
        self.master.title('Brand New Window')

        # self.table_list_frame = tk.LabelFrame(self.master, text='Table list')
        self.table_list_frame = TablesListFrame(self.master, text='Table list')
        self.table_list_frame.place(x=0, y=0, anchor='nw', width=250, height=250)

        self.file_upload_frame = DbUploadFrame(self.master, text='Uploading a file to the database')
        self.file_upload_frame.place(x=0, y=250, anchor='nw', width=500, height=250)

        # self.table_list_frame.pack(anchor=tk.NW)
        self.to_tree_frame = tk.LabelFrame(self.master, text='Open a table')
        self.to_tree_frame.place(x=250, y=0, anchor='nw', width=250, height=250)
        self.table_open_button = tk.Button(self.to_tree_frame, text='Open a TreeView in the new window',
                                           command=self.open_table).pack(side=tk.BOTTOM)

    def open_table(self):
        new_window = TableView(tk.Toplevel(self.master), '500x500')


class DbUploadFrame(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
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
        tk.Label(self, textvariable=self.db_create_info).pack(anchor=tk.CENTER)
        # self.table_name_entry = tk.Entry

    def open_file(self):
        self.file_path = askopenfile(mode='r').name
        self.file_path_info.set('File path: ' + self.file_path)

    def create_table(self):
        conn = sqlite3.connect('main.sqlite3')
        try:
            pd.read_csv(str(self.file_path)).to_sql('mall', conn, if_exists='replace')
            self.db_create_info.set('Table created successfully')
        except FileNotFoundError:
            self.db_create_info.set('Upload the file first!')
        finally:
            conn.close()


class TablesListFrame(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.tables_list = self.get_tables_list()
        self.list_box = tk.Listbox(self)
        for table in self.tables_list:
            self.list_box.insert(tk.END, table)
        tk.Label(self, text='List of all database tables').pack(side=tk.TOP)
        self.list_box.pack(anchor=tk.CENTER)

    def get_tables_list(self):
        conn = sqlite3.connect('main.sqlite3')
        curs = conn.cursor()
        try:
            tables_list = curs.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchone()
            print(tables_list)
            return tables_list
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


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root, '500x500')
    root.mainloop()
