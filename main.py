import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile


class App(tk.Tk):
    def __init__(self,*args, **kwargs):
        super(App, self).__init__(*args, **kwargs)
        # self.table_frame = TableFrame(self)
        # self.db_table_creation = DatabaseTableCreation(self)
        self.file_path = ''
        self.button = tk.Button(self, text='Open', command=self.open_file())
        self.button.pack()
    def open_file(self):
        self.file_path = askopenfile(mode='r')
        print(str(self.file_path.name))


class TableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(self, parent, *args, **kwargs)


class DatabaseTableCreation(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(self, parent, *args, **kwargs)



if __name__ == '__main__':
    root = tk.Tk()
    app = App()
    app.mainloop()
