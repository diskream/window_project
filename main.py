import tkinter as tk
from tkinter import ttk
from main_window_frames import TopFrame, BottomFrame


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


if __name__ == '__main__':
    app = App()
    print(ttk.Style().theme_names())
    app.mainloop()
