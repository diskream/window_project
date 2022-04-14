import tkinter as tk
from tkinter import ttk
from functional_views.main_window_frames import TopFrame, BottomFrame


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.w, self.h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.HEIGHT = self.h // 2
        self.WIDTH = (7 * self.HEIGHT) // 9
        self.geometry(f'{self.WIDTH}x{self.HEIGHT}')
        self.title('Дипломный проект v0.068a')

        self.table_list_frame = TopFrame(self)
        self.table_list_frame.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

        self.file_upload_frame = BottomFrame(self)
        self.file_upload_frame.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=True, padx=5, pady=5)

    def update_list(self):
        self.table_list_frame = TopFrame(self, text='Table list')
        self.table_list_frame.place(x=0, y=0, anchor='nw', width=250, height=250)


if __name__ == '__main__':
    app = App()
    # app.tk.call('source', r'tools/Sun-Valley-ttk-theme-master/sun-valley.tcl')
    # app.tk.call('set_theme', 'light')
    app.mainloop()
