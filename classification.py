import tkinter as tk
from tkinter import ttk
import sqlite3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from functions import serialize, deserialize, update_entry


class MLView(tk.Tk):
    def __init__(self, entry):
        tk.Tk.__init__(self)
        self.geometry('500x500')
        self.entry = entry
        update_entry(self.entry)
        self.pd_data = deserialize(self.entry.table_file)
        self.upper_frm = tk.LabelFrame(self, text='Выбор алгоритма')
        self.upper_frm.pack(side=tk.TOP, fill=tk.X)
        self.alg_cb = ttk.Combobox(self.upper_frm, values=['Дерево реешений',
                                                           'Случайный лес',
                                                           'k Ближайших соседей'],
                                   )
        self.alg_cb.current(0)
        self.alg_cb.pack(side=tk.TOP, pady=10)
        tk.Button(self.upper_frm, text='Выбрать', command=self.get_alg).pack(side=tk.TOP, pady=5)
        self.update_title()
        self.dt_frm = DecisionTreeFrame(self)
        self.rf_frm = RandomForestFrame(self)
        self.knn_frm = KNeighborsFrame(self)
        self.algs = {
            'Дерево реешений': self.dt_frm,
            'Случайный лес': self.rf_frm,
            'k Ближайших соседей': self.knn_frm
        }
        self.get_alg()

    def update_title(self):
        self.title(f'Работа с {self.entry.name} с помощью алгоритма "{self.alg_cb.get()}"')

    def get_alg(self):
        alg = self.alg_cb.get()
        self.title(f'Работа с {self.entry.name} с помощью алгоритма "{alg}"')
        self.algs[alg].pack(side=tk.TOP, fill=tk.BOTH, expand=1)


class DecisionTreeFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text='Decision Tree').pack()


class RandomForestFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text='Random Forest').pack()


class KNeighborsFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        tk.Label(self, text='k Nearest Neighbors').pack()
        self.clf = DecisionTreeClassifier


class MLViewOld(tk.Tk):
    def __init__(self, table, data):
        tk.Tk.__init__(self)
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
                cur.execute('INSERT INTO Models (model_id, variant_id, task_id, name, bin_file, accuracy) '
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
