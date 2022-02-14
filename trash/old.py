# main_window_frames
def insert_tv_old(self):
    # Сделать иерархию по составному ключу: 1-1; 1-2 иерархически отобразить как 1: 1, 2

    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        n = 0
        tables_list = cur.execute('SELECT name FROM sqlite_master WHERE type = "table";').fetchall()
        if len(tables_list) >= 4:
            tables_list = tables_list[1:]
        for table in tables_list:
            self.tv_hier.insert('', '0', f'item{n}', text=table, tags='table')
            names = cur.execute('SELECT name FROM {}'.format(table[0])).fetchall()
            if table[0] != 'Task_variant':  # Для обычных случаев
                for name in names:
                    self.tv_hier.insert(f'item{n}', tk.END, text=name, tags=f'{table[0]}')
                if table[0] == 'Tasks':
                    _sql = 'SELECT task_id, name FROM TASKS'
                    query = cur.execute(_sql).fetchall()
                    _temp_lst = []
                    for i in query:
                        _temp_lst.append(Task(i[0], i[1]))
                    self.db['Tasks'] = _temp_lst
                    del _temp_lst
                    print(self.db)
                elif table[0] == 'Models':
                    _sql = "SELECT model_id, variant_id, task_id, name FROM Models"
                    query = cur.execute(_sql).fetchall()
                    print('Models: ', query)
            else:  # Для расширенной иерархии вариантов заданий
                _sql = 'SELECT tv.task_id, tv.variant_id, tv.name, t.name FROM Task_variant as tv ' + \
                       'INNER JOIN Tasks as t ON tv.task_id = t.task_id'
                query = cur.execute(_sql).fetchall()
                print('Task_variant: ', query)
                hierarchy_dict = {}  # Словарь для удобной запаковки данных в treeview
                for variant in query:  # Заполняем словарь Task: информация из Task_variant
                    _temp_dict = {
                        'task_id': variant[0],
                        'variant_id': variant[1],
                        'variant_name': variant[2]
                    }
                    if variant[-1] not in hierarchy_dict.keys():
                        hierarchy_dict[variant[-1]] = [_temp_dict]
                    else:
                        hierarchy_dict[variant[-1]].append(_temp_dict)
                for task, info in hierarchy_dict.items():
                    self.tv_hier.insert(f'item{n}', '1', f'it{n + 1}', text=task)
                    for info1 in info:
                        self.tv_hier.insert(f'it{n + 1}', tk.END, text=info1['variant_name'], tags=f'{table[0]}')

            n += 1
    finally:
        conn.close()