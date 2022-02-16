import sqlite3


class Task:
    def __init__(self, task_id: int, name: str, table_file=None):
        self.task_id = task_id
        self.name = name
        self.table = 'Tasks'
        self.columns = ['task_id', 'name', 'table_file']
        if table_file:
            self.table_file = table_file
        else:
            self.table_file = None

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def get_name(self, id):
        if id == self.task_id:
            return self.name
        else:
            return False


class Variant:
    def __init__(self, task_id: int, variant_id: int, name: str, table_file=None):
        self.task_id = task_id
        self.variant_id = variant_id
        self.name = name,
        self.parent_name = get_parent(self.task_id, self.variant_id)
        self.table = 'Task_variant'
        self.columns = ['task_id', 'variant_id', 'name', 'table_file']
        if table_file:
            self.table_file = table_file
        else:
            self.table_file = None


    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class Model:
    def __init__(self, model_id: int, task_id: int, variant_id: int, name: str, acc: float, bin_file=None):
        self.model_id = model_id
        self.task_id = task_id
        self.variant_id = variant_id
        self.name = name
        self.acc = acc
        self.table = 'Models'
        if bin_file:
            self.bin = bin_file
        else:
            self.bin = None

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


def get_parent(task_id, variant_id):
    conn = sqlite3.connect('main.sqlite3')
    cur = conn.cursor()
    try:
        sql = 'SELECT t.name FROM Tasks as t INNER JOIN Task_variant Tv on t.task_id = Tv.task_id ' \
              f'WHERE t.task_id = {task_id} AND Tv.variant_id = {variant_id}'
        return cur.execute(sql).fetchone()[0]
    finally:
        conn.close()
