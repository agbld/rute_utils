#%%
import sqlite3
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import html
from multiprocessing import Pool
import time

#%%
class RutenItemNamesDataset(Dataset):
    def __init__(self, 
                 db_path,
                 table_name='ruten_items',
                 col_item_name='G_NAME',
                 create_db=False, 
                 path_to_ruten_items_folder=None, 
                 batch_size=1,  # for multi-processing, not working yet
                 num_workers=1, # for multi-processing, not working yet
                 top_n=None, 
                 verbose=False):
        self._db_path = db_path
        self._db_name = os.path.basename(self._db_path)
        self._table_name = table_name
        self._col_item_name = col_item_name
        self._path_to_ruten_items_folder = path_to_ruten_items_folder
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._top_n = top_n
        self._verbose = verbose
        '''
        Please notice that if the database already exists, setting create_db to True WILL remove it and re-create a new one.
        '''

        if create_db or not os.path.exists(self._db_path):
            # Remove the existing database if it exists
            if os.path.exists(self._db_path):   
                os.remove(self._db_path)

            # Create a new sqlite .db file
            self._connection = sqlite3.connect(self._db_path) 
            self._cursor = self._connection.cursor()

            # Create the table and insert data into it
            self._create_db()
        else:
            # Connect to the SQLite database
            self._connection = sqlite3.connect(self._db_path)
            self._cursor = self._connection.cursor()
        
        # Get the number of rows in the table
        # TODO: create a tabel for the number of rows, or other metadata.
        self._print(f'Calculating the number of rows in the table {self._table_name}...')
        start = time.time()
        self._num_rows = self._cursor.execute(f'SELECT COUNT(*) FROM {self._table_name}').fetchone()[0]
        self._print(f'Number of rows: {self._num_rows} (took {time.time() - start:.2f} seconds)')

    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, index) -> str:
        index += 1  # SQLite starts from 1
        select_query = f"SELECT * FROM {self._table_name} WHERE id = ?"
        self._cursor.execute(select_query, (index,))
        row = self._cursor.fetchone()[1]
        if row is None:
            raise IndexError(f'Index {index} is out of range.')
        row = html.unescape(row) # unescape html entities, e.g. &amp; -> &.

        return row  # return a tuple: (id, G_NAME)

    def _print(self, msg):
        if self._verbose:
            print(f'[RutenItemNamesDataset] {msg}')

    def _create_db(self):   # slow as fuck, took 30+ min to load 250M rows.
        self._print('Creating database... (this may take a while)')

        # Function to read parquet files and insert data into the SQLite database
        def process_parquet_file(file_path):
            df = pd.read_parquet(file_path, columns=[self._col_item_name])
            self._cursor.executemany(f"INSERT INTO {self._table_name} ({self._col_item_name}) VALUES (?)", 
                               df[[self._col_item_name]].values)

        create_table_query = f"""
        CREATE TABLE {self._table_name} (
            id INTEGER PRIMARY KEY,
            G_NAME TEXT
        );
        """
        self._cursor.execute(create_table_query)
        self._connection.commit()

        # Get a list of all parquet files in the current directory
        parquet_files = []
        for filename in os.listdir(self._path_to_ruten_items_folder):
            if filename.endswith('.parquet'):
                parquet_files.append(os.path.join(self._path_to_ruten_items_folder, filename))
        parquet_files = parquet_files[:self._top_n]

        # Process each parquet file and insert data into the table
        with tqdm(total=len(parquet_files), desc="Processing parquet files", disable=not self._verbose) as pbar:
            for idx, file in enumerate(parquet_files):
                process_parquet_file(file)
                self._connection.commit()
                pbar.update(1)

        self._print("All parquet data has been loaded into the SQLite database.")

    def _create_db_multi(self):  # TODO: multi-processing version, not working yet
        self._print('Loading parquets into database... (this may take a while)')
        parquet_files = []
        for filename in os.listdir(self._path_to_ruten_items_folder):
            if filename.endswith('.parquet'):
                parquet_files.append(os.path.join(self._path_to_ruten_items_folder, filename))
        parquet_files = parquet_files[:self._top_n]
        
        parquet_files_batches = []
        for i in range(0, len(parquet_files), self._batch_size):
            parquet_files_batches.append(parquet_files[i:i+self._batch_size])

        def insert_batch(paths):
            for path in paths:
                df = pd.read_parquet(path, columns=[self._col_item_name])
                df.to_sql(self._table_name, self._connection, if_exists='append', index=False)

        with Pool(processes=self._num_workers) as pool:
            tmp = list(tqdm(pool.imap(insert_batch, parquet_files_batches), total=len(parquet_files_batches), desc="Inserting Batches", disable=not self._verbose))

        # with Pool(processes=self._num_workers) as pool:
        #     with tqdm(total=len(parquet_files_batches), desc="Inserting Batches", disable=not self._verbose) as pbar:
        #         for _ in pool.imap(insert_batch, parquet_files_batches):
        #             pbar.update(1)
        self._print("All parquet data has been loaded into the SQLite database.")

#%%
# Test the dataset
if __name__ == '__main__':
    dataset = RutenItemNamesDataset(db_path='ruten.db',
                                    table_name='ruten_items',
                                    col_item_name='G_NAME',
                                    create_db=False,
                                    path_to_ruten_items_folder='/mnt/share_disk/Datasets/Ruten/item/activate_item/',
                                    batch_size=5,
                                    num_workers=8,
                                    top_n=10,
                                    verbose=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    for batch in dataloader:
        for item in batch:
            print(item)
        input()


#%%