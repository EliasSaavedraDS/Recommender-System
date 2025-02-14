import sqlite3
import pandas as pd

csv_files = ['Data_collection/DallasDataset.csv','Data_collection/Houston&AustinDataset.csv']

df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

conn = sqlite3.connect('Sqlite.db')

df.to_sql('Fulldataset',conn,if_exists='replace',index=False)


df.to_csv('FullDataset.csv',index=False)

conn.close()

print("Data has been successfully integrated into SQLite and saved as FullDataset.csv!")