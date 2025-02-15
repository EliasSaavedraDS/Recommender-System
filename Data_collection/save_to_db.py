import sqlite3
import pandas as pd

csv_files = ['Data_collection/DallasDataset.csv','Data_collection/Houston&AustinDataset.csv']

df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

with sqlite3.connect('Sqlite.db') as conn:
    cursor = conn.cursor()

    #Create a table and csv for the main or full dataset
    df.to_sql('Fulldataset',conn,if_exists='replace',index=False)
    df.to_csv('FullDataset.csv',index=False)

    #Create a table and csv for numerical and categorical feature types (exluding the description feature)
    df_without_description = df.drop(columns=['Description']).copy()
    df_without_description.to_sql('AttributesTable', conn, if_exists='replace', index=False)

    #Create a table for the Description column
    cursor.execute("DROP TABLE IF EXISTS DescriptionTable")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DescriptionTable AS
        SELECT "Post ID", Description
        FROM Fulldataset
    """)
    conn.commit()

#Note: Sqlite Database and FullDataset.csv are stored in the notebook/data directory
print("Data has been successfully integrated into SQLite and saved")