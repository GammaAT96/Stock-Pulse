from etl.ingestion.csv_ingestion import ingest_csv_files
from etl.storage.database_writer import insert_to_bronze

def run_pipeline():
    print("Starting ETL Pipeline...")

    dfs = ingest_csv_files()

    for df in dfs:
        insert_to_bronze(df)

    print("ETL Completed")


if __name__ == "__main__":
    run_pipeline()