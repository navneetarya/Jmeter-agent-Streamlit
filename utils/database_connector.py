import sqlite3
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    db_type: str
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    database: str = ""
    file_path: str = ""


class DatabaseConnector:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None

    def connect(self):
        try:
            if self.config.db_type.lower() == 'sqlite':
                self.connection = sqlite3.connect(self.config.file_path)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def get_tables(self) -> List[str]:
        if not self.connection:
            return []

        try:
            if self.config.db_type.lower() == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                return tables
        except Exception as e:
            logger.error(f"Failed to get tables: {e}")
            return []

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        if not self.connection:
            return []

        try:
            if self.config.db_type.lower() == 'sqlite':
                cursor = self.connection.cursor()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                return [{'name': col[1], 'type': col[2]} for col in columns]
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return []

    def preview_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        if not self.connection:
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            logger.error(f"Failed to preview data: {e}")
            return pd.DataFrame()