import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database connection. Can be initialized with a connection string or explicit params."""
    db_type: str = ""  # e.g., 'sqlite', 'postgresql', 'mysql'
    connection_string: Optional[str] = None  # For SQLite primarily
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    database: str = ""
    file_path: str = ""  # Specific for SQLite


class DatabaseConnector:
    """Handles database connections and queries based on a DatabaseConfig."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        # If connection string is provided, try to parse it first
        if self.config.connection_string:
            self._parse_connection_string()

        # If db_type is still not set (e.g., from explicit parameters), infer it
        if not self.config.db_type:
            if self.config.file_path:
                self.config.db_type = 'sqlite'
            elif self.config.host and self.config.username and self.config.database:
                # Assuming if host, user, db are present, it's a network DB like mysql/postgresql
                # This is a heuristic, better to explicitly pass db_type from UI
                logger.info(
                    "Inferred database type as network DB (MySQL/PostgreSQL likely) due to explicit host/user/db.")
                # For this environment, we'll default to 'mysql' for simulation if no type given
                self.config.db_type = 'mysql'
            else:
                self.config.db_type = 'unknown'

    def _parse_connection_string(self):
        """Parses the connection string for SQLite specifically."""
        if not self.config.connection_string:
            return

        try:
            # Simple check for sqlite connection string format
            if self.config.connection_string.startswith("sqlite:///"):
                self.config.db_type = 'sqlite'
                self.config.file_path = self.config.connection_string.replace("sqlite:///", "")
                # Clean up path for Windows compatibility if it starts with /X:/
                if len(self.config.file_path) > 3 and self.config.file_path[0] == '/' and self.config.file_path[
                    2] == ':':
                    self.config.file_path = self.config.file_path[1:]
            elif self.config.connection_string.startswith("sqlite:"):  # Handle sqlite:mydb.db
                self.config.db_type = 'sqlite'
                self.config.file_path = self.config.connection_string.replace("sqlite:", "")
            else:
                logger.warning(f"Connection string '{self.config.connection_string}' does not look like SQLite. "
                               "Please provide explicit host, username, password, database for other types or a valid connection string.")
                self.config.db_type = 'unknown_string_format'

        except Exception as e:
            logger.error(f"Failed to parse connection string '{self.config.connection_string}': {e}")
            self.config.db_type = "unknown"  # Mark as unknown on parsing failure

    def connect(self) -> bool:
        """Establishes database connection based on parsed config."""
        try:
            if self.config.db_type == 'sqlite':
                if not self.config.file_path:
                    logger.error("SQLite file path not specified.")
                    return False
                # Use check_same_thread=False for Streamlit's multi-threading environment
                self.connection = sqlite3.connect(self.config.file_path, check_same_thread=False)
                logger.info(f"Successfully connected to SQLite database at {self.config.file_path}")
                return True
            elif self.config.db_type == 'mysql':  # Simulate MySQL connection
                if not (self.config.host and self.config.username and self.config.database):
                    logger.error("MySQL connection requires host, username, and database.")
                    return False
                logger.info(
                    f"Simulating connection to MySQL database: {self.config.username}@{self.config.host}/{self.config.database}. "
                    "A real connection requires 'mysql-connector-python' which is not installed in this environment.")
                # We don't establish a real connection, just mark as successful for simulation
                self.connection = None
                return True
            elif self.config.db_type == 'postgresql':  # Simulate PostgreSQL connection
                if not (self.config.host and self.config.username and self.config.database):
                    logger.error("PostgreSQL connection requires host, username, and database.")
                    return False
                logger.info(
                    f"Simulating connection to PostgreSQL database: {self.config.username}@{self.config.host}/{self.config.database}. "
                    "A real connection requires 'psycopg2' which is not installed in this environment.")
                self.connection = None
                return True
            else:
                logger.error(f"Unsupported or unparseable database type: {self.config.db_type}")
                return False
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.connection = None
            return False

    def get_tables(self) -> List[str]:
        """Gets a list of tables in the database."""
        if self.config.db_type == 'sqlite':
            if not self.connection:
                logger.error("No SQLite connection established.")
                return []
            try:
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                return tables
            except Exception as e:
                logger.error(f"Failed to get tables from SQLite: {e}")
                return []
        elif self.config.db_type in ['mysql', 'postgresql']:
            logger.info(f"Simulating table fetch for {self.config.db_type}. (Live connection not established).")
            # For demonstration, return dummy tables appropriate for the use case
            return ["users", "products", "orders", "inventory_items", "roles"]  # Added roles and inventory_items
        else:
            return []

    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """Gets schema information for a table (column names and types)."""
        if self.config.db_type == 'sqlite':
            if not self.connection:
                logger.error("No SQLite connection established.")
                return []
            try:
                cursor = self.connection.cursor()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                # col[1] is name, col[2] is type
                return [{'name': col[1], 'type': col[2]} for col in columns]
            except Exception as e:
                logger.error(f"Failed to get schema for table '{table_name}' from SQLite: {e}")
                return []
        elif self.config.db_type in ['mysql', 'postgresql']:
            logger.info(f"Simulating schema fetch for {self.config.db_type}. (Live connection not established).")
            # For demonstration, return dummy schema based on table name
            if table_name == "users":
                return [{'name': 'id', 'type': 'INTEGER'}, {'name': 'username', 'type': 'TEXT'},
                        {'name': 'password', 'type': 'TEXT'}, {'name': 'email', 'type': 'TEXT'},
                        {'name': 'role_id', 'type': 'INTEGER', 'is_fk': True,
                         'references': 'roles.id'}]  # Added role_id and FK info
            elif table_name == "products":
                return [{'name': 'id', 'type': 'INTEGER'}, {'name': 'name', 'type': 'TEXT'},
                        {'name': 'price', 'type': 'REAL'}]
            elif table_name == "orders":
                return [{'name': 'order_id', 'type': 'INTEGER'},
                        {'name': 'user_id', 'type': 'INTEGER', 'is_fk': True, 'references': 'users.id'},
                        {'name': 'status', 'type': 'TEXT'}]
            elif table_name == "inventory_items":
                return [{'name': 'item_id', 'type': 'INTEGER'},
                        {'name': 'product_id', 'type': 'INTEGER', 'is_fk': True, 'references': 'products.id'},
                        {'name': 'quantity', 'type': 'INTEGER'}]
            elif table_name == "roles":  # New table for FK example
                return [{'name': 'id', 'type': 'INTEGER'}, {'name': 'role_name', 'type': 'TEXT'}]
            return []
        else:
            return []

    def preview_data(self, table_name: str, limit: int = 3) -> pd.DataFrame:
        """Previews data from a table, limited to a specified number of rows."""
        if self.config.db_type == 'sqlite':
            if not self.connection:
                logger.error("No SQLite connection established.")
                return pd.DataFrame()
            try:
                query = f"SELECT * FROM {table_name}"
                if limit is not None:
                    query += f" LIMIT {limit}"
                df = pd.read_sql_query(query, self.connection)
                return df
            except Exception as e:
                logger.error(f"Failed to preview data from table '{table_name}' from SQLite: {e}")
                return pd.DataFrame()
        elif self.config.db_type in ['mysql', 'postgresql']:
            logger.info(f"Simulating data preview for {self.config.db_type}. (Live connection not established).")
            # For demonstration, return dummy dataframes
            if table_name == "users":
                return pd.DataFrame({
                    'id': [1, 2, 3],
                    'username': ['user_A', 'user_B', 'user_C'],
                    'password': ['pass_A', 'pass_B', 'pass_C'],
                    'email': ['a@example.com', 'b@example.com', 'c@example.com'],
                    'role_id': [101, 102, 101]  # Dummy role IDs
                })
            elif table_name == "products":
                return pd.DataFrame({
                    'id': [101, 102, 103],
                    'name': ['Laptop', 'Mouse', 'Keyboard'],
                    'price': [1200.0, 25.0, 75.0]
                })
            elif table_name == "orders":
                return pd.DataFrame({
                    'order_id': [1001, 1002, 1003],
                    'user_id': [1, 2, 1],
                    'status': ['pending', 'completed', 'shipped']
                })
            elif table_name == "inventory_items":
                return pd.DataFrame({
                    'item_id': [2001, 2002, 2003],
                    'product_id': [101, 102, 101],
                    'quantity': [50, 120, 75]
                })
            elif table_name == "roles":
                return pd.DataFrame({
                    'id': [101, 102],
                    'role_name': ['Admin', 'User']
                })
            return pd.DataFrame()
        else:
            return pd.DataFrame()

