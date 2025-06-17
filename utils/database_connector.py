import sqlite3
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import os

# Import pyodbc for SQL Server connectivity
try:
    import pyodbc
except ImportError:
    pyodbc = None
    logging.warning(
        "pyodbc not found. SQL Server connection will be simulated. Please install it (`pip install pyodbc`).")

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration details for connecting to different database types."""
    db_type: str  # e.g., "sqlite", "mysql", "postgresql", "sqlserver"
    file_path: Optional[str] = None  # For SQLite
    host: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    port: Optional[int] = None


class DatabaseConnector:
    """
    Handles connections and data retrieval from various database types.
    Includes logic to extract detailed table schemas, primary keys, and foreign keys.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None  # Database connection object

    def connect(self) -> bool:
        """Establishes a database connection based on the configured type."""
        try:
            if self.config.db_type == "sqlite":
                # Ensure directory exists for SQLite DB file
                os.makedirs(os.path.dirname(self.config.file_path), exist_ok=True)
                self.conn = sqlite3.connect(self.config.file_path)
            elif self.config.db_type == "sqlserver":
                if pyodbc is None:
                    logger.error("pyodbc is not installed. Cannot connect to SQL Server.")
                    return False

                # Construct connection string for SQL Server
                conn_str_parts = [
                    "DRIVER={ODBC Driver 17 for SQL Server}",  # Common driver, adjust if needed
                    f"SERVER={self.config.host}",
                    f"DATABASE={self.config.database}",
                    f"UID={self.config.username}",
                    f"PWD={self.config.password}"
                ]
                if self.config.port:
                    conn_str_parts[1] = f"SERVER={self.config.host},{self.config.port}"  # Add port if specified

                conn_str = ";".join(conn_str_parts)
                logger.info(
                    f"Attempting SQL Server connection with: {conn_str_parts[0]} SERVER={self.config.host}, DATABASE={self.config.database}, UID={self.config.username} (password hidden)")
                self.conn = pyodbc.connect(conn_str)

            elif self.config.db_type in ["mysql", "postgresql"]:
                # Keep simulation for other external DBs as they might require different drivers
                logger.warning(
                    f"Simulating {self.config.db_type.upper()} connection. Real connection requires specific drivers.")
                self.conn = f"DUMMY_{self.config.db_type.upper()}_CONN"  # Simulate connection
            else:
                logger.error(f"Unsupported database type: {self.config.db_type}")
                return False

            logger.info(f"Successfully connected to {self.config.db_type}.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.db_type}: {e}")
            self.conn = None
            return False

    def disconnect(self):
        """Closes the database connection if it exists."""
        if self.conn:
            try:
                if self.config.db_type in ["sqlite", "sqlserver"]:
                    self.conn.close()
                logger.info(f"Disconnected from {self.config.db_type}.")
            except Exception as e:
                logger.error(f"Error disconnecting from {self.config.db_type}: {e}")
            finally:
                self.conn = None

    def get_tables(self) -> List[str]:
        """Retrieves a list of table names in the connected database."""
        if not self.conn:
            logger.error("No database connection established.")
            return []
        try:
            cursor = self.conn.cursor()
            if self.config.db_type == "sqlite":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            elif self.config.db_type == "sqlserver":
                # Fetch tables from INFORMATION_SCHEMA.TABLES for SQL Server
                cursor.execute(
                    "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo';")  # Assuming 'dbo' schema
            elif self.config.db_type in ["mysql", "postgresql"]:
                # Simulated table names for other non-SQLite connections
                return ["pets", "users", "orders", "inventory_items", "roles"]  # Dummy data for simulation

            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tables for {self.config.db_type}: {e}")
            return []

    def get_primary_keys(self, table_name: str) -> List[str]:
        """
        Returns a list of primary key column names for a given table.
        This uses database-specific PRAGMA or information schema queries.
        """
        if not self.conn:
            logger.error("No database connection established.")
            return []
        try:
            cursor = self.conn.cursor()
            if self.config.db_type == "sqlite":
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                pk_columns = [row[1] for row in cursor.fetchall() if row[5] == 1]
                return pk_columns
            elif self.config.db_type == "sqlserver":
                query = f"""
                SELECT KCU.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KCU
                    ON TC.CONSTRAINT_NAME = KCU.CONSTRAINT_NAME
                WHERE TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND KCU.TABLE_NAME = '{table_name}'
                AND KCU.TABLE_SCHEMA = 'dbo';
                """
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
            elif self.config.db_type in ["mysql", "postgresql"]:
                # Simulated PKs for dummy data
                if table_name == "pets": return ["id"]
                if table_name == "users": return ["id"]
                if table_name == "orders": return ["order_id"]
                if table_name == "inventory_items": return ["item_id"]
                if table_name == "roles": return ["id"]
                return []
            return []
        except Exception as e:
            logger.error(f"Error getting primary keys for table {table_name}: {e}")
            return []

    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Returns a list of foreign key details for a given table.
        Each dictionary contains 'column', 'ref_table', and 'ref_column'.
        """
        if not self.conn:
            logger.error("No database connection established.")
            return []
        try:
            cursor = self.conn.cursor()
            if self.config.db_type == "sqlite":
                cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
                fks = []
                for row in cursor.fetchall():
                    fks.append({
                        "column": row[3],  # Child column name in current table (from)
                        "ref_table": row[2],  # Parent table name (table)
                        "ref_column": row[4]  # Parent column name in parent table (to)
                    })
                return fks
            elif self.config.db_type == "sqlserver":
                query = f"""
                SELECT
                    FK.COLUMN_NAME AS FkColumn,
                    PK.TABLE_NAME AS PkTableName,
                    PK.COLUMN_NAME AS PkColumnName
                FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS AS RC
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS FK
                    ON RC.CONSTRAINT_NAME = FK.CONSTRAINT_NAME
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS PK
                    ON RC.UNIQUE_CONSTRAINT_NAME = PK.CONSTRAINT_NAME
                WHERE FK.TABLE_NAME = '{table_name}'
                AND FK.TABLE_SCHEMA = 'dbo';
                """
                cursor.execute(query)
                fks = []
                for row in cursor.fetchall():
                    fks.append({
                        "column": row[0],
                        "ref_table": row[1],
                        "ref_column": row[2]
                    })
                return fks
            elif self.config.db_type in ["mysql", "postgresql"]:
                # Simulated FKs for dummy data
                if table_name == "users":
                    return [{"column": "role_id", "ref_table": "roles", "ref_column": "id"}]
                if table_name == "orders":
                    return [{"column": "user_id", "ref_table": "users", "ref_column": "id"}]
                if table_name == "inventory_items":
                    return [{"column": "product_id", "ref_table": "pets", "ref_column": "id"}]
                return []
            return []
        except Exception as e:
            logger.error(f"Error getting foreign keys for table {table_name}: {e}")
            return []

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Returns detailed schema for a table, including column names, types,
        nullability, primary key status, and foreign key references.
        """
        if not self.conn:
            logger.error("No database connection established.")
            return []
        try:
            schema = []
            pk_columns = self.get_primary_keys(table_name)
            fk_constraints = self.get_foreign_keys(table_name)

            cursor = self.conn.cursor()
            if self.config.db_type == "sqlite":
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns_info = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

                for col_info in columns_info:
                    column_name = col_info[1]
                    data_type = col_info[2]
                    is_nullable = "NO" if col_info[3] == 1 else "YES"  # SQLite: notnull is 1 for NOT NULL
                    is_pk = column_name in pk_columns

                    fk_details = next((fk for fk in fk_constraints if fk["column"] == column_name), None)
                    is_fk = bool(fk_details)
                    references_info = fk_details if is_fk else None

                    schema.append({
                        "name": column_name,
                        "type": data_type,
                        "nullable": is_nullable,
                        "is_primary_key": is_pk,
                        "is_foreign_key": is_fk,
                        "references": references_info
                    })
            elif self.config.db_type == "sqlserver":
                # Query INFORMATION_SCHEMA.COLUMNS for column details
                column_query = f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                AND TABLE_SCHEMA = 'dbo';
                """
                cursor.execute(column_query)
                columns_info = cursor.fetchall()

                for col_info in columns_info:
                    column_name = col_info[0]
                    data_type = col_info[1]
                    is_nullable = col_info[2]  # 'YES' or 'NO'

                    is_pk = column_name in pk_columns

                    fk_details = next((fk for fk in fk_constraints if fk["column"] == column_name), None)
                    is_fk = bool(fk_details)
                    references_info = fk_details if is_fk else None

                    schema.append({
                        "name": column_name,
                        "type": data_type,
                        "nullable": is_nullable,
                        "is_primary_key": is_pk,
                        "is_foreign_key": is_fk,
                        "references": references_info
                    })
            elif self.config.db_type in ["mysql", "postgresql"]:
                # Return simulated schema with PK/FK details for other DB types
                if table_name == "pets":
                    schema = [
                        {"name": "id", "type": "INTEGER", "nullable": "NO", "is_primary_key": True,
                         "is_foreign_key": False, "references": None},
                        {"name": "name", "type": "TEXT", "nullable": "NO", "is_primary_key": False,
                         "is_foreign_key": False, "references": None},
                        {"name": "status", "type": "TEXT", "nullable": "NO", "is_primary_key": False,
                         "is_foreign_key": False, "references": None},
                        {"name": "tags", "type": "TEXT", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": False, "references": None}
                    ]
                elif table_name == "users":
                    schema = [
                        {"name": "id", "type": "INTEGER", "nullable": "NO", "is_primary_key": True,
                         "is_foreign_key": False, "references": None},
                        {"name": "username", "type": "TEXT", "nullable": "NO", "is_primary_key": False,
                         "is_foreign_key": False, "references": None},
                        {"name": "password", "type": "TEXT", "nullable": "NO", "is_primary_key": False,
                         "is_foreign_key": False, "references": None},
                        {"name": "email", "type": "TEXT", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": False, "references": None},
                        {"name": "role_id", "type": "INTEGER", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": True, "references": {"ref_table": "roles", "ref_column": "id"}}
                    ]
                elif table_name == "orders":
                    schema = [
                        {"name": "order_id", "type": "INTEGER", "nullable": "NO", "is_primary_key": True,
                         "is_foreign_key": False, "references": None},
                        {"name": "user_id", "type": "INTEGER", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": True, "references": {"ref_table": "users", "ref_column": "id"}},
                        {"name": "status", "type": "TEXT", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": False, "references": None}
                    ]
                elif table_name == "inventory_items":
                    schema = [
                        {"name": "item_id", "type": "INTEGER", "nullable": "NO", "is_primary_key": True,
                         "is_foreign_key": False, "references": None},
                        {"name": "product_id", "type": "INTEGER", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": True, "references": {"ref_table": "pets", "ref_column": "id"}},
                        {"name": "quantity", "type": "INTEGER", "nullable": "YES", "is_primary_key": False,
                         "is_foreign_key": False, "references": None}
                    ]
                elif table_name == "roles":
                    schema = [
                        {"name": "id", "type": "INTEGER", "nullable": "NO", "is_primary_key": True,
                         "is_foreign_key": False, "references": None},
                        {"name": "role_name", "type": "TEXT", "nullable": "NO", "is_primary_key": False,
                         "is_foreign_key": False, "references": None}
                    ]
                else:
                    return []
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return []

    def preview_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Retrieves a limited number of rows from a specified table for preview."""
        if not self.conn:
            logger.error("No database connection established.")
            return pd.DataFrame()
        try:
            query = f"SELECT TOP {limit} * FROM [{table_name}];" if self.config.db_type == "sqlserver" else f"SELECT * FROM '{table_name}' LIMIT {limit};"
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            logger.error(f"Error previewing data for table {table_name}: {e}")
            return pd.DataFrame()

