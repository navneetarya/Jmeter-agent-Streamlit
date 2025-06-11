import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from swagger_parser import SwaggerParser, SwaggerEndpoint
from database_connector import DatabaseConnector, DatabaseConfig
from data_mapper import DataMapper

# Rest of your app code...

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    db_type: str
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    database: str = ""
    file_path: str = ""  # For SQLite


@dataclass
class SwaggerEndpoint:
    path: str
    method: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]


class SwaggerParser:
    """Parse Swagger/OpenAPI specifications"""

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_spec = None

    def fetch_swagger_spec(self) -> Dict[str, Any]:
        """Fetch and parse Swagger specification"""
        try:
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()
            self.swagger_spec = response.json()
            return self.swagger_spec
        except Exception as e:
            logger.error(f"Failed to fetch Swagger spec: {e}")
            raise

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        """Extract API endpoints from Swagger spec"""
        if not self.swagger_spec:
            self.fetch_swagger_spec()

        endpoints = []
        paths = self.swagger_spec.get('paths', {})

        for path, path_data in paths.items():
            for method, method_data in path_data.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    parameters = method_data.get('parameters', [])
                    responses = method_data.get('responses', {})

                    endpoints.append(SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        parameters=parameters,
                        responses=responses
                    ))

        return endpoints


class DatabaseConnector:
    """Handle database connections and queries"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None

    def connect(self):
        """Establish database connection"""
        try:
            if self.config.db_type.lower() == 'sqlite':
                self.connection = sqlite3.connect(self.config.file_path)
            # Add other database types as needed
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
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
        """Get schema information for a table"""
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
        """Preview data from a table"""
        if not self.connection:
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            logger.error(f"Failed to preview data: {e}")
            return pd.DataFrame()


class DataMapper:
    """Map database fields to API parameters using AI suggestions"""

    @staticmethod
    def suggest_mappings(endpoints: List[SwaggerEndpoint], tables_schema: Dict[str, List[Dict[str, str]]]) -> Dict[
        str, Dict[str, str]]:
        """Suggest mappings between API parameters and database fields"""
        mappings = {}

        for endpoint in endpoints:
            endpoint_key = f"{endpoint.method} {endpoint.path}"
            mappings[endpoint_key] = {}

            # Extract parameters that need data
            for param in endpoint.parameters:
                param_name = param.get('name', '')
                param_type = param.get('type', param.get('schema', {}).get('type', ''))

                # Simple AI-like matching based on name similarity and type
                best_match = DataMapper._find_best_match(param_name, param_type, tables_schema)
                if best_match:
                    mappings[endpoint_key][param_name] = best_match

        return mappings

    @staticmethod
    def _find_best_match(param_name: str, param_type: str, tables_schema: Dict[str, List[Dict[str, str]]]) -> str:
        """Find best matching database field for a parameter"""
        # Simple matching logic - can be enhanced with ML/AI
        param_name_lower = param_name.lower()

        for table_name, columns in tables_schema.items():
            for column in columns:
                column_name = column['name'].lower()

                # Exact match
                if param_name_lower == column_name:
                    return f"{table_name}.{column['name']}"

                # Partial match
                if param_name_lower in column_name or column_name in param_name_lower:
                    return f"{table_name}.{column['name']}"

        return ""


def main():
    st.set_page_config(
        page_title="JMeter Agentic Framework",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ JMeter Agentic Framework")
    st.markdown("Generate JMeter scripts from Swagger APIs and SQL databases using AI")

    # Initialize session state
    if 'swagger_endpoints' not in st.session_state:
        st.session_state.swagger_endpoints = []
    if 'db_tables' not in st.session_state:
        st.session_state.db_tables = []
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}

    # Create three columns for the main inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("🔗 Swagger Configuration")
        swagger_url = st.text_input(
            "Swagger JSON URL",
            value="https://petstore.swagger.io/v2/swagger.json",
            help="Enter the URL to your Swagger/OpenAPI specification"
        )

        if st.button("Fetch Swagger", key="fetch_swagger"):
            with st.spinner("Fetching Swagger specification..."):
                try:
                    parser = SwaggerParser(swagger_url)
                    endpoints = parser.extract_endpoints()
                    st.session_state.swagger_endpoints = endpoints
                    st.success(f"Found {len(endpoints)} API endpoints")
                except Exception as e:
                    st.error(f"Failed to fetch Swagger: {str(e)}")

        # Display endpoints if available
        if st.session_state.swagger_endpoints:
            st.subheader("API Endpoints")
            for endpoint in st.session_state.swagger_endpoints[:5]:  # Show first 5
                st.code(f"{endpoint.method} {endpoint.path}")
            if len(st.session_state.swagger_endpoints) > 5:
                st.info(f"... and {len(st.session_state.swagger_endpoints) - 5} more endpoints")

    with col2:
        st.header("🗄️ Database Configuration")

        db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"])

        if db_type == "SQLite":
            db_file = st.text_input(
                "Database File Path",
                value="petstore.db",
                help="Path to your SQLite database file"
            )
            db_config = DatabaseConfig(db_type="sqlite", file_path=db_file)
        else:
            st.info("MySQL/PostgreSQL support coming soon!")
            db_config = None

        if st.button("Connect Database", key="connect_db") and db_config:
            with st.spinner("Connecting to database..."):
                try:
                    connector = DatabaseConnector(db_config)
                    if connector.connect():
                        tables = connector.get_tables()
                        st.session_state.db_tables = tables
                        st.session_state.db_connector = connector
                        st.success(f"Connected! Found {len(tables)} tables")
                    else:
                        st.error("Failed to connect to database")
                except Exception as e:
                    st.error(f"Database connection error: {str(e)}")

        # Display tables if connected
        if st.session_state.db_tables:
            st.subheader("Database Tables")
            for table in st.session_state.db_tables:
                st.code(table)

    with col3:
        st.header("🤖 AI Prompt Interface")

        prompt = st.text_area(
            "Describe your test scenario",
            placeholder="Example: Create a load test for pet adoption workflow with 50 concurrent users",
            height=100
        )

        # Advanced options
        with st.expander("Advanced Options"):
            thread_group_users = st.number_input("Number of Users", min_value=1, max_value=1000, value=10)
            ramp_up_time = st.number_input("Ramp-up Time (seconds)", min_value=1, max_value=3600, value=30)
            loop_count = st.number_input("Loop Count", min_value=1, max_value=100, value=1)

        if st.button("Generate JMeter Script", key="generate_script", type="primary"):
            if not st.session_state.swagger_endpoints:
                st.error("Please fetch Swagger specification first")
            elif not st.session_state.db_tables:
                st.error("Please connect to database first")
            elif not prompt.strip():
                st.error("Please provide a test scenario description")
            else:
                st.success("JMeter script generation will be implemented here!")
                st.info("This will integrate with your jmeter-agentic-framework")

    # Data Mapping Section
    if st.session_state.swagger_endpoints and st.session_state.db_tables:
        st.header("🔄 Data Mapping")

        if st.button("Generate AI Mapping Suggestions"):
            with st.spinner("Generating mapping suggestions..."):
                # Get table schemas
                tables_schema = {}
                for table in st.session_state.db_tables:
                    schema = st.session_state.db_connector.get_table_schema(table)
                    tables_schema[table] = schema

                # Generate mappings
                mappings = DataMapper.suggest_mappings(st.session_state.swagger_endpoints, tables_schema)
                st.session_state.mappings = mappings

        # Display mappings
        if st.session_state.mappings:
            st.subheader("Suggested Parameter Mappings")
            for endpoint, params in st.session_state.mappings.items():
                if params:  # Only show endpoints with mappings
                    st.write(f"**{endpoint}**")
                    for param, mapping in params.items():
                        st.write(f"  • `{param}` → `{mapping}`")

    # Database Preview Section
    if 'db_connector' in st.session_state and st.session_state.db_tables:
        st.header("📊 Database Preview")

        selected_table = st.selectbox("Select table to preview", st.session_state.db_tables)
        if selected_table:
            preview_df = st.session_state.db_connector.preview_data(selected_table)
            if not preview_df.empty:
                st.dataframe(preview_df)
            else:
                st.info("No data found in selected table")


if __name__ == "__main__":
    main()