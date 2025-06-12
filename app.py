import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field
import os
import sys
import re  # Import regex module
from urllib.parse import urlparse  # Import urlparse for parsing base URL

# FIX: Increase the string conversion limit for large integers.
sys.set_int_max_str_digits(0)  # 0 means unlimited

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Assuming these are available and correct from the 'utils' directory
from utils.jmeter_generator import JMeterScriptGenerator

# Configure logging to DEBUG level for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    summary: Optional[str] = None
    operation_id: Optional[str] = None
    produces: List[str] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None


class SwaggerParser:
    """Parse Swagger/OpenAPI specifications"""

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_spec = None
        self.swagger_data = {}

    def fetch_swagger_spec(self) -> Dict[str, Any]:
        """Fetch and parse Swagger specification"""
        try:
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()
            self.swagger_spec = response.json()
            self.swagger_data = self.swagger_spec  # Store fetched data
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
                    summary = method_data.get('summary')
                    operation_id = method_data.get('operationId')
                    produces = method_data.get('produces', [])

                    request_body_schema = None
                    for param in parameters:
                        if param.get('in') == 'body' and 'schema' in param:
                            request_body_schema = param['schema']
                            break  # Assume only one body parameter

                    endpoints.append(SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        parameters=parameters,
                        responses=responses,
                        summary=summary,
                        operation_id=operation_id,
                        produces=produces,
                        request_body=request_body_schema
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
                # Use check_same_thread=False for Streamlit's multi-threading environment
                self.connection = sqlite3.connect(self.config.file_path, check_same_thread=False)
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
            query = f"SELECT * FROM {table_name}"
            if limit is not None:
                query += f" LIMIT {limit}"
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

            for param in endpoint.parameters:
                param_name = param.get('name', '')
                param_type = param.get('type', param.get('schema', {}).get('type', ''))

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


def _clean_url_string(url_string: str) -> str:
    """
    Attempts to extract a clean URL from a potentially malformed or markdown-formatted string.
    This is a highly defensive function to counter unexpected formatting.
    """
    logger.debug(f"Entering _clean_url_string with: {repr(url_string)}")

    match = re.search(r'(?:\[.*?\]\()?(https?:\/\/[^\s\]\)]+)(?:\))?', url_string)
    if match:
        cleaned_url = match.group(1).strip()
        logger.debug(f"Regex match found, extracted: {repr(cleaned_url)}")
        return cleaned_url

    temp_url = url_string.replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()
    logger.debug(f"After basic bracket removal: {repr(temp_url)}")

    if "https://" in temp_url:
        start_index = temp_url.find("https://")
        cleaned_url = temp_url[start_index:].strip()
        logger.debug(f"Found 'https://' after basic strip, result: {repr(cleaned_url)}")
        return cleaned_url
    elif "http://" in temp_url:
        start_index = temp_url.find("http://")
        cleaned_url = temp_url[start_index:].strip()
        logger.debug(f"Found 'http://' after basic strip, result: {repr(cleaned_url)}")
        return cleaned_url

    logger.debug(f"No clear URL pattern. Returning raw strip: {repr(temp_url)}")
    return temp_url


def call_llm_for_scenario_plan(prompt: str, swagger_endpoints: List[SwaggerEndpoint],
                               db_tables_schema: Dict[str, List[Dict[str, str]]],
                               existing_mappings: Dict[str, Dict[str, str]],
                               thread_group_users: int,
                               ramp_up_time: int,
                               loop_count: int,
                               api_key: str) -> Dict[str, Any]:
    """
    Calls an LLM to generate a high-level test plan suggestion.
    """

    swagger_summary = []
    for ep in swagger_endpoints:
        params_summary = ", ".join([p.get('name', '') for p in ep.parameters])
        headers_summary = " (No specific headers)"
        if ep.method == "POST" or ep.method == "PUT":
            headers_summary = " (Common headers: Content-Type: application/json)"
        swagger_summary.append(f"{ep.method} {ep.path} (Parameters: {params_summary}){headers_summary}")

    db_schema_summary = {}
    for table_name, columns in db_tables_schema.items():
        db_schema_summary[table_name] = [col['name'] for col in columns]

    sanitized_prompt = json.dumps(prompt)

    llm_prompt = f"""
    You are an expert in performance testing and JMeter script generation.
    Your task is to propose a high-level test plan based on the user's request,
    Swagger API endpoints, and a database schema. This proposal will be used
    to guide the user in setting up their test scenario in a UI.

    User's test scenario request: {sanitized_prompt}

    Available Swagger Endpoints:
    {json.dumps(swagger_summary, indent=2)}

    Database Schema:
    {json.dumps(db_schema_summary, indent=2)}

    Existing Data Mappings (parameter -> database.column):
    {json.dumps(existing_mappings, indent=2)}

    Thread Group Configuration (proposed):
    - Number of Users: {thread_group_users}
    - Ramp-up Time (seconds): {ramp_up_time}
    - Loop Count: {loop_count}

    Based on the user's request, provide a concise, high-level summary of
    a suggested test flow. Do not generate the full detailed JSON scenario plan.
    Instead, focus on suggesting:
    1.  Which key API endpoints should be included.
    2.  A logical sequence for these endpoints.
    3.  Any crucial parameters that might need mapping or dynamic generation.
    4.  General advice on assertions or correlation needed.

    Respond in markdown format. Start with "Suggested Test Flow:"
    """

    protocol_segment = "https://"
    domain_segment = "generativelanguage.googleapis.com"
    path_segment = "/v1beta/models/gemini-2.0-flash:generateContent?key="

    temp_base_api_url = "".join([protocol_segment, domain_segment, path_segment])
    base_api_url_cleaned = _clean_url_string(temp_base_api_url)
    api_url = base_api_url_cleaned + api_key

    payload = {
        "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
                result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
                len(result['candidates'][0]['content']['parts']) > 0:
            return {"suggestion": result['candidates'][0]['content']['parts'][0]['text']}
        else:
            logger.error(f"LLM did not return a valid suggestion structure: {result}")
            return {"suggestion": "Could not generate a specific suggestion at this time."}
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling LLM API for suggestion: {e}. "
        if e.response is not None:
            try:
                error_details = e.response.json()
                error_message += f"API Error Details: {json.dumps(error_details)}"
            except json.JSONDecodeError:
                error_message += f"Raw API Error Text: {e.response.text}"
        st.error(f"An error occurred while getting AI suggestion: {error_message}")
        logger.error(error_message)
        return {"suggestion": f"An error occurred while getting AI suggestion: {e}"}
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM call for suggestion: {e}")
        logger.error(f"An unexpected error occurred: {e}")
        return {"suggestion": f"An unexpected error occurred: {e}"}


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
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = None
    if 'db_tables_schema' not in st.session_state:
        st.session_state.db_tables_schema = {}
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    if 'current_swagger_url' not in st.session_state:
        st.session_state.current_swagger_url = ""
    if 'selected_endpoint_keys' not in st.session_state:
        st.session_state.selected_endpoint_keys = []
    if 'scenario_requests_configs' not in st.session_state:
        st.session_state.scenario_requests_configs = []
    if 'llm_suggestion' not in st.session_state:
        st.session_state.llm_suggestion = ""
    if 'enable_constant_timer' not in st.session_state:
        st.session_state.enable_constant_timer = False
    if 'constant_timer_delay_ms' not in st.session_state:
        st.session_state.constant_timer_delay_ms = 300  # Default to 300 ms
    if 'include_scenario_assertions' not in st.session_state:
        st.session_state.include_scenario_assertions = True

    # Create three columns for the main inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("🔗 Swagger Configuration")
        swagger_url_input = st.text_input(
            "Swagger JSON URL",
            value="https://petstore.swagger.io/v2/swagger.json",
            help="Enter the URL to your Swagger/OpenAPI specification"
        )
        swagger_url = _clean_url_string(swagger_url_input)
        logger.debug(f"Swagger URL after cleaning: {repr(swagger_url)}")

        if st.button("Fetch Swagger", key="fetch_swagger") or \
                (
                        swagger_url and swagger_url != st.session_state.current_swagger_url and not st.session_state.swagger_endpoints):
            if swagger_url:
                with st.spinner("Fetching Swagger specification..."):
                    try:
                        parser = SwaggerParser(swagger_url)
                        endpoints = parser.extract_endpoints()
                        st.session_state.swagger_endpoints = endpoints
                        st.session_state.swagger_parser = parser  # Store the parser instance to access swagger_data
                        st.session_state.current_swagger_url = swagger_url
                        st.session_state.selected_endpoint_keys = []
                        st.session_state.scenario_requests_configs = []
                        st.success(f"Found {len(endpoints)} API endpoints.")
                    except Exception as e:
                        st.error(f"Failed to fetch Swagger: {str(e)}")
                        st.session_state.swagger_endpoints = []
                        st.session_state.selected_endpoint_keys = []
                        st.session_state.scenario_requests_configs = []
            else:
                st.warning("Please enter a Swagger JSON URL.")

        if st.session_state.swagger_endpoints:
            st.subheader("API Endpoints")
            for endpoint in st.session_state.swagger_endpoints[:5]:
                st.code(f"{endpoint.method} {endpoint.path}")
            if len(st.session_state.swagger_endpoints) > 5:
                st.info(f"... and {len(st.session_state.swagger_endpoints) - 5} more endpoints")

    with col2:
        st.header("🗄️ Database Configuration")

        db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"], key="db_type_select")

        db_file = ""
        db_config = None
        if db_type == "SQLite":
            db_file = st.text_input(
                "Database File Path",
                value="database/petstore.db",
                help="Path to your SQLite database file"
            )
            db_config = DatabaseConfig(db_type="sqlite", file_path=db_file)
        else:
            st.info("MySQL/PostgreSQL support coming soon!")

        if st.button("Connect Database", key="connect_db") and db_config:
            if db_type == "SQLite" and not os.path.exists(db_file):
                st.error(f"Database file not found at: {db_file}. Please ensure the file exists or create it.")
                if st.button("Create Dummy SQLite DB", key="create_dummy_db"):
                    try:
                        os.makedirs(os.path.dirname(db_file), exist_ok=True)
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        cursor.execute("""
                                       CREATE TABLE IF NOT EXISTS pets
                                       (
                                           id
                                           INTEGER
                                           PRIMARY
                                           KEY,
                                           name
                                           TEXT
                                           NOT
                                           NULL,
                                           status
                                           TEXT
                                           NOT
                                           NULL,
                                           tags
                                           TEXT
                                       );
                                       """)
                        cursor.execute("""
                                       CREATE TABLE IF NOT EXISTS users
                                       (
                                           id
                                           INTEGER
                                           PRIMARY
                                           KEY,
                                           username
                                           TEXT
                                           NOT
                                           NULL,
                                           password
                                           TEXT
                                           NOT
                                           NULL,
                                           email
                                           TEXT
                                       );
                                       """)
                        cursor.execute(
                            "INSERT INTO pets (name, status, tags) VALUES ('Buddy', 'available', 'dog,friendly');")
                        cursor.execute("INSERT INTO pets (name, status, tags) VALUES ('Whiskers', 'pending', 'cat');")
                        cursor.execute(
                            "INSERT INTO users (username, password, email) VALUES ('testuser', 'testpass', 'test@example.com');")
                        conn.commit()
                        conn.close()
                        st.success(f"Dummy SQLite database created at {db_file} with sample data.")
                        st.rerun()  # Replaced st.experimental_rerun()
                    except Exception as ex:
                        st.error(f"Error creating dummy DB: {ex}")
            else:
                with st.spinner("Connecting to database..."):
                    try:
                        connector = DatabaseConnector(db_config)
                        if connector.connect():
                            st.session_state.db_connector = connector
                            tables = connector.get_tables()
                            st.session_state.db_tables = tables
                            tables_schema = {}
                            for table in tables:
                                schema = connector.get_table_schema(table)
                                tables_schema[table] = schema
                            st.session_state.db_tables_schema = tables_schema
                            st.success(f"Connected! Found {len(tables)} tables")
                        else:
                            st.error("Failed to connect to database")
                    except Exception as e:
                        st.error(f"Database connection error: {str(e)}")
                        st.session_state.db_tables = []

        if st.session_state.db_tables:
            st.subheader("Database Tables")
            for table in st.session_state.db_tables:
                st.code(table)

    with col3:
        st.header("🤖 AI Assistance & API Key")

        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Enter your Google Gemini API Key. Get one from Google AI Studio."
        )

        prompt = st.text_area(
            "Describe your test scenario (for AI suggestions)",
            value="Generate a JMeter script that covers all available API endpoints. For each endpoint, use mapped database fields if available, otherwise generate dummy data. Aim for a typical user flow involving GET, POST, PUT, and DELETE operations where appropriate.",
            height=100
        )

        if st.button("Get AI Suggestions", key="get_ai_suggestions_btn"):
            if not st.session_state.swagger_endpoints:
                st.error("Please fetch Swagger specification first to get AI suggestions.")
            elif not st.session_state.db_tables:
                st.error("Please connect to database first to get AI suggestions for mappings.")
            elif not st.session_state.gemini_api_key.strip():
                st.error("Please provide your Gemini API Key to get AI suggestions.")
            else:
                with st.spinner("Getting AI mapping and scenario suggestions..."):
                    if not st.session_state.mappings:
                        tables_schema_for_mapping = {}
                        for table in st.session_state.db_tables:
                            schema = st.session_state.db_connector.get_table_schema(table)
                            tables_schema_for_mapping[table] = schema
                        st.session_state.mappings = DataMapper.suggest_mappings(st.session_state.swagger_endpoints,
                                                                                tables_schema_for_mapping)

                    llm_response = call_llm_for_scenario_plan(
                        prompt=prompt,
                        swagger_endpoints=st.session_state.swagger_endpoints,
                        db_tables_schema=st.session_state.db_tables_schema,
                        existing_mappings=st.session_state.mappings,
                        thread_group_users=st.session_state.num_users_input,
                        ramp_up_time=st.session_state.ramp_up_time_input,
                        loop_count=st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1,
                        api_key=st.session_state.gemini_api_key
                    )
                    st.session_state.llm_suggestion = llm_response.get("suggestion", "No specific suggestion provided.")

        if st.session_state.llm_suggestion:
            st.subheader("AI Suggested Test Flow (Read-Only)")
            st.markdown(st.session_state.llm_suggestion)
            st.info("Use this as a guide to configure your scenario below.")

    st.header("2. Load Profile Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        num_users = st.number_input(
            "Number of Concurrent Users (Threads)",
            min_value=1,
            value=10,
            step=1,
            key="num_users_input"
        )
    with col2:
        ramp_up_time = st.number_input(
            "Ramp-up Time (seconds)",
            min_value=0,
            value=30,
            step=5,
            key="ramp_up_time_input"
        )
    with col3:
        loop_count_option = st.selectbox(
            "Loop Count",
            options=["Infinite", "Specify iterations"],
            key="loop_count_option"
        )
        loop_count = 1
        if loop_count_option == "Specify iterations":
            loop_count = st.number_input(
                "Number of Iterations",
                min_value=1,
                value=1,
                step=1,
                key="loop_count_input_specific"
            )
        else:
            loop_count = -1

    st.header("2.1 Optional Global JMeter Elements")
    col_timer, col_assertion = st.columns(2)

    with col_timer:
        st.session_state.enable_constant_timer = st.checkbox(
            "Enable Global Constant Timer",
            value=st.session_state.enable_constant_timer,
            key="enable_constant_timer_checkbox"
        )
        if st.session_state.enable_constant_timer:
            st.session_state.constant_timer_delay_ms = st.number_input(
                "Constant Timer Delay (milliseconds)",
                min_value=0,
                value=st.session_state.constant_timer_delay_ms,
                step=100,
                key="constant_timer_delay_input",
                help="Delay (in ms) to pause between requests. This is applied globally to the Thread Group."
            )
        else:
            st.session_state.constant_timer_delay_ms = 0

    with col_assertion:
        st.session_state.include_scenario_assertions = st.checkbox(
            "Include Response Assertions in Scenario",
            value=st.session_state.include_scenario_assertions,
            key="include_scenario_assertions_checkbox",
            help="If unchecked, no assertion configuration fields will be shown or generated for any request. A default 'Response Code 200' will be added if enabled."
        )

    st.header("3. Select Endpoints for Scenario")
    if st.session_state.swagger_endpoints:
        endpoint_options = [f"{ep.method} {ep.path}" for ep in st.session_state.swagger_endpoints]

        selected_endpoint_keys = st.multiselect(
            "Select API Endpoints for your scenario (order reflects execution sequence)",
            options=endpoint_options,
            default=st.session_state.selected_endpoint_keys,
            key="endpoint_selector"
        )

        if selected_endpoint_keys != st.session_state.selected_endpoint_keys:
            st.session_state.selected_endpoint_keys = selected_endpoint_keys
            new_scenario_configs = []

            if not st.session_state.mappings and st.session_state.db_tables:
                tables_schema_for_mapping = {}
                for table in st.session_state.db_tables:
                    schema = st.session_state.db_connector.get_table_schema(table)
                    tables_schema_for_mapping[table] = schema
                st.session_state.mappings = DataMapper.suggest_mappings(st.session_state.swagger_endpoints,
                                                                        tables_schema_for_mapping)

            for ep_key in selected_endpoint_keys:
                current_endpoint = next(
                    (ep for ep in st.session_state.swagger_endpoints if f"{ep.method} {ep.path}" == ep_key), None)

                if current_endpoint:
                    request_config = {
                        "endpoint_key": ep_key,
                        "name": f"{current_endpoint.method} {current_endpoint.path.replace('/', '_')}",
                        "method": current_endpoint.method,
                        "path": current_endpoint.path,
                        "parameters": {},
                        "headers": {},
                        "body": None,
                        "assertions": [],
                        "json_extractors": [],
                        "think_time": 0
                    }

                    if current_endpoint.parameters:
                        for param in current_endpoint.parameters:
                            if param.get('in') in ['query', 'path']:
                                mapped_value = st.session_state.mappings.get(ep_key, {}).get(param['name'], None)
                                if mapped_value:
                                    request_config['parameters'][param['name']] = f"${{{mapped_value}}}"
                                else:
                                    if param.get('type') == 'string':
                                        request_config['parameters'][param['name']] = f"dummy_{param['name']}"
                                    elif param.get('type') == 'integer':
                                        request_config['parameters'][param['name']] = "123"
                                    elif param.get('type') == 'boolean':
                                        request_config['parameters'][param['name']] = "true"
                                    else:
                                        request_config['parameters'][param['name']] = f"dummy_{param['name']}"

                    if current_endpoint.method in ["POST", "PUT", "PATCH"]:
                        request_config["headers"] = {"Content-Type": "application/json"}
                        if current_endpoint.request_body and st.session_state.get('swagger_parser'):
                            try:
                                def resolve_ref(schema_obj, definitions):
                                    if '$ref' in schema_obj:
                                        ref_path = schema_obj['$ref'].split('/')
                                        definition_name = ref_path[-1]
                                        return definitions.get(definition_name, {})
                                    return schema_obj

                                definitions = st.session_state.swagger_parser.swagger_data.get('definitions', {})
                                resolved_schema = resolve_ref(current_endpoint.request_body, definitions)
                                def_props = resolved_schema.get('properties', {})

                                if def_props:
                                    dummy_body = {}
                                    for prop_name, prop_details in def_props.items():
                                        if prop_details.get('type') == 'string':
                                            dummy_body[prop_name] = f"auto_{prop_name}"
                                        elif prop_details.get('type') == 'integer':
                                            dummy_body[prop_name] = 456
                                        elif prop_details.get('type') == 'boolean':
                                            dummy_body[prop_name] = True
                                        else:
                                            dummy_body[prop_name] = f"auto_value_{prop_name}"
                                    request_config["body"] = json.dumps(dummy_body, indent=2)
                                else:
                                    request_config["body"] = "{\n  \"message\": \"auto-generated dummy body\"\n}"
                            except Exception as e:
                                logger.warning(f"Could not infer request body from schema for {ep_key}: {e}")
                                request_config["body"] = "{\n  \"message\": \"auto-generated dummy body\"\n}"
                        else:
                            request_config["body"] = "{\n  \"message\": \"auto-generated dummy body\"\n}"

                    if st.session_state.include_scenario_assertions:
                        request_config['assertions'].append({"type": "Response Code", "value": "200"})

                    new_scenario_configs.append(request_config)

            st.session_state.scenario_requests_configs = new_scenario_configs
            st.rerun()  # Replaced st.experimental_rerun()

        st.markdown("---")
        st.subheader("Selected Endpoints (Auto-Configured)")
        if st.session_state.scenario_requests_configs:
            for i, config in enumerate(st.session_state.scenario_requests_configs):
                st.code(f"Request {i + 1}: {config['method']} {config['path']}")
                with st.expander(f"View Auto-Configuration for {config['name']}"):
                    st.json(config)
        else:
            st.info("Select endpoints above to auto-configure their details.")

    else:
        st.info("Please fetch a Swagger file to build your test scenario.")

    st.markdown("---")

    if st.button("Generate JMeter Script", key="generate_button_final", type="primary"):
        if not st.session_state.swagger_endpoints:
            st.error("Please fetch a Swagger/OpenAPI JSON file and ensure endpoints are parsed.")
            return
        if not st.session_state.selected_endpoint_keys:
            st.error("Please select at least one API request to include in your scenario.")
            return

        st.info("Generating JMeter script... Please wait.")

        try:
            scenario_plan = {"requests": st.session_state.scenario_requests_configs}

            num_users = st.session_state.num_users_input
            ramp_up_time = st.session_state.ramp_up_time_input
            loop_count = st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1

            global_constant_timer_delay = st.session_state.constant_timer_delay_ms if st.session_state.enable_constant_timer else 0

            generator = JMeterScriptGenerator()
            jmx_content, csv_content = generator.generate_jmx(
                app_base_url=swagger_url,
                thread_group_users=num_users,
                ramp_up_time=ramp_up_time,
                loop_count=loop_count,
                scenario_plan=scenario_plan,
                csv_data=None,
                global_constant_timer_delay=global_constant_timer_delay
            )

            st.success("JMeter script generated successfully!")

            st.subheader("Generated JMeter Test Plan (.jmx)")
            st.code(jmx_content, language="xml")

            st.download_button(
                label="Download JMX File",
                data=jmx_content.encode("utf-8"),
                file_name="generated_test_plan.jmx",
                mime="application/xml",
                key="download_jmx_final"
            )

            if csv_content:
                st.subheader("Generated CSV Data (data.csv)")
                st.code(csv_content, language="csv")
                st.download_button(
                    label="Download CSV File",
                    data=csv_content.encode("utf-8"),
                    file_name="data.csv",
                    mime="text/csv",
                    key="download_csv_final"
                )
            else:
                st.info("No CSV data was generated for this test plan.")

        except Exception as e:
            st.error(f"An error occurred during script generation: {e}")
            logger.error(f"Error in main app execution: {e}", exc_info=True)

    st.markdown("---")
    st.markdown("Developed with ❤️ by Your AI Assistant")


if __name__ == "__main__":
    if not os.path.exists("swagger.json"):
        dummy_swagger_content = """
{
  "swagger": "2.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore"
  },
  "host": "petstore.swagger.io",
  "basePath": "/v2",
  "schemes": [
    "https"
  ],
  "paths": {
    "/pet/findByStatus": {
      "get": {
        "summary": "Finds Pets by status",
        "operationId": "findPetsByStatus",
        "produces": [
          "application/xml",
          "application/json"
        ],
        "parameters": [
          {
            "name": "status",
            "in": "query",
            "description": "Status values that need to be considered for filter",
            "required": true,
            "type": "array",
            "items": {
              "type": "string",
              "enum": [
                "available",
                "pending",
                "sold"
              ],
              "default": "available"
            },
            "collectionFormat": "multi"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation"
          }
        }
      }
    },
    "/user": {
      "post": {
        "summary": "Create user",
        "operationId": "createUser",
        "produces": [
          "application/xml",
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "Created user object",
            "required": true,
            "schema": {
              "$ref": "#/definitions/User"
            }
          }
        ],
        "responses": {
          "default": {
            "description": "successful operation"
          }
        }
      }
    },
    "/login": {
      "post": {
        "summary": "Logs user into the system",
        "operationId": "loginUser",
        "produces": [
          "application/xml",
          "application/json"
        ],
        "parameters": [
          {
            "name": "username",
            "in": "query",
            "description": "The user name for login",
            "required": true,
            "type": "string"
          },
          {
            "name": "password",
            "in": "query",
            "description": "The password for login in clear text",
            "required": true,
            "type": "string"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation"
          }
        }
      }
    }
  },
  "definitions": {
    "User": {
      "type": "object",
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "username": {
          "type": "string"
        },
        "email": {
          "type": "string"
        }
      }
    }
  }
}
        """
        with open("swagger.json", "w") as f:
            f.write(dummy_swagger_content)
        logger.info("Created a dummy swagger.json file for demonstration.")

    dummy_db_path = "database/petstore.db"
    if not os.path.exists(dummy_db_path):
        try:
            os.makedirs(os.path.dirname(dummy_db_path), exist_ok=True)
            conn = sqlite3.connect(dummy_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS pets
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY,
                               name
                               TEXT
                               NOT
                               NULL,
                               status
                               TEXT
                               NOT
                               NULL,
                               tags
                               TEXT
                           );
                           """)
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS users
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY,
                               username
                               TEXT
                               NOT
                               NULL,
                               password
                               TEXT
                               NOT
                               NULL,
                               email
                               TEXT
                           );
                           """)
            cursor.execute("INSERT INTO pets (name, status, tags) VALUES ('Buddy', 'available', 'dog,friendly');")
            cursor.execute("INSERT INTO pets (name, status, tags) VALUES ('Whiskers', 'pending', 'cat');")
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES ('testuser', 'testpass', 'test@example.com');")
            conn.commit()
            conn.close()
            logger.info(f"Dummy SQLite database created at {dummy_db_path} with sample data.")
        except Exception as ex:
            logger.error(f"Error creating dummy DB during __main__ init: {ex}")

    main()
