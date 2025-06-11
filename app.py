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
import re  # Import regex module

# FIX: Increase the string conversion limit for large integers.
sys.set_int_max_str_digits(0)  # 0 means unlimited

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from swagger_parser import SwaggerParser, SwaggerEndpoint
from database_connector import DatabaseConnector, DatabaseConfig
from data_mapper import DataMapper
from jmeter_generator import JMeterScriptGenerator  # Re-enabled the JMeterScriptGenerator import

# Configure logging to DEBUG level for detailed output
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG
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

    # Step 1: Look for http(s):// at the beginning of the string or inside markdown link parentheses.
    # This regex tries to capture the URL part from various malformed inputs, prioritizing the part inside markdown.
    # (?:\[.*?\]\()?: Non-capturing group for optional markdown text part like [text](
    # (https?:\/\/[^\s\]\)]+): Captures the actual URL (http/https followed by non-whitespace, non-bracket characters).
    # (?:\))?: Non-capturing group for optional closing parenthesis for markdown links.
    match = re.search(r'(?:\[.*?\]\()?(https?:\/\/[^\s\]\)]+)(?:\))?', url_string)
    if match:
        cleaned_url = match.group(1).strip()
        logger.debug(f"Regex match found, extracted: {repr(cleaned_url)}")
        return cleaned_url

    # Fallback 1: If no strong URL pattern (like the regex above) is found,
    # try to remove common markdown/bracket characters directly.
    temp_url = url_string.replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()
    logger.debug(f"After basic bracket removal: {repr(temp_url)}")

    # Fallback 2: If after basic removal, it still contains "https://" or "http://",
    # take everything from that point onwards as the URL. This handles cases where
    # the URL might be prefixed by other junk.
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

    # Final fallback: Return the string after just stripping whitespace and initial markdown attempt.
    logger.debug(f"No clear URL pattern. Returning raw strip: {repr(temp_url)}")
    return temp_url


def call_llm_for_scenario_plan(prompt: str, swagger_endpoints: List[SwaggerEndpoint],
                               db_tables_schema: Dict[str, List[Dict[str, str]]],
                               existing_mappings: Dict[str, Dict[str, str]],
                               thread_group_users: int,
                               ramp_up_time: int,
                               loop_count: int,
                               api_key: str) -> Dict[str, Any]:  # api_key now passed as argument
    """
    Calls an LLM to generate a structured scenario plan for JMeter.
    The LLM will parse the prompt, swagger, and DB schema to decide the test flow.
    """

    # Prepare context for the LLM
    swagger_summary = []
    for ep in swagger_endpoints:
        params_summary = ", ".join([p.get('name', '') for p in ep.parameters])
        # Include a placeholder for headers in the summary to guide LLM
        headers_summary = " (No specific headers)"
        if ep.method == "POST" or ep.method == "PUT":  # Common methods for Content-Type
            headers_summary = " (Common headers: Content-Type: application/json)"
        swagger_summary.append(f"{ep.method} {ep.path} (Parameters: {params_summary}){headers_summary}")

    db_schema_summary = {}
    for table_name, columns in db_tables_schema.items():
        db_schema_summary[table_name] = [col['name'] for col in columns]

    # Sanitize the user's prompt by escaping it for JSON inclusion
    sanitized_prompt = json.dumps(prompt)

    # Construct the prompt for the LLM
    llm_prompt = f"""
    You are an expert in performance testing and JMeter script generation.
    Your task is to create a JMeter test plan scenario based on the user's request,
    Swagger API endpoints, and a database schema.

    User's test scenario request: {sanitized_prompt}

    Available Swagger Endpoints:
    {json.dumps(swagger_summary, indent=2)}

    Database Schema:
    {json.dumps(db_schema_summary, indent=2)}

    Existing Data Mappings (parameter -> database.column):
    {json.dumps(existing_mappings, indent=2)}

    Thread Group Configuration:
    - Number of Users: {thread_group_users}
    - Ramp-up Time (seconds): {ramp_up_time}
    - Loop Count: {loop_count}

    Based on the user's request, determine which API endpoints should be hit, in what order,
    what parameters to use (prioritizing mapped parameters from the database, or generating
    dummy/random data if no mapping exists and it's a POST/PUT), and if any chaining
    (e.g., extracting an ID from one response to use in a subsequent request) is needed.
    For POST/PUT requests, include appropriate 'Content-Type' headers as a JSON string.

    VERY IMPORTANT: Your response MUST be a single, valid JSON object that strictly adheres to the following schema.
    DO NOT include any introductory text, conversational filler, markdown formatting (like triple backticks, e.g., ```json), or any other extraneous characters before or after the JSON object.
    The output should start directly with '{{' and end directly with '}}'.

    {{
        "test_name": "A descriptive name for the test",
        "thread_group": {{
            "num_users": int,
            "ramp_up": int,
            "loop_count": int
        }},
        "requests": [
            {{
                "endpoint_key": "METHOD /path",
                "name": "Human-readable sampler name",
                "method": "GET|POST|PUT|DELETE|PATCH",
                "path": "/actual/path",
                "parameters": "JSON string of parameters, e.g., '{{\\"param1\\": \\"value1\\", \\"param2\\": \\"${{mapped_variable_name}}\\"}}'",
                "body": "Raw body for POST/PUT (JSON/XML string), or null",
                "use_mapping": true/false,
                "headers": "JSON string of HTTP headers, e.g., '{{\\"Content-Type\\": \\"application/json\\", \\"Authorization\\": \\"Bearer xyz\\"}}'"
            }}
        ]
    }}

    If the user's prompt implies a sequence, reflect that in the order of requests.
    If no specific flow is mentioned, generate a default scenario that includes common operations
    like fetching data (GET) and potentially creating/updating data (POST/PUT) based on mappings.
    For path parameters (e.g., in /pet/{{petId}}), ensure the path reflects the JMeter variable
    if mapped (e.g., /pet/${{pets_id}}).
    For dynamic data (like IDs for new creations), use JMeter functions where appropriate (e.g., ${{__Random(1,1000,)}}).
    """

    # Building URL from fragments to ensure no markdown is embedded in the literal.
    # This is an extreme measure to bypass external parsing issues.
    protocol_segment = "https://"
    domain_segment = "generativelanguage.googleapis.com"
    path_segment = "/v1beta/models/gemini-2.0-flash:generateContent?key="

    logger.debug(f"Protocol part (literal): {repr(protocol_segment)}")
    logger.debug(f"Domain part (literal): {repr(domain_segment)}")
    logger.debug(f"Path part (literal): {repr(path_segment)}")

    # Joining fragments
    temp_base_api_url = "".join([protocol_segment, domain_segment, path_segment])
    logger.debug(f"Combined base_api_url before explicit cleaning: {repr(temp_base_api_url)}")

    # Explicitly clean the base URL after combination
    base_api_url_cleaned = _clean_url_string(temp_base_api_url)
    logger.debug(f"Combined base_api_url after explicit cleaning: {repr(base_api_url_cleaned)}")

    # Final API URL to be used in the request
    api_url = base_api_url_cleaned + api_key
    logger.debug(f"Final api_url before request: {repr(api_url)}")
    logger.info(f"Attempting to call LLM API at URL: {api_url}")
    logger.info(f"Type of api_url: {type(api_url)}")
    logger.info(f"Raw repr of api_url: {repr(api_url)}")  # This is the ultimate check

    # Construct the payload for the LLM API
    payload = {
        "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "test_name": {"type": "STRING"},
                    "thread_group": {
                        "type": "OBJECT",
                        "properties": {
                            "num_users": {"type": "INTEGER"},
                            "ramp_up": {"type": "INTEGER"},
                            "loop_count": {"type": "INTEGER"}
                        }
                    },
                    "requests": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "endpoint_key": {"type": "STRING"},
                                "name": {"type": "STRING"},
                                "method": {"type": "STRING"},
                                "path": {"type": "STRING"},
                                # Parameters are now a STRING type in the schema
                                "parameters": {"type": "STRING"},
                                "body": {"type": "STRING", "nullable": True},
                                "use_mapping": {"type": "BOOLEAN"},
                                # FIX: Add headers as a STRING type in the schema
                                "headers": {"type": "STRING"}
                            },
                            "required": ["endpoint_key", "name", "method", "path", "parameters", "use_mapping",
                                         "headers"]
                        }
                    }
                },
                "required": ["test_name", "thread_group", "requests"]
            }
        }
    }

    # Log the full JSON payload before sending
    logger.debug(f"Full JSON payload being sent to LLM API: {json.dumps(payload, indent=2)}")

    scenario_plan = None
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
                result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
                len(result['candidates'][0]['content']['parts']) > 0:

            generated_json_str = result['candidates'][0]['content']['parts'][0]['text']

            # Robust JSON parsing: try to extract JSON from potentially malformed LLM output
            # Look for content between the first '{' and the last '}'
            json_match = re.search(r'(\{.*\})', generated_json_str, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)  # Get the entire matched group
            else:
                clean_json_str = generated_json_str.strip()  # Fallback if no match, just strip whitespace

            # Attempt to parse the cleaned string as JSON.
            try:
                scenario_plan = json.loads(clean_json_str)
                logger.debug(f"LLM generated scenario plan (raw): {scenario_plan}")

                # Post-process: Parse the 'parameters' and 'headers' strings into dictionaries for each request
                for req in scenario_plan.get('requests', []):
                    # Process parameters
                    if 'parameters' in req and isinstance(req['parameters'], str):
                        logger.debug(f"Attempting to parse parameters string: {repr(req['parameters'])}")
                        try:
                            req['parameters'] = json.loads(req['parameters'])
                            if req['parameters'] is None:  # If string was 'null'
                                req['parameters'] = {}
                            logger.debug(f"Parameters parsed successfully: {req['parameters']}")
                        except json.JSONDecodeError as param_e:
                            logger.error(
                                f"Failed to parse parameters JSON string: {param_e}. Raw: {repr(req['parameters'])}")
                            req['parameters'] = {}  # Default to empty dict on failure
                    elif 'parameters' not in req:
                        req['parameters'] = {}  # Ensure parameters field always exists as a dict

                    # Process headers
                    if 'headers' in req and isinstance(req['headers'], str):
                        logger.debug(f"Attempting to parse headers string: {repr(req['headers'])}")
                        try:
                            req['headers'] = json.loads(req['headers'])
                            if req['headers'] is None:  # If string was 'null'
                                req['headers'] = {}
                            logger.debug(f"Headers parsed successfully: {req['headers']}")
                        except json.JSONDecodeError as header_e:
                            logger.error(
                                f"Failed to parse headers JSON string: {header_e}. Raw: {repr(req['headers'])}")
                            req['headers'] = {}  # Default to empty dict on failure
                    elif 'headers' not in req:
                        req['headers'] = {}  # Ensure headers field always exists as a dict

                return scenario_plan
            except json.JSONDecodeError as e:
                st.error(f"LLM response was not valid JSON after cleaning: {e}. Raw response: {clean_json_str}")
                logger.error(f"LLM response was not valid JSON after cleaning: {e}. Raw response: {clean_json_str}")
                # Fallback to default if JSON parsing fails
                scenario_plan = None  # Ensure it falls through to default generation
        else:
            st.error("LLM did not return a valid scenario plan. Generating default.")
            logger.error(f"LLM response structure unexpected: {result}")
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling LLM API: {e}. "
        if e.response is not None:
            try:
                error_details = e.response.json()
                error_message += f"API Error Details: {json.dumps(error_details)}"
                if e.response.status_code == 403:
                    st.error(f"**Authentication/Permission Error (403 Forbidden):**")
                    st.error(
                        f"Please ensure your API Key is correct and has the necessary permissions to access the Gemini API.")
                    st.error(
                        f"You may need to enable the Gemini API in your Google Cloud project and check your billing settings.")
            except json.JSONDecodeError:
                error_message += f"Raw API Error Text: {e.response.text}"
        else:
            error_message += "No response text."
        st.error(error_message)
        logger.error(error_message)
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM call: {e}. Generating default scenario.")
        logger.error(f"An unexpected error occurred: {e}")

    # Fallback/Default Scenario Plan if LLM call fails or returns invalid format
    default_requests = []
    # If scenario_plan is still None here, it means LLM failed or returned invalid JSON
    # so we proceed with the default generation logic.
    if scenario_plan is None:
        for endpoint in swagger_endpoints:
            # Simple heuristic: include GET and POST requests by default
            if endpoint.method in ["GET", "POST"]:
                params_for_llm = {}
                for param in endpoint.parameters:
                    param_name = param.get('name')
                    mapped_db_col = existing_mappings.get(f"{endpoint.method} {endpoint.path}", {}).get(param_name)
                    if mapped_db_col:
                        params_for_llm[param_name] = f"${{{mapped_db_col.replace('.', '_')}}}"
                    else:
                        params_for_llm[param_name] = ""  # Empty or placeholder

                body_content = None
                headers_content = {}  # Default empty headers
                if endpoint.method == "POST":
                    # For a simple default POST, assume a JSON body if parameters exist
                    # This needs to be smarter based on swagger 'schema' for body
                    if endpoint.parameters:
                        # Filter parameters that are 'in': 'body' or 'formData' for JSON body
                        body_params = [p for p in endpoint.parameters if p.get('in') in ['body', 'formData', 'query']]
                        if body_params:
                            dummy_body = {}
                            for p in body_params:
                                param_name = p.get('name')
                                if param_name:  # Ensure param_name is not None
                                    mapped_db_col = existing_mappings.get(f"{endpoint.method} {endpoint.path}", {}).get(
                                        param_name)
                                    if mapped_db_col:
                                        dummy_body[param_name] = f"${{{mapped_db_col.replace('.', '_')}}}"
                                    else:
                                        # Provide a placeholder value if no mapping
                                        # Based on common types, providing string 'value' or random int
                                        if p.get('type') == 'integer':
                                            dummy_body[param_name] = "${__Random(1,1000,)}"
                                        elif p.get('type') == 'string':
                                            dummy_body[param_name] = "string_value"
                                        else:
                                            dummy_body[param_name] = "value"
                            if dummy_body:
                                body_content = json.dumps(dummy_body)
                                # Add Content-Type for default POST body
                                headers_content['Content-Type'] = 'application/json'

                default_requests.append({
                    "endpoint_key": f"{endpoint.method} {endpoint.path}",
                    "name": f"{endpoint.method} {endpoint.path.replace('/', '_')}",
                    "method": endpoint.method,
                    "path": endpoint.path,
                    "parameters": params_for_llm,
                    "body": body_content,
                    "use_mapping": True,
                    "headers": headers_content  # Add headers to default request
                })

        default_scenario_plan = {
            "test_name": "Default Generated Test Plan",
            "thread_group": {
                "num_users": thread_group_users,
                "ramp_up": ramp_up_time,
                "loop_count": loop_count,
            },
            "requests": default_requests
        }
        return default_scenario_plan
    else:
        return scenario_plan


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
    # New state for API key input
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""

    # Create three columns for the main inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("🔗 Swagger Configuration")
        # FIX: Changed default value to a clean URL string, as requested.
        swagger_url_input = st.text_input(
            "Swagger JSON URL",
            value="https://petstore.swagger.io/v2/swagger.json",
            # Clean default
            help="Enter the URL to your Swagger/OpenAPI specification"
        )
        # Apply cleaning to the swagger URL input field as well
        swagger_url = _clean_url_string(swagger_url_input)
        logger.debug(f"Swagger URL after cleaning: {repr(swagger_url)}")

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

        db_file = ""
        if db_type == "SQLite":
            db_file = st.text_input(
                "Database File Path",
                value="database/petstore.db",
                help="Path to your SQLite database file"
            )
            db_config = DatabaseConfig(db_type="sqlite", file_path=db_file)
        else:
            st.info("MySQL/PostgreSQL support coming soon!")
            db_config = None

        if st.button("Connect Database", key="connect_db") and db_config:
            if not os.path.exists(db_file):
                st.error(f"Database file not found at: {db_file}")
            else:
                with st.spinner("Connecting to database..."):
                    try:
                        @st.cache_resource
                        def get_database_connector(config):
                            connector = DatabaseConnector(config)
                            if connector.connect():
                                return connector
                            return None

                        connector = get_database_connector(db_config)
                        st.session_state.db_connector = connector

                        if connector:
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

        if st.session_state.db_tables:
            st.subheader("Database Tables")
            for table in st.session_state.db_tables:
                st.code(table)

    with col3:
        st.header("🤖 AI Prompt & API Key")

        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",  # Use password type for sensitivity
            value=st.session_state.gemini_api_key,
            help="Enter your Google Gemini API Key. Get one from Google AI Studio."
        )

        # Updated default prompt text
        prompt = st.text_area(
            "Describe your test scenario",
            value="Generate a JMeter script that covers all available API endpoints. For each endpoint, use mapped database fields if available, otherwise generate dummy data. Aim for a typical user flow involving GET, POST, PUT, and DELETE operations where appropriate.",
            height=100
        )

        with st.expander("Advanced Options"):
            thread_group_users = st.number_input("Number of Users", min_value=1, max_value=1000, value=10,
                                                 key="num_users")
            ramp_up_time = st.number_input("Ramp-up Time (seconds)", min_value=1, max_value=3600, value=30,
                                           key="ramp_up")
            loop_count = st.number_input("Loop Count", min_value=1, max_value=100, value=1, key="loop_count")
            base_url = st.text_input("Base URL for API (e.g., petstore.swagger.io)", value="petstore.swagger.io",
                                     key="base_url")
            protocol = st.selectbox("Protocol", ["http", "https"], index=1 if "https" in swagger_url else 0,
                                    key="protocol")
            port = st.text_input("Port (Optional)", value="", key="port")

        if st.button("Generate JMeter Script", key="generate_script", type="primary"):
            if not st.session_state.swagger_endpoints:
                st.error("Please fetch Swagger specification first")
            elif not st.session_state.db_tables:
                st.error("Please connect to database first")
            elif not prompt.strip():
                st.error("Please provide a test scenario description")
            elif not st.session_state.gemini_api_key.strip():
                st.error("Please provide your Gemini API Key in the 'Gemini API Key' field.")
            else:
                with st.spinner("Generating JMeter script with AI..."):
                    try:
                        # Ensure mappings are generated first if not already
                        if not st.session_state.mappings:
                            # Get table schemas
                            tables_schema_for_mapping = {}
                            for table in st.session_state.db_tables:
                                schema = st.session_state.db_connector.get_table_schema(table)
                                tables_schema_for_mapping[table] = schema
                            st.session_state.mappings = DataMapper.suggest_mappings(st.session_state.swagger_endpoints,
                                                                                    tables_schema_for_mapping)

                        # Call LLM to get scenario plan, passing the API key from session state
                        scenario_plan = call_llm_for_scenario_plan(
                            prompt=prompt,
                            swagger_endpoints=st.session_state.swagger_endpoints,
                            db_tables_schema=st.session_state.db_tables_schema,
                            existing_mappings=st.session_state.mappings,
                            thread_group_users=thread_group_users,
                            ramp_up_time=ramp_up_time,
                            loop_count=loop_count,
                            api_key=st.session_state.gemini_api_key  # Pass the API key from input
                        )

                        # Check if scenario_plan was successfully generated (not None due to LLM error)
                        if scenario_plan:
                            # Generate JMeter JMX
                            generator = JMeterScriptGenerator()  # Re-instantiate if needed
                            jmx_content, csv_content = generator.generate_jmx(
                                swagger_endpoints=st.session_state.swagger_endpoints,
                                mappings=st.session_state.mappings,
                                thread_group_users=thread_group_users,
                                ramp_up_time=ramp_up_time,
                                loop_count=loop_count,
                                scenario_plan=scenario_plan,
                                database_connector=st.session_state.db_connector,
                                db_tables_schema=st.session_state.db_tables_schema
                            )

                            # Suppress JMX content display, only offer download
                            st.subheader("JMeter Script Generated!")
                            # st.code(jmx_content, language="xml") # Commented out to hide from UI

                            # Add download button for JMX
                            st.download_button(
                                label="Download JMX Script",
                                data=jmx_content,
                                file_name="generated_test_plan.jmx",
                                mime="application/xml"
                            )

                            if csv_content:
                                st.subheader("Data CSV Generated!")
                                # st.code(csv_content, language="csv") # Commented out to hide from UI
                                st.download_button(
                                    label="Download Data CSV",
                                    data=csv_content,
                                    file_name="data.csv",
                                    mime="text/csv"
                                )
                                st.info(
                                    "Place 'data.csv' in the same directory as the '.jmx' file when running JMeter.")

                            st.success("JMeter script generated successfully!")
                        else:
                            st.warning(
                                "JMeter script generation skipped due to previous LLM error or empty scenario. Using default JMeter generation approach.")
                            # Fallback to a basic JMeter script generation if LLM failed to provide a plan
                            # This part is rudimentary and can be enhanced if LLM fails consistently
                            st.info(
                                "Generating a basic JMeter script based on available endpoints and mappings (without AI scenario planning).")
                            try:
                                # Re-using a simplified generation if scenario_plan is None
                                generator = JMeterScriptGenerator()
                                jmx_content_fallback, csv_content_fallback = generator.generate_jmx(
                                    swagger_endpoints=st.session_state.swagger_endpoints,
                                    mappings=st.session_state.mappings,
                                    thread_group_users=thread_group_users,
                                    ramp_up_time=ramp_up_time,
                                    loop_count=loop_count,
                                    scenario_plan=None,  # Explicitly tell generator to use a simple approach
                                    database_connector=st.session_state.db_connector,
                                    db_tables_schema=st.session_state.db_tables_schema
                                )
                                st.download_button(
                                    label="Download Basic JMX Script (Fallback)",
                                    data=jmx_content_fallback,
                                    file_name="generated_test_plan_fallback.jmx",
                                    mime="application/xml"
                                )
                                if csv_content_fallback:
                                    st.download_button(
                                        label="Download Data CSV (Fallback)",
                                        data=csv_content_fallback,
                                        file_name="data_fallback.csv",
                                        mime="text/csv"
                                    )
                                st.success("Basic JMeter script generated successfully as a fallback!")
                            except Exception as fallback_e:
                                st.error(f"Error generating fallback JMeter script: {str(fallback_e)}")
                                logger.exception("Error generating fallback JMeter script")


                    except Exception as e:
                        st.error(f"Error during script generation: {str(e)}")
                        logger.exception("Error during script generation")

    # Data Mapping Section (kept for structure)
    if st.session_state.swagger_endpoints and st.session_state.db_tables:
        st.header("🔄 Data Mapping")

        if st.button("Generate AI Mapping Suggestions", key="generate_mapping_btn"):
            with st.spinner("Generating mapping suggestions..."):
                tables_schema = {}
                for table in st.session_state.db_tables:
                    schema = st.session_state.db_connector.get_table_schema(table)
                    tables_schema[table] = schema
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
            else:
                st.info("No mappings suggested.")

    # Database Preview Section
    if 'db_connector' in st.session_state and st.session_state.db_tables:
        st.header("📊 Database Preview")
        selected_table = st.selectbox("Select table to preview", st.session_state.db_tables, key="db_preview_table")
        if selected_table:
            preview_df = st.session_state.db_connector.preview_data(selected_table)
            if not preview_df.empty:
                st.dataframe(preview_df)
            else:
                st.info("No data found in selected table")


if __name__ == "__main__":
    main()
