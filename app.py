import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, field  # Keep dataclass and field for other @dataclass uses
import os
import sys
import re  # Import regex module
from urllib.parse import urlparse  # Import urlparse for parsing base URL
import uuid  # For UUID generation
import time  # For timestamp generation

# FIX: Increase the string conversion limit for large integers.
sys.set_int_max_str_digits(0)  # 0 means unlimited

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Assuming these are available and correct from the 'utils' directory
from utils.jmeter_generator import JMeterScriptGenerator
from utils.database_connector import DatabaseConnector, DatabaseConfig
from utils.swagger_parser import SwaggerParser, SwaggerEndpoint  # IMPORT from utils
from utils.data_mapper import DataMapper  # Import DataMapper

# Configure logging to DEBUG level for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# SwaggerEndpoint and SwaggerParser classes are now imported from utils.swagger_parser
# No local definition needed here anymore.


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
                               existing_mappings: Dict[str, Dict[str, Any]],  # Changed type hint
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
        if ep.method == "POST" or ep.method == "PUT" or ep.method == "PATCH":
            if ep.consumes:
                headers_summary = f" (Common headers: Content-Type: {ep.consumes[0]})"
            else:
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

    Existing Data Mappings (parameter -> source: "...", value: "..."):
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
    if 'db_sampled_data' not in st.session_state:  # New: Store sampled data
        st.session_state.db_sampled_data = {}
    if 'mappings' not in st.session_state:  # Stores detailed mapping info
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
        st.session_state.include_scenario_assertions = True  # Default to True
    if 'test_plan_name' not in st.session_state:
        st.session_state.test_plan_name = "Web Application Performance Test"  # Default
    if 'thread_group_name' not in st.session_state:
        st.session_state.thread_group_name = "Users"  # Default
    if 'select_all_endpoints' not in st.session_state:
        st.session_state.select_all_endpoints = False  # New: for select all checkbox
    if 'jmx_content_download' not in st.session_state:  # New: for persistent download
        st.session_state.jmx_content_download = None
    if 'csv_content_download' not in st.session_state:  # New: for persistent download
        st.session_state.csv_content_download = None
    if 'mapping_metadata_download' not in st.session_state:  # New: for mapping metadata
        st.session_state.mapping_metadata_download = None
    if 'full_swagger_spec_download' not in st.session_state:  # New: for full swagger spec download
        st.session_state.full_swagger_spec_download = None

    # New Auth State
    if 'enable_auth_flow' not in st.session_state:
        st.session_state.enable_auth_flow = False
    if 'auth_login_endpoint_path' not in st.session_state:
        st.session_state.auth_login_endpoint_path = "/user/login"  # Updated example
    if 'auth_login_method' not in st.session_state:
        st.session_state.auth_login_method = "POST"
    if 'auth_login_username_param' not in st.session_state:  # For dynamic login
        st.session_state.auth_login_username_param = "username"
    if 'auth_login_password_param' not in st.session_state:  # For dynamic login
        st.session_state.auth_login_password_param = "password"
    if 'auth_login_body_template' not in st.session_state:  # Use template for dynamic body
        st.session_state.auth_login_body_template = '{"username": "${csv_users_username}", "password": "${csv_users_password}"}'
    if 'auth_token_json_path' not in st.session_state:
        st.session_state.auth_token_json_path = "$.access_token"
    if 'auth_header_name' not in st.session_state:
        st.session_state.auth_header_name = "Authorization"
    if 'auth_header_prefix' not in st.session_state:
        st.session_state.auth_header_prefix = "Bearer "

    # Database connection parameters for MySQL/PostgreSQL
    if 'db_host' not in st.session_state:
        st.session_state.db_host = ""
    if 'db_user' not in st.session_state:
        st.session_state.db_user = ""
    if 'db_password' not in st.session_state:
        st.session_state.db_password = ""
    if 'db_name' not in st.session_state:
        st.session_state.db_name = ""
    if 'db_type_selected' not in st.session_state:  # Keep track of selected DB type
        st.session_state.db_type_selected = "SQLite"  # Default

    # Helper function to recursively build JSON body based on schema and mappings
    def _build_recursive_json_body(schema: Dict[str, Any], endpoint_key: str,
                                   current_path_segments: List[str],
                                   extracted_vars: Dict[str, str]) -> Any:
        """
        Recursively builds a JSON body (or part of it) based on schema definitions,
        integrating mappings and extracted variables.
        Ensures required fields are populated with appropriate dummy values if no specific mapping exists.
        """
        logger.debug(
            f"Entering _build_recursive_json_body for path: {'.'.join(current_path_segments)}, schema: {schema}")

        if not isinstance(schema, dict):
            logger.debug(f"Schema is not a dictionary: {schema}. Returning as is.")
            return schema

        schema_type = schema.get('type', 'object')
        required_properties = schema.get('required', [])  # Get list of required properties

        if schema_type == 'object':
            body_obj = {}
            properties = schema.get('properties', {})

            # Iterate through all properties defined in the schema
            for prop_name, prop_details in properties.items():
                full_param_name = ".".join(current_path_segments + [prop_name])

                # Priority 1: Extracted variables from previous responses
                if prop_name.lower() in extracted_vars:
                    body_obj[prop_name] = extracted_vars[prop_name.lower()]
                    logger.debug(
                        f"Prop '{full_param_name}' populated from extracted var: {extracted_vars[prop_name.lower()]}")
                    continue

                # Priority 2: Explicit mappings from DataMapper (DB, Generated, Static)
                mapping_info = st.session_state.mappings.get(endpoint_key, {}).get(full_param_name)

                if mapping_info:
                    # For CSV variables or generated functions, embed as string literal for JMeter to resolve
                    if isinstance(mapping_info['value'], str) and mapping_info['value'].startswith("${") and \
                            mapping_info['value'].endswith("}"):
                        body_obj[prop_name] = mapping_info['value']
                        logger.debug(
                            f"Prop '{full_param_name}' populated from mapping (JMeter var): {mapping_info['value']}")
                    else:  # Directly use the mapped value, attempt type conversion if needed
                        target_type = prop_details.get('type', mapping_info.get('type', 'string'))
                        if target_type == 'integer':
                            try:
                                body_obj[prop_name] = int(mapping_info['value'])
                            except (ValueError, TypeError):
                                body_obj[prop_name] = mapping_info['value']
                        elif target_type == 'boolean':
                            body_obj[prop_name] = str(mapping_info['value']).lower() == 'true'
                        else:
                            body_obj[prop_name] = mapping_info['value']
                        logger.debug(
                            f"Prop '{full_param_name}' populated from mapping (direct value): {mapping_info['value']}")
                    continue  # Move to next property after populating from mapping

                # Priority 3: Recursively build for nested objects/arrays, or generate dummy for required primitives
                if prop_details.get('type') == 'object':
                    body_obj[prop_name] = _build_recursive_json_body(
                        prop_details, endpoint_key, current_path_segments + [prop_name], extracted_vars
                    )
                    logger.debug(f"Prop '{full_param_name}' populated by recursive call (object).")
                elif prop_details.get('type') == 'array':
                    if 'items' in prop_details:
                        array_item_path_segments = current_path_segments + [prop_name, "_item"]
                        item_mapping_info = st.session_state.mappings.get(endpoint_key, {}).get(
                            ".".join(array_item_path_segments[:-1]))

                        if item_mapping_info and isinstance(item_mapping_info['value'], list):
                            body_obj[prop_name] = item_mapping_info['value']
                            logger.debug(f"Prop '{full_param_name}' (array) populated from explicit list mapping.")
                        else:
                            # Generate one item for the array as a sample
                            body_obj[prop_name] = [_build_recursive_json_body(
                                prop_details['items'], endpoint_key, array_item_path_segments, extracted_vars
                            )]
                            logger.debug(f"Prop '{full_param_name}' (array) populated by recursive call for item.")
                    else:
                        body_obj[prop_name] = []  # Empty array if no items schema
                        logger.debug(f"Prop '{full_param_name}' (array) populated as empty (no item schema).")
                else:  # Primitive type (string, integer, boolean, number etc.)
                    # Always assign a dummy value if not populated by extracted vars or explicit mappings,
                    # to ensure complete request body for required objects/parameters.
                    prop_type = prop_details.get('type', 'string')
                    if prop_type == 'string':
                        body_obj[prop_name] = f"dummy_{prop_name}"
                    elif prop_type == 'integer':
                        body_obj[prop_name] = 789
                    elif prop_type == 'boolean':
                        body_obj[prop_name] = True
                    elif prop_type == 'number':
                        body_obj[prop_name] = 789.0
                    else:
                        body_obj[prop_name] = "<<UNSUPPORTED_PRIMITIVE_TYPE>>"
                    logger.debug(
                        f"Prop '{full_param_name}' (primitive, no mapping) populated with dummy: {body_obj[prop_name]}")

            return body_obj

        elif schema_type == 'array':
            logger.debug(f"Schema is an array. Path: {'.'.join(current_path_segments)}")
            if 'items' in schema:
                # Check for a direct mapping of the root array (e.g., if the entire body is mapped to a CSV variable for an array)
                root_array_mapping_info = st.session_state.mappings.get(endpoint_key, {}).get(
                    ".".join(current_path_segments))
                if root_array_mapping_info and isinstance(root_array_mapping_info['value'], list):
                    logger.debug(
                        f"Root array body populated from explicit list mapping: {root_array_mapping_info['value']}")
                    return root_array_mapping_info['value']
                else:
                    # Generate a single item for the array as a sample based on its item schema
                    generated_item = _build_recursive_json_body(
                        schema['items'], endpoint_key, current_path_segments + ["_item"], extracted_vars
                    )
                    logger.debug(f"Root array body populated with one recursive item: {generated_item}")
                    return [generated_item]
            else:
                logger.debug("Root array body populated as empty (no item schema).")
                return []

                # Handle primitive types directly at the current level (e.g., if body is just a string or integer)
        elif schema_type in ['string', 'integer', 'boolean', 'number']:
            logger.debug(f"Schema is a primitive type: {schema_type}. Path: {'.'.join(current_path_segments)}")
            mapping_info = st.session_state.mappings.get(endpoint_key, {}).get(".".join(current_path_segments))
            if mapping_info:
                if schema_type == 'integer':
                    try:
                        return int(mapping_info['value'])
                    except (ValueError, TypeError):
                        return mapping_info['value']
                elif schema_type == 'boolean':
                    return str(mapping_info['value']).lower() == 'true'
                elif schema_type == 'number':
                    try:
                        return float(mapping_info['value'])
                    except (ValueError, TypeError):
                        return mapping_info['value']
                else:  # string
                    return mapping_info['value']
            else:
                # Provide a default/dummy for a root-level primitive body
                if schema_type == 'string':
                    return "dummy_root_string_body"
                elif schema_type == 'integer':
                    return 789
                elif schema_type == 'boolean':
                    return True
                elif schema_type == 'number':
                    return 789.0
                else:
                    return "<<UNSUPPORTED_ROOT_PRIMITIVE_TYPE>>"
        else:
            logger.warning(f"Unsupported schema type at root: {schema_type}. Path: {'.'.join(current_path_segments)}")
            return "<<UNSUPPORTED_BODY_TYPE>>"

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

                        # Clear scenario and generated outputs on new Swagger fetch for isolation
                        st.session_state.selected_endpoint_keys = []
                        st.session_state.scenario_requests_configs = []
                        st.session_state.jmx_content_download = None
                        st.session_state.csv_content_download = None
                        st.session_state.mapping_metadata_download = None
                        st.session_state.full_swagger_spec_download = None  # Clear full spec on new fetch
                        st.session_state.mappings = {}  # Clear mappings too
                        st.success(f"Found {len(endpoints)} API endpoints.")
                    except Exception as e:
                        st.error(f"Failed to fetch Swagger: {str(e)}")
                        st.session_state.swagger_endpoints = []
                        st.session_state.selected_endpoint_keys = []
                        st.session_state.scenario_requests_configs = []
                        st.session_state.jmx_content_download = None
                        st.session_state.csv_content_download = None
                        st.session_state.mapping_metadata_download = None
                        st.session_state.full_swagger_spec_download = None
                        st.session_state.mappings = {}
            else:
                st.warning("Please enter a Swagger JSON URL.")

        if st.session_state.swagger_endpoints:
            st.subheader("API Endpoints")
            for endpoint in st.session_state.swagger_endpoints[:5]:
                st.code(f"{endpoint.method} {endpoint.path}")
            if len(st.session_state.swagger_endpoints) > 5:
                st.info(f"... and {len(st.session_state.swagger_endpoints) - 5} more endpoints")

            # Make the full swagger spec available for download after fetch
            if st.session_state.swagger_parser and st.session_state.swagger_parser.swagger_data:
                st.session_state.full_swagger_spec_download = json.dumps(
                    st.session_state.swagger_parser.get_full_swagger_spec(), indent=2
                )

    with col2:
        st.header("🗄️ Database Configuration")

        st.session_state.db_type_selected = st.selectbox(
            "Database Type",
            ["SQLite", "MySQL", "PostgreSQL"],
            key="db_type_select",
            index=["SQLite", "MySQL", "PostgreSQL"].index(st.session_state.db_type_selected)
        )

        db_connection_params_changed = False
        if st.session_state.db_type_selected == "SQLite":
            db_file_path = st.text_input(
                "SQLite Database File Path",
                value="database/petstore.db",
                help="Path to your SQLite database file (e.g., database/petstore.db)"
            )
            current_config_str = f"sqlite:///{db_file_path}"
            db_config = DatabaseConfig(connection_string=current_config_str, db_type="sqlite", file_path=db_file_path)

            # Check if relevant config changed
            if db_file_path != st.session_state.get('last_db_file_path', ''):
                db_connection_params_changed = True
                st.session_state.last_db_file_path = db_file_path

        else:  # MySQL or PostgreSQL
            st.warning(
                f"Note: Live connection to {st.session_state.db_type_selected} is simulated in this environment. Schema and data will be dummy.")
            db_host_input = st.text_input("DB Host", value=st.session_state.db_host, key="db_host_input")
            db_user_input = st.text_input("DB Username", value=st.session_state.db_user, key="db_user_input")
            db_password_input = st.text_input("DB Password", type="password", value=st.session_state.db_password,
                                              key="db_password_input")
            db_name_input = st.text_input("DB Name", value=st.session_state.db_name, key="db_name_input")

            db_config = DatabaseConfig(
                db_type=st.session_state.db_type_selected.lower(),
                host=db_host_input,
                username=db_user_input,
                password=db_password_input,
                database=db_name_input
            )

            # Check if relevant config changed
            if (db_host_input != st.session_state.db_host or
                    db_user_input != st.session_state.db_user or
                    db_password_input != st.session_state.db_password or
                    db_name_input != st.session_state.db_name):
                db_connection_params_changed = True
                st.session_state.db_host = db_host_input
                st.session_state.db_user = db_user_input
                st.session_state.db_password = db_password_input
                st.session_state.db_name = db_name_input

        if st.button("Connect Database",
                     key="connect_db") or db_connection_params_changed:  # Auto-reconnect if params change
            if st.session_state.db_type_selected == "SQLite" and not db_file_path:
                st.error("Please enter a SQLite database file path.")
                st.session_state.db_tables = []
                st.session_state.db_connector = None
                st.session_state.db_tables_schema = {}
                st.session_state.db_sampled_data = {}
                st.session_state.mappings = {}  # Clear mappings
            elif st.session_state.db_type_selected != "SQLite" and not (
                    db_config.host and db_config.username and db_config.database):
                st.error(
                    f"Please provide Host, Username, and Database for {st.session_state.db_type_selected} connection.")
                st.session_state.db_tables = []
                st.session_state.db_connector = None
                st.session_state.db_tables_schema = {}
                st.session_state.db_sampled_data = {}
                st.session_state.mappings = {}  # Clear mappings
            else:
                with st.spinner("Connecting to database..."):
                    try:
                        connector = DatabaseConnector(db_config)
                        if connector.connect():
                            st.session_state.db_connector = connector
                            tables = connector.get_tables()
                            st.session_state.db_tables = tables

                            tables_schema = {}
                            sampled_data = {}
                            for table in tables:
                                schema = connector.get_table_schema(table)
                                tables_schema[table] = schema
                                # Sample 3 rows for each table
                                sampled_data[table] = connector.preview_data(table, limit=3)

                            st.session_state.db_tables_schema = tables_schema
                            st.session_state.db_sampled_data = sampled_data  # Store sampled data

                            # Clear previous mappings and generated output on new DB connection
                            st.session_state.mappings = {}
                            st.session_state.jmx_content_download = None
                            st.session_state.csv_content_download = None
                            st.session_state.mapping_metadata_download = None

                            st.success(f"Connected to {connector.config.db_type.upper()}! Found {len(tables)} tables.")

                            # For SQLite, if dummy DB does not exist, offer to create
                            if st.session_state.db_type_selected == "SQLite" and not os.path.exists(db_file_path):
                                if st.button("Create Dummy SQLite DB", key="create_dummy_db_on_connect"):
                                    try:
                                        os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
                                        conn = sqlite3.connect(db_file_path)
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
                                                           TEXT,
                                                           role_id
                                                           INTEGER
                                                       );
                                                       """)
                                        cursor.execute("""
                                                       CREATE TABLE IF NOT EXISTS orders
                                                       (
                                                           order_id
                                                           INTEGER
                                                           PRIMARY
                                                           KEY,
                                                           user_id
                                                           INTEGER,
                                                           status
                                                           TEXT
                                                       );
                                                       """)
                                        cursor.execute("""
                                                       CREATE TABLE IF NOT EXISTS inventory_items
                                                       (
                                                           item_id
                                                           INTEGER
                                                           PRIMARY
                                                           KEY,
                                                           product_id
                                                           INTEGER,
                                                           quantity
                                                           INTEGER
                                                       );
                                                       """)
                                        cursor.execute("""
                                                       CREATE TABLE IF NOT EXISTS roles
                                                       (
                                                           id
                                                           INTEGER
                                                           PRIMARY
                                                           KEY,
                                                           role_name
                                                           TEXT
                                                       );
                                                       """)
                                        cursor.execute(
                                            "INSERT INTO pets (id, name, status, tags) VALUES (1, 'Buddy', 'available', 'dog,friendly');")
                                        cursor.execute(
                                            "INSERT INTO pets (id, name, status, tags) VALUES (2, 'Whiskers', 'pending', 'cat');")
                                        cursor.execute(
                                            "INSERT INTO users (id, username, password, email, role_id) VALUES (101, 'testuser', 'testpass', 'test@example.com', 1);")
                                        cursor.execute(
                                            "INSERT INTO users (id, username, password, email, role_id) VALUES (102, 'user2', 'pass2', 'user2@example.com', 2);")
                                        cursor.execute(
                                            "INSERT INTO orders (order_id, user_id, status) VALUES (1001, 101, 'pending');")
                                        cursor.execute(
                                            "INSERT INTO orders (order_id, user_id, status) VALUES (1002, 102, 'completed');")
                                        cursor.execute(
                                            "INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (1, 1, 50);")
                                        cursor.execute(
                                            "INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (2, 2, 120);")
                                        cursor.execute("INSERT INTO roles (id, role_name) VALUES (1, 'Admin');")
                                        cursor.execute("INSERT INTO roles (id, role_name) VALUES (2, 'User');")
                                        conn.commit()
                                        conn.close()
                                        st.success(f"Dummy SQLite database created at {db_file_path} with sample data.")
                                        st.rerun()  # Rerun to re-connect and load data
                                    except Exception as ex:
                                        st.error(f"Error creating dummy DB: {ex}")

                        else:
                            st.error("Failed to connect to database. Check connection parameters and logs.")
                            st.session_state.db_tables = []
                            st.session_state.db_connector = None
                            st.session_state.db_tables_schema = {}
                            st.session_state.db_sampled_data = {}
                            st.session_state.mappings = {}
                    except Exception as e:
                        st.error(f"Database connection error: {str(e)}")
                        st.session_state.db_tables = []
                        st.session_state.db_connector = None
                        st.session_state.db_tables_schema = {}
                        st.session_state.db_sampled_data = {}
                        st.session_state.mappings = {}

        if st.session_state.db_tables:
            st.subheader("Database Tables")
            for table in st.session_state.db_tables:
                st.code(table)

            st.subheader("Sampled Data (First 3 Rows)")
            for table_name, df in st.session_state.db_sampled_data.items():
                if not df.empty:
                    st.write(f"**Table: {table_name}**")
                    st.dataframe(df)
                else:
                    st.info(f"No data sampled for table: {table_name}")

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
            value="Generate a JMeter script that covers all available API endpoints. For each endpoint, use mapped database fields if available, otherwise generate dummy data. Aim for a typical user flow involving GET, POST, PUT, and DELETE operations where appropriate. If possible, include an authentication flow by logging in first and using the obtained token.",
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
                    # Recalculate mappings including sampled data logic
                    st.session_state.mappings = DataMapper.suggest_mappings(
                        st.session_state.swagger_endpoints,
                        st.session_state.db_tables_schema,
                        st.session_state.db_sampled_data  # Pass sampled data here
                    )

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
        st.session_state.test_plan_name = st.text_input(
            "Test Plan Name",
            value=st.session_state.test_plan_name,
            key="test_plan_name_input",
            help="Name for the overall JMeter Test Plan."
        )
    with col2:
        st.session_state.thread_group_name = st.text_input(
            "Thread Group Name",
            value=st.session_state.thread_group_name,
            key="thread_group_name_input",
            help="Name for the main Thread Group (Users)."
        )
    with col3:
        num_users = st.number_input(
            "Number of Concurrent Users (Threads)",
            min_value=1,
            value=10,
            step=1,
            key="num_users_input"
        )
    col4, col5 = st.columns(2)
    with col4:
        ramp_up_time = st.number_input(
            "Ramp-up Time (seconds)",
            min_value=0,
            value=30,
            step=5,
            key="ramp_up_time_input"
        )
    with col5:
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
            "Include Response Assertions in Scenario (Status 200)",
            value=st.session_state.include_scenario_assertions,
            key="include_scenario_assertions_checkbox",
            help="If checked, a 'Response Code 200' assertion will be added to each request in the scenario."
        )

    st.header("2.2 Authentication Flow Configuration")
    st.session_state.enable_auth_flow = st.checkbox(
        "Enable Authentication Flow",
        value=st.session_state.enable_auth_flow,
        key="enable_auth_flow_checkbox"
    )

    if st.session_state.enable_auth_flow:
        st.session_state.auth_login_endpoint_path = st.text_input(
            "Login API Endpoint Path",
            value=st.session_state.auth_login_endpoint_path,
            key="auth_login_endpoint_path_input",
            help="e.g., /auth/login. Must exist in Swagger spec."
        )
        st.session_state.auth_login_method = st.selectbox(
            "Login API Method",
            options=["POST", "GET", "PUT"],
            index=["POST", "GET", "PUT"].index(st.session_state.auth_login_method),
            key="auth_login_method_select"
        )

        st.markdown("**Login Request Body Configuration**")
        st.info("You can use CSV variables (e.g., `${csv_users_username}`) if your DB has user credentials.")

        st.session_state.auth_login_username_param = st.text_input(
            "Username Parameter Name (in body)",
            value=st.session_state.auth_login_username_param,
            key="auth_login_username_param_input",
            help="Name of the field in the login request body for username (e.g., 'username')."
        )
        st.session_state.auth_login_password_param = st.text_input(
            "Password Parameter Name (in body)",
            value=st.session_state.auth_login_password_param,
            key="auth_login_password_param_input",
            help="Name of the field in the login request body for password (e.g., 'password')."
        )
        st.session_state.auth_login_body_template = st.text_area(
            "Login Request Body Template (JSON)",
            value=st.session_state.auth_login_body_template,
            key="auth_login_body_template_textarea",
            height=100,
            help="JSON body for login request. Use JMeter variables like ${csv_users_username}."
        )

        st.session_state.auth_token_json_path = st.text_input(
            "Auth Token JSON Path (from Login Response)",
            value=st.session_state.auth_token_json_path,
            key="auth_token_json_path_input",
            help="JSONPath expression to extract token, e.g., $.access_token"
        )
        st.session_state.auth_header_name = st.text_input(
            "Authorization Header Name",
            value=st.session_state.auth_header_name,
            key="auth_header_name_input",
            help="e.g., Authorization"
        )
        st.session_state.auth_header_prefix = st.text_input(
            "Authorization Header Prefix",
            value=st.session_state.auth_header_prefix,
            key="auth_header_prefix_input",
            help="e.g., Bearer "
        )

    st.header("3. Select Endpoints for Scenario")
    if st.session_state.swagger_endpoints:
        endpoint_options = [f"{ep.method} {ep.path}" for ep in st.session_state.swagger_endpoints]

        # New: Select All Endpoints checkbox
        select_all_toggle = st.checkbox("Select All Endpoints", key="select_all_endpoints_checkbox")

        # Determine default selection based on toggle
        default_selected_endpoints = []
        if select_all_toggle:
            default_selected_endpoints = endpoint_options
        else:
            default_selected_endpoints = st.session_state.selected_endpoint_keys

        selected_endpoint_keys = st.multiselect(
            "Select API Endpoints for your scenario (order reflects execution sequence)",
            options=endpoint_options,
            default=default_selected_endpoints,  # Use the dynamic default
            key="endpoint_selector"
        )

        # Recalculate scenario configurations when endpoints change or "Refresh" is clicked
        if selected_endpoint_keys != st.session_state.selected_endpoint_keys or st.button(
                "Refresh Scenario Configuration", key="refresh_scenario_btn"):
            st.session_state.selected_endpoint_keys = selected_endpoint_keys
            new_scenario_configs = []

            # Recalculate mappings if not already done or if DB/Swagger changed
            st.session_state.mappings = DataMapper.suggest_mappings(
                st.session_state.swagger_endpoints,
                st.session_state.db_tables_schema,
                st.session_state.db_sampled_data
            )

            extracted_variables_map = {}  # {parameter_name_lower: JMeter_variable_name}

            # --- Add Login Request if authentication is enabled ---
            if st.session_state.enable_auth_flow:
                # For login, still rely on the endpoint object for basic path/method lookup
                login_endpoint = next((ep for ep in st.session_state.swagger_endpoints if
                                       ep.path == st.session_state.auth_login_endpoint_path and ep.method == st.session_state.auth_login_method),
                                      None)
                if login_endpoint:
                    # Login body template is handled separately
                    resolved_login_body = st.session_state.auth_login_body_template

                    login_request_config = {
                        "endpoint_key": f"{login_endpoint.method} {login_endpoint.path}",
                        "name": "Login_Request",
                        "method": login_endpoint.method,
                        "path": login_endpoint.path,
                        "parameters": {},
                        "headers": {"Content-Type": "application/json"},
                        "body": resolved_login_body,
                        "assertions": [{"type": "Response Code", "value": "200"}],
                        "json_extractors": [
                            {
                                "json_path_expr": st.session_state.auth_token_json_path,
                                "var_name": "authToken"
                            }
                        ],
                        "think_time": 0
                    }
                    new_scenario_configs.append(login_request_config)
                    extracted_variables_map["authtoken"] = "${authToken}"
                else:
                    st.warning(
                        f"Login endpoint {st.session_state.auth_login_method} {st.session_state.auth_login_endpoint_path} not found in Swagger spec. Authentication flow might not work as expected.")

            # For selected endpoints in the main scenario
            full_swagger_spec = st.session_state.swagger_parser.get_full_swagger_spec()
            base_path_from_swagger = full_swagger_spec.get('basePath', '')

            for ep_key in selected_endpoint_keys:
                method_str, path_str = ep_key.split(' ', 1)
                method_key = method_str.lower()

                resolved_endpoint_data = full_swagger_spec.get('paths', {}).get(path_str, {}).get(method_key, {})

                if resolved_endpoint_data:
                    clean_path_name = re.sub(r'[^\w\s-]', '', path_str).replace('/', '_').strip('_')
                    operation_id = resolved_endpoint_data.get('operationId')
                    if operation_id:
                        request_name = f"{method_str}_{operation_id}"
                    else:
                        request_name = f"{method_str}_{clean_path_name}"

                    # Start with the raw path from Swagger
                    jmeter_formatted_path = path_str

                    # Process path parameters and convert to JMeter variables
                    if 'parameters' in resolved_endpoint_data:
                        for param in resolved_endpoint_data['parameters']:
                            if param.get('in') == 'path':
                                param_name = param['name']

                                jmeter_var_for_path_param = f"${{{param_name}}}"  # Default fallback

                                if param_name.lower() in extracted_variables_map:
                                    jmeter_var_for_path_param = extracted_variables_map[param_name.lower()]
                                    logger.debug(
                                        f"Path param '{param_name}' for {ep_key} using extracted variable: {jmeter_var_for_path_param}")
                                else:
                                    mapping_info = st.session_state.mappings.get(ep_key, {}).get(param_name)
                                    if mapping_info and mapping_info['source'] == "DB Sample (CSV)":
                                        jmeter_var_for_path_param = mapping_info['value']
                                        logger.debug(
                                            f"Path param '{param_name}' for {ep_key} using CSV mapping: {jmeter_var_for_path_param}")
                                    else:
                                        logger.warning(
                                            f"Path param '{param_name}' for {ep_key} has no explicit DB/extraction mapping. Using generic JMeter variable: {jmeter_var_for_path_param}")

                                # Replace {paramName} with ${jmeterVarName}
                                jmeter_formatted_path = jmeter_formatted_path.replace(f"{{{param_name}}}",
                                                                                      jmeter_var_for_path_param)

                    # Prepend the base path from Swagger. Ensure no double slashes.
                    final_request_path_for_jmeter = f"{base_path_from_swagger}{jmeter_formatted_path}"
                    final_request_path_for_jmeter = final_request_path_for_jmeter.replace('//', '/')

                    request_config = {
                        "endpoint_key": ep_key,
                        "name": request_name,
                        "method": method_str,
                        "path": final_request_path_for_jmeter,  # This is the crucial line
                        "parameters": {},  # Query parameters are handled below
                        "headers": {},
                        "body": None,
                        "assertions": [],
                        "json_extractors": [],
                        "think_time": 0
                    }

                    # Add Authorization header if auth flow is enabled and it's not the login request itself
                    if st.session_state.enable_auth_flow and ep_key != f"{st.session_state.auth_login_method} {st.session_state.auth_login_endpoint_path}":
                        if "authtoken" in extracted_variables_map:
                            request_config["headers"][
                                st.session_state.auth_header_name] = f"{st.session_state.auth_header_prefix}{extracted_variables_map['authtoken']}"
                        else:
                            st.warning(
                                f"Authentication flow enabled but auth token not found for {ep_key}. Check login configuration.")

                    # Process URL/Query parameters
                    if 'parameters' in resolved_endpoint_data:
                        for param in resolved_endpoint_data['parameters']:
                            if param.get('in') == 'query':
                                if param['name'].lower() in extracted_variables_map:
                                    request_config['parameters'][param['name']] = extracted_variables_map[
                                        param['name'].lower()]
                                else:
                                    mapping_info = st.session_state.mappings.get(ep_key, {}).get(param['name'])
                                    if mapping_info:
                                        request_config['parameters'][param['name']] = mapping_info['value']
                                    else:
                                        request_config['parameters'][param['name']] = "<<NO_MATCH_FOUND>>"

                    # Process Request Body
                    if method_key in ["post", "put", "patch"]:
                        # Determine Content-Type
                        content_type_header = "application/json"
                        if resolved_endpoint_data.get('consumes'):  # Swagger 2.0
                            content_type_header = resolved_endpoint_data['consumes'][0]
                        elif 'requestBody' in resolved_endpoint_data and 'content' in resolved_endpoint_data[
                            'requestBody']:  # OpenAPI 3.0
                            first_content_type = next(iter(resolved_endpoint_data['requestBody'].get('content', {})),
                                                      None)
                            if first_content_type:
                                content_type_header = first_content_type
                        request_config["headers"]["Content-Type"] = content_type_header

                        request_body_schema = None
                        if 'requestBody' in resolved_endpoint_data:  # OpenAPI 3.0 structure
                            content_types = resolved_endpoint_data['requestBody'].get('content', {})
                            if content_types:
                                # Prioritize application/json, then the first available content type
                                if 'application/json' in content_types:
                                    request_body_schema = content_types['application/json'].get('schema')
                                else:
                                    request_body_schema = next(iter(content_types.values())).get('schema')
                        else:  # Swagger 2.0 structure (body parameter)
                            for param in resolved_endpoint_data.get('parameters', []):
                                if param.get('in') == 'body' and 'schema' in param:
                                    request_body_schema = param['schema']
                                    break

                        logger.debug(
                            f"Resolved request_body_schema for {ep_key}: {json.dumps(request_body_schema, indent=2)}")

                        if request_body_schema:
                            try:
                                generated_body = _build_recursive_json_body(
                                    request_body_schema,  # Pass the resolved schema directly
                                    ep_key,
                                    [],
                                    extracted_variables_map
                                )
                                if isinstance(generated_body, (dict, list, int, float, bool)):
                                    request_config["body"] = json.dumps(generated_body, indent=2)
                                else:
                                    request_config["body"] = str(generated_body)

                            except Exception as e:
                                logger.error(f"Error building recursive JSON body for {ep_key}: {e}", exc_info=True)
                                request_config[
                                    "body"] = "{\n  \"message\": \"Error building dynamic body for bodySchema from full spec\"\n}"
                        else:
                            request_config[
                                "body"] = "{\n  \"message\": \"auto-generated dummy body (no schema found in full spec)\"\n}"

                    # Add standard 200 assertion if enabled
                    if st.session_state.include_scenario_assertions:
                        request_config['assertions'].append({"type": "Response Code", "value": "200"})

                    # --- Correlation: Auto-add JSON Extractors for response IDs ---
                    # Get response schemas directly from resolved_endpoint_data
                    for status_code, response_obj in resolved_endpoint_data.get('responses', {}).items():
                        if status_code.startswith('2'):  # Successful response
                            resp_schema = None
                            if 'schema' in response_obj:  # Swagger 2.0
                                resp_schema = response_obj['schema']
                            elif 'content' in response_obj:  # OpenAPI 3.0
                                # Get the first content type's schema
                                first_content_type = next(iter(response_obj['content']), None)
                                if first_content_type:
                                    resp_schema = response_obj['content'][first_content_type].get('schema')

                            if resp_schema and 'properties' in resp_schema:
                                for prop_name, prop_details in resp_schema['properties'].items():
                                    if (prop_name.lower() == 'id' and prop_details.get('type') in ['string',
                                                                                                   'integer']) or \
                                            (prop_name.lower() == 'username' and prop_details.get('type') == 'string'):
                                        var_base_name = prop_name.replace('_', '').lower()
                                        correlated_var_name = f"{operation_id or clean_path_name}{var_base_name.capitalize()}"
                                        request_config['json_extractors'].append({
                                            "json_path_expr": f"$.{prop_name}",
                                            "var_name": correlated_var_name
                                        })
                                        extracted_variables_map[prop_name.lower()] = f"${{{correlated_var_name}}}"
                                        logger.info(
                                            f"Detected potential correlation: '{prop_name}' from {ep_key} will be extracted as ${{{correlated_var_name}}}")

                            if resp_schema and resp_schema.get('type') == 'array' and 'items' in resp_schema:
                                item_schema = resp_schema['items']
                                if 'properties' in item_schema:
                                    for prop_name, prop_details in item_schema['properties'].items():
                                        if (prop_name.lower() == 'id' and prop_details.get('type') in ['string',
                                                                                                       'integer']) or \
                                                (prop_name.lower() == 'username' and prop_details.get(
                                                    'type') == 'string'):
                                            var_base_name = prop_name.replace('_', '').lower()
                                            correlated_var_name = f"{operation_id or clean_path_name}First{var_base_name.capitalize()}"
                                            request_config['json_extractors'].append({
                                                "json_path_expr": f"$[0].{prop_name}",
                                                "var_name": correlated_var_name
                                            })
                                            extracted_variables_map[prop_name.lower()] = f"${{{correlated_var_name}}}"
                                            logger.info(
                                                f"Detected array correlation: First '{prop_name}' from {ep_key} will be extracted as ${{{correlated_var_name}}}")

                    new_scenario_configs.append(request_config)
                else:
                    logger.warning(f"Could not find fully resolved endpoint data for {ep_key}. Skipping.")

            st.session_state.scenario_requests_configs = new_scenario_configs
            st.rerun()

        st.markdown("---")
        st.subheader("Selected Endpoints (Auto-Configured)")
        if st.session_state.scenario_requests_configs:
            for i, config in enumerate(st.session_state.scenario_requests_configs):
                st.code(
                    f"Request {i + 1}: {config['method']} {config['path']} (Name: {config['name']})")  # Show new name
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
        if not st.session_state.selected_endpoint_keys and not st.session_state.enable_auth_flow:
            st.error(
                "Please select at least one API request to include in your scenario, or enable authentication flow only.")
            return
        if st.session_state.enable_auth_flow and not st.session_state.auth_login_endpoint_path:
            st.error("Authentication flow is enabled, but no Login API Endpoint Path is provided.")
            return

        st.info("Generating JMeter script... Please wait.")

        try:
            scenario_plan = {"requests": st.session_state.scenario_requests_configs}

            num_users = st.session_state.num_users_input
            ramp_up_time = st.session_state.ramp_up_time_input
            loop_count = st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1

            global_constant_timer_delay = st.session_state.constant_timer_delay_ms if st.session_state.enable_constant_timer else 0

            # Construct sampled data for CSV based on *actual used* mappings
            # This ensures only variables explicitly marked as 'csv_' are included
            csv_data_for_jmeter = {}
            csv_headers = set()

            # Iterate through the `mappings` dictionary which now contains full paths for body parameters
            for endpoint_key, params_map in st.session_state.mappings.items():
                for param_name, mapping_info in params_map.items():
                    if mapping_info['source'] == "DB Sample (CSV)":
                        jmeter_var_name_raw = mapping_info['value'].replace('${', '').replace('}',
                                                                                              '')  # e.g., csv_users_id
                        # Extract original table and column from jmeter_var_name
                        parts = jmeter_var_name_raw.split('_')
                        if len(parts) >= 3 and parts[0] == 'csv':
                            table_name = parts[1]
                            # Reconstruct original column name (can have underscores in DB)
                            column_name = "_".join(parts[2:])

                            if table_name in st.session_state.db_sampled_data and column_name in \
                                    st.session_state.db_sampled_data[table_name].columns:
                                if jmeter_var_name_raw not in csv_data_for_jmeter:
                                    csv_data_for_jmeter[jmeter_var_name_raw] = \
                                    st.session_state.db_sampled_data[table_name][column_name].tolist()
                                    csv_headers.add(jmeter_var_name_raw)
                                else:
                                    # Ensure lists are of same length, pad if necessary for multiple uses of same column
                                    if len(csv_data_for_jmeter[jmeter_var_name_raw]) < len(
                                            st.session_state.db_sampled_data[table_name][column_name]):
                                        csv_data_for_jmeter[jmeter_var_name_raw] = \
                                        st.session_state.db_sampled_data[table_name][column_name].tolist()

            generated_csv_content = None
            if csv_headers and csv_data_for_jmeter:
                csv_headers_list = sorted(list(csv_headers))  # Ensure consistent order
                generated_csv_content = ",".join(csv_headers_list) + "\n"  # Use full JMeter variable names as headers

                max_rows = 0
                if csv_data_for_jmeter:
                    max_rows = max(len(v) for v in csv_data_for_jmeter.values())

                for i in range(max_rows):
                    row_values = []
                    for header_key in csv_headers_list:
                        values = csv_data_for_jmeter.get(header_key, [])
                        row_values.append(str(values[i]) if i < len(values) else "")
                    generated_csv_content += ",".join(row_values) + "\n"

            generator = JMeterScriptGenerator(
                test_plan_name=st.session_state.test_plan_name,
                thread_group_name=st.session_state.thread_group_name
            )

            # Parse base URL components from the swagger_url
            parsed_url = urlparse(swagger_url)
            protocol = parsed_url.scheme
            domain = parsed_url.hostname
            port = parsed_url.port if parsed_url.port else ""
            # basePath for Swagger 2.0 or just a path segment for OpenAPI 3.0
            base_path = parsed_url.path.rsplit('/', 1)[0] if parsed_url.path.endswith(
                '.json') or parsed_url.path.endswith('.yaml') else parsed_url.path
            if not base_path:  # Ensure base_path is at least "/" if empty
                base_path = "/"

            # Use swagger_parser instance to get the full resolved spec
            full_resolved_swagger_spec = st.session_state.swagger_parser.get_full_swagger_spec()

            jmx_content, _ = generator.generate_jmx(
                app_base_url=swagger_url,  # Original URL for consistency
                thread_group_users=num_users,
                ramp_up_time=ramp_up_time,
                loop_count=loop_count,
                scenario_plan=scenario_plan,
                csv_data_to_include=generated_csv_content,  # Pass pre-generated CSV content
                global_constant_timer_delay=global_constant_timer_delay,
                test_plan_name=st.session_state.test_plan_name,
                thread_group_name=st.session_state.thread_group_name,
                http_defaults_protocol=protocol,
                http_defaults_domain=domain,
                http_defaults_port=port,
                http_defaults_base_path=base_path,
                full_swagger_spec=full_resolved_swagger_spec  # Pass the full resolved spec
            )

            st.session_state.jmx_content_download = jmx_content  # Store for persistence
            st.session_state.csv_content_download = generated_csv_content  # Store for persistence
            st.session_state.mapping_metadata_download = json.dumps(st.session_state.mappings,
                                                                    indent=2)  # Store mapping metadata

            st.success("JMeter script generated successfully!")

        except Exception as e:
            st.error(f"An error occurred during script generation: {e}")
            logger.error(f"Error in main app execution: {e}", exc_info=True)

    # --- Download Links (Rendered Persistently) ---
    st.subheader("Download Generated Files")
    if st.session_state.full_swagger_spec_download:  # New download button for full Swagger spec
        st.download_button(
            label="Download Full Swagger Spec (.json)",
            data=st.session_state.full_swagger_spec_download.encode("utf-8"),
            file_name="full_swagger_spec.json",
            mime="application/json",
            key="download_full_swagger_spec"
        )

    if st.session_state.jmx_content_download:
        st.download_button(
            label="Download JMeter Test Plan (.jmx)",
            data=st.session_state.jmx_content_download.encode("utf-8"),
            file_name="generated_test_plan.jmx",
            mime="application/xml",
            key="download_jmx_final"
        )

    if st.session_state.csv_content_download:
        st.download_button(
            label="Download CSV Data (data.csv)",
            data=st.session_state.csv_content_download.encode("utf-8"),
            file_name="data.csv",
            mime="text/csv",
            key="download_csv_final"
        )
    elif st.session_state.jmx_content_download:  # Only show info if JMX is generated but CSV is not
        st.info("No CSV data was generated for this test plan (no DB mappings used).")
    else:
        st.info("Generate the JMeter script above to enable download links.")

    if st.session_state.mapping_metadata_download:
        st.download_button(
            label="Download Mapping Metadata (.json)",
            data=st.session_state.mapping_metadata_download.encode("utf-8"),
            file_name="mapping_metadata.json",
            mime="application/json",
            key="download_mapping_metadata"
        )
    else:
        st.info("Mapping metadata will be available after script generation.")

    st.markdown("---")
    st.markdown("Developed with ❤️ by Your AI Assistant")


if __name__ == "__main__":
    # Ensure dummy swagger.json and petstore.db exist for initial run
    if not os.path.exists("swagger.json"):
        dummy_swagger_content = """
{
  "swagger": "2.0",
  "info": {
    "description": "This is a sample server Petstore server.  You can find out more about Swagger at [http://swagger.io](http://swagger.io) or on [irc.freenode.net, #swagger](http://swagger.io/irc/).  For this sample, you can use the api key `special-key` to test the authorization filters.",
    "version": "1.0.7",
    "title": "Swagger Petstore",
    "termsOfService": "http://swagger.io/terms/",
    "contact": {
      "email": "apiteam@swagger.io"
    },
    "license": {
      "name": "Apache 2.0",
      "url": "http://www.apache.org/licenses/LICENSE-2.0.html"
    }
  },
  "host": "petstore.swagger.io",
  "basePath": "/v2",
  "tags": [
    {
      "name": "pet",
      "description": "Everything about your Pets",
      "externalDocs": {
        "description": "Find out more",
        "url": "http://swagger.io"
      }
    },
    {
      "name": "store",
      "description": "Access to Petstore orders"
    },
    {
      "name": "user",
      "description": "Operations about user",
      "externalDocs": {
        "description": "Find out more about our store",
        "url": "http://swagger.io"
      }
    }
  ],
  "schemes": [
    "https",
    "http"
  ],
  "paths": {
    "/pet/{petId}/uploadImage": {
      "post": {
        "tags": [
          "pet"
        ],
        "summary": "uploads an image",
        "description": "",
        "operationId": "uploadFile",
        "consumes": [
          "multipart/form-data"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "description": "ID of pet to update",
            "required": true,
            "type": "integer",
            "format": "int64"
          },
          {
            "name": "additionalMetadata",
            "in": "formData",
            "description": "Additional data to pass to server",
            "required": false,
            "type": "string"
          },
          {
            "name": "file",
            "in": "formData",
            "description": "file to upload",
            "required": false,
            "type": "file"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "properties": {
                "code": {
                  "type": "integer",
                  "format": "int32"
                },
                "type": {
                  "type": "string"
                },
                "message": {
                  "type": "string"
                }
              }
            }
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      }
    },
    "/pet": {
      "post": {
        "tags": [
          "pet"
        ],
        "summary": "Add a new pet to the store",
        "description": "",
        "operationId": "addPet",
        "consumes": [
          "application/json",
          "application/xml"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "Pet object that needs to be added to the store",
            "required": true,
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          }
        ],
        "responses": {
          "405": {
            "description": "Invalid input"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      },
      "put": {
        "tags": [
          "pet"
        ],
        "summary": "Update an existing pet",
        "description": "",
        "operationId": "updatePet",
        "consumes": [
          "application/json",
          "application/xml"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "Pet object that needs to be added to the store",
            "required": true,
            "schema": {
              "$ref": "#/definitions/Pet"
            }
          }
        ],
        "responses": {
          "400": {
            "description": "Invalid ID supplied"
          },
          "404": {
            "description": "Pet not found"
          },
          "405": {
            "description": "Validation exception"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      }
    },
    "/pet/findByStatus": {
      "get": {
        "tags": [
          "pet"
        ],
        "summary": "Finds Pets by status",
        "description": "Multiple status values can be provided with comma separated strings",
        "operationId": "findPetsByStatus",
        "produces": [
          "application/json",
          "application/xml"
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
            "description": "successful operation",
            "schema": {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "name",
                  "photoUrls"
                ],
                "properties": {
                  "id": {
                    "type": "integer",
                    "format": "int64"
                  },
                  "category": {
                    "type": "object",
                    "properties": {
                      "id": {
                        "type": "integer",
                        "format": "int64"
                      },
                      "name": {
                        "type": "string"
                      }
                    },
                    "xml": {
                      "name": "Category"
                    }
                  },
                  "name": {
                    "type": "string",
                    "example": "doggie"
                  },
                  "photoUrls": {
                    "type": "array",
                    "xml": {
                      "wrapped": true
                    },
                    "items": {
                      "type": "string",
                      "xml": {
                        "name": "photoUrl"
                      }
                    }
                  },
                  "tags": {
                    "type": "array",
                    "xml": {
                      "wrapped": true
                    },
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": {
                          "type": "integer",
                          "format": "int64"
                        },
                        "name": {
                          "type": "string"
                        }
                      },
                      "xml": {
                        "name": "Tag"
                      }
                    }
                  },
                  "status": {
                    "type": "string",
                    "description": "pet status in the store",
                    "enum": [
                      "available",
                      "pending",
                      "sold"
                    ]
                  }
                },
                "xml": {
                  "name": "Pet"
                }
              }
            }
          },
          "400": {
            "description": "Invalid status value"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      }
    },
    "/pet/findByTags": {
      "get": {
        "tags": [
          "pet"
        ],
        "summary": "Finds Pets by tags",
        "description": "Multiple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing.",
        "operationId": "findPetsByTags",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "tags",
            "in": "query",
            "description": "Tags to filter by",
            "required": true,
            "type": "array",
            "items": {
              "type": "string"
            },
            "collectionFormat": "multi"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "name",
                  "photoUrls"
                ],
                "properties": {
                  "id": {
                    "type": "integer",
                    "format": "int64"
                  },
                  "category": {
                    "type": "object",
                    "properties": {
                      "id": {
                        "type": "integer",
                        "format": "int64"
                      },
                      "name": {
                        "type": "string"
                      }
                    },
                    "xml": {
                      "name": "Category"
                    }
                  },
                  "name": {
                    "type": "string",
                    "example": "doggie"
                  },
                  "photoUrls": {
                    "type": "array",
                    "xml": {
                      "wrapped": true
                    },
                    "items": {
                      "type": "string",
                      "xml": {
                        "name": "photoUrl"
                      }
                    }
                  },
                  "tags": {
                    "type": "array",
                    "xml": {
                      "wrapped": true
                    },
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": {
                          "type": "integer",
                          "format": "int64"
                        },
                        "name": {
                          "type": "string"
                        }
                      },
                      "xml": {
                        "name": "Tag"
                      }
                    }
                  },
                  "status": {
                    "type": "string",
                    "description": "pet status in the store",
                    "enum": [
                      "available",
                      "pending",
                      "sold"
                    ]
                  }
                },
                "xml": {
                  "name": "Pet"
                }
              }
            }
          },
          "400": {
            "description": "Invalid tag value"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ],
        "deprecated": true
      }
    },
    "/pet/{petId}": {
      "get": {
        "tags": [
          "pet"
        ],
        "summary": "Find pet by ID",
        "description": "Returns a single pet",
        "operationId": "getPetById",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "description": "ID of pet to return",
            "required": true,
            "type": "integer",
            "format": "int64"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "required": [
                "name",
                "photoUrls"
              ],
              "properties": {
                "id": {
                  "type": "integer",
                  "format": "int64"
                },
                "category": {
                  "type": "object",
                  "properties": {
                    "id": {
                      "type": "integer",
                      "format": "int64"
                    },
                    "name": {
                      "type": "string"
                    }
                  },
                  "xml": {
                    "name": "Category"
                  }
                },
                "name": {
                  "type": "string",
                  "example": "doggie"
                },
                "photoUrls": {
                  "type": "array",
                  "xml": {
                    "wrapped": true
                  },
                  "items": {
                    "type": "string",
                    "xml": {
                      "name": "photoUrl"
                    }
                  }
                },
                "tags": {
                  "type": "array",
                  "xml": {
                    "wrapped": true
                  },
                  "items": {
                    "type": "object",
                    "properties": {
                      "id": {
                        "type": "integer",
                        "format": "int64"
                      },
                      "name": {
                        "type": "string"
                      }
                    },
                    "xml": {
                      "name": "Tag"
                    }
                  }
                },
                "status": {
                  "type": "string",
                  "description": "pet status in the store",
                  "enum": [
                    "available",
                    "pending",
                    "sold"
                  ]
                }
              },
              "xml": {
                "name": "Pet"
              }
            }
          },
          "400": {
            "description": "Invalid ID supplied"
          },
          "404": {
            "description": "Pet not found"
          }
        },
        "security": [
          {
            "api_key": []
          }
        ]
      },
      "post": {
        "tags": [
          "pet"
        ],
        "summary": "Updates a pet in the store with form data",
        "description": "",
        "operationId": "updatePetWithForm",
        "consumes": [
          "application/x-www-form-urlencoded"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "description": "ID of pet that needs to be updated",
            "required": true,
            "type": "integer",
            "format": "int64"
          },
          {
            "name": "name",
            "in": "formData",
            "description": "Updated name of the pet",
            "required": false,
            "type": "string"
          },
          {
            "name": "status",
            "in": "formData",
            "description": "Updated status of the pet",
            "required": false,
            "type": "string"
          }
        ],
        "responses": {
          "405": {
            "description": "Invalid input"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      },
      "delete": {
        "tags": [
          "pet"
        ],
        "summary": "Deletes a pet",
        "description": "",
        "operationId": "deletePet",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "api_key",
            "in": "header",
            "required": false,
            "type": "string"
          },
          {
            "name": "petId",
            "in": "path",
            "description": "Pet id to delete",
            "required": true,
            "type": "integer",
            "format": "int64"
          }
        ],
        "responses": {
          "400": {
            "description": "Invalid ID supplied"
          },
          "404": {
            "description": "Pet not found"
          }
        },
        "security": [
          {
            "petstore_auth": [
              "write:pets",
              "read:pets"
            ]
          }
        ]
      }
    },
    "/store/inventory": {
      "get": {
        "tags": [
          "store"
        ],
        "summary": "Returns pet inventories by status",
        "description": "Returns a map of status codes to quantities",
        "operationId": "getInventory",
        "produces": [
          "application/json"
        ],
        "parameters": [],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "additionalProperties": {
                "type": "integer",
                "format": "int32"
              }
            }
          }
        },
        "security": [
          {
            "api_key": []
          }
        ]
      }
    },
    "/store/order": {
      "post": {
        "tags": [
          "store"
        ],
        "summary": "Place an order for a pet",
        "description": "",
        "operationId": "placeOrder",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "order placed for purchasing the pet",
            "required": true,
            "schema": {
              "$ref": "#/definitions/Order"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "properties": {
                "id": {
                  "type": "integer",
                  "format": "int64"
                },
                "petId": {
                  "type": "integer",
                  "format": "int64"
                },
                "quantity": {
                  "type": "integer",
                  "format": "int32"
                },
                "shipDate": {
                  "type": "string",
                  "format": "date-time"
                },
                "status": {
                  "type": "string",
                  "description": "Order Status",
                  "enum": [
                    "placed",
                    "approved",
                    "delivered"
                  ]
                },
                "complete": {
                  "type": "boolean"
                }
              },
              "xml": {
                "name": "Order"
              }
            }
          },
          "400": {
            "description": "Invalid Order"
          }
        }
      }
    },
    "/store/order/{orderId}": {
      "get": {
        "tags": [
          "store"
        ],
        "summary": "Find purchase order by ID",
        "description": "For valid response try integer IDs with value >= 1 and <= 10. Other values will generated exceptions",
        "operationId": "getOrderById",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "orderId",
            "in": "path",
            "description": "ID of pet that needs to be fetched",
            "required": true,
            "type": "integer",
            "maximum": 10,
            "minimum": 1,
            "format": "int64"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "properties": {
                "id": {
                  "type": "integer",
                  "format": "int64"
                },
                "petId": {
                  "type": "integer",
                  "format": "int64"
                },
                "quantity": {
                  "type": "integer",
                  "format": "int32"
                },
                "shipDate": {
                  "type": "string",
                  "format": "date-time"
                },
                "status": {
                  "type": "string",
                  "description": "Order Status",
                  "enum": [
                    "placed",
                    "approved",
                    "delivered"
                  ]
                },
                "complete": {
                  "type": "boolean"
                }
              },
              "xml": {
                "name": "Order"
              }
            }
          },
          "400": {
            "description": "Invalid ID supplied"
          },
          "404": {
            "description": "Order not found"
          }
        }
      },
      "delete": {
        "tags": [
          "store"
        ],
        "summary": "Delete purchase order by ID",
        "description": "For valid response try integer IDs with positive integer value. Negative or non-integer values will generate API errors",
        "operationId": "deleteOrder",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "orderId",
            "in": "path",
            "description": "ID of the order that needs to be deleted",
            "required": true,
            "type": "integer",
            "minimum": 1,
            "format": "int64"
          }
        ],
        "responses": {
          "400": {
            "description": "Invalid ID supplied"
          },
          "404": {
            "description": "Order not found"
          }
        }
      }
    },
    "/user/createWithList": {
      "post": {
        "tags": [
          "user"
        ],
        "summary": "Creates list of users with given input array",
        "description": "",
        "operationId": "createUsersWithListInput",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "List of user object",
            "required": true,
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/User"
              }
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
    "/user/{username}": {
      "get": {
        "tags": [
          "user"
        ],
        "summary": "Get user by user name",
        "description": "",
        "operationId": "getUserByName",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "username",
            "in": "path",
            "description": "The name that needs to be fetched. Use user1 for testing. ",
            "required": true,
            "type": "string"
          }
        ],
        "responses": {
          "200": {
            "description": "successful operation",
            "schema": {
              "type": "object",
              "properties": {
                "id": {
                  "type": "integer",
                  "format": "int64"
                },
                "username": {
                  "type": "string"
                },
                "firstName": {
                  "type": "string"
                },
                "lastName": {
                  "type": "string"
                },
                "email": {
                  "type": "string"
                },
                "password": {
                  "type": "string"
                },
                "phone": {
                  "type": "string"
                },
                "userStatus": {
                  "type": "integer",
                  "format": "int32",
                  "description": "User Status"
                }
              },
              "xml": {
                "name": "User"
              }
            }
          },
          "400": {
            "description": "Invalid username supplied"
          },
          "404": {
            "description": "User not found"
          }
        }
      },
      "put": {
        "tags": [
          "user"
        ],
        "summary": "Updated user",
        "description": "This can only be done by the logged in user.",
        "operationId": "updateUser",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "username",
            "in": "path",
            "description": "name that need to be updated",
            "required": true,
            "type": "string"
          },
          {
            "in": "body",
            "name": "body",
            "description": "Updated user object",
            "required": true,
            "schema": {
              "$ref": "#/definitions/User"
            }
          }
        ],
        "responses": {
          "400": {
            "description": "Invalid user supplied"
          },
          "404": {
            "description": "User not found"
          }
        }
      },
      "delete": {
        "tags": [
          "user"
        ],
        "summary": "Delete user",
        "description": "This can only be done by the logged in user.",
        "operationId": "deleteUser",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "name": "username",
            "in": "path",
            "description": "The name that needs to be deleted",
            "required": true,
            "type": "string"
          }
        ],
        "responses": {
          "400": {
            "description": "Invalid username supplied"
          },
          "404": {
            "description": "User not found"
          }
        }
      }
    },
    "/user/login": {
      "get": {
        "tags": [
          "user"
        ],
        "summary": "Logs user into the system",
        "description": "",
        "operationId": "loginUser",
        "produces": [
          "application/json",
          "application/xml"
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
            "description": "successful operation",
            "headers": {
              "X-Expires-After": {
                "type": "string",
                "format": "date-time",
                "description": "date in UTC when token expires"
              },
              "X-Rate-Limit": {
                "type": "integer",
                "format": "int32",
                "description": "calls per hour allowed by the user"
              }
            },
            "schema": {
              "type": "string"
            }
          },
          "400": {
            "description": "Invalid username/password supplied"
          }
        }
      }
    },
    "/user/logout": {
      "get": {
        "tags": [
          "user"
        ],
        "summary": "Logs out current logged in user session",
        "description": "",
        "operationId": "logoutUser",
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [],
        "responses": {
          "default": {
            "description": "successful operation"
          }
        }
      }
    },
    "/user/createWithArray": {
      "post": {
        "tags": [
          "user"
        ],
        "summary": "Creates list of users with given input array",
        "description": "",
        "operationId": "createUsersWithArrayInput",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json",
          "application/xml"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "List of user object",
            "required": true,
            "schema": {
              "type": "array",
              "items": {
                "$ref": "#/definitions/User"
              }
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
    "/user": {
      "post": {
        "tags": [
          "user"
        ],
        "summary": "Create user",
        "description": "This can only be done by the logged in user.",
        "operationId": "createUser",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json",
          "application/xml"
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
    }
  },
  "securityDefinitions": {
    "api_key": {
      "type": "apiKey",
      "name": "api_key",
      "in": "header"
    },
    "petstore_auth": {
      "type": "oauth2",
      "authorizationUrl": "https://petstore.swagger.io/oauth/authorize",
      "flow": "implicit",
      "scopes": {
        "read:pets": "read your pets",
        "write:pets": "modify pets in your account"
      }
    }
  },
  "definitions": {
    "ApiResponse": {
      "type": "object",
      "properties": {
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "type": {
          "type": "string"
        },
        "message": {
          "type": "string"
        }
      }
    },
    "Category": {
      "type": "object",
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "name": {
          "type": "string"
        }
      },
      "xml": {
        "name": "Category"
      }
    },
    "Pet": {
      "type": "object",
      "required": [
        "name",
        "photoUrls"
      ],
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "category": {
          "type": "object",
          "properties": {
            "id": {
              "type": "integer",
              "format": "int64"
            },
            "name": {
              "type": "string"
            }
          },
          "xml": {
            "name": "Category"
          }
        },
        "name": {
          "type": "string",
          "example": "doggie"
        },
        "photoUrls": {
          "type": "array",
          "xml": {
            "wrapped": true
          },
          "items": {
            "type": "string",
            "xml": {
              "name": "photoUrl"
            }
          }
        },
        "tags": {
          "type": "array",
          "xml": {
            "wrapped": true
          },
          "items": {
            "type": "object",
            "properties": {
              "id": {
                "type": "integer",
                "format": "int64"
              },
              "name": {
                "type": "string"
              }
            },
            "xml": {
              "name": "Tag"
            }
          }
        },
        "status": {
          "type": "string",
          "description": "pet status in the store",
          "enum": [
            "available",
            "pending",
            "sold"
          ]
        }
      },
      "xml": {
        "name": "Pet"
      }
    },
    "Tag": {
      "type": "object",
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "name": {
          "type": "string"
        }
      },
      "xml": {
        "name": "Tag"
      }
    },
    "Order": {
      "type": "object",
      "properties": {
        "id": {
          "type": "integer",
          "format": "int64"
        },
        "petId": {
          "type": "integer",
          "format": "int64"
        },
        "quantity": {
          "type": "integer",
          "format": "int32"
        },
        "shipDate": {
          "type": "string",
          "format": "date-time"
        },
        "status": {
          "type": "string",
          "description": "Order Status",
          "enum": [
            "placed",
            "approved",
            "delivered"
          ]
        },
        "complete": {
          "type": "boolean"
        }
      },
      "xml": {
        "name": "Order"
      }
    },
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
        "firstName": {
          "type": "string"
        },
        "lastName": {
          "type": "string"
        },
        "email": {
          "type": "string"
        },
        "password": {
          "type": "string"
        },
        "phone": {
          "type": "string"
        },
        "userStatus": {
          "type": "integer",
          "format": "int32",
          "description": "User Status"
        }
      },
      "xml": {
        "name": "User"
      }
    }
  },
  "externalDocs": {
    "description": "Find out more about Swagger",
    "url": "http://swagger.io"
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
                               TEXT,
                               role_id
                               INTEGER
                           );
                           """)
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS orders
                           (
                               order_id
                               INTEGER
                               PRIMARY
                               KEY,
                               user_id
                               INTEGER,
                               status
                               TEXT
                           );
                           """)
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS inventory_items
                           (
                               item_id
                               INTEGER
                               PRIMARY
                               KEY,
                               product_id
                               INTEGER,
                               quantity
                               INTEGER
                           );
                           """)
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS roles
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY,
                               role_name
                               TEXT
                           );
                           """)
            cursor.execute(
                "INSERT INTO pets (id, name, status, tags) VALUES (1, 'Buddy', 'available', 'dog,friendly');")
            cursor.execute("INSERT INTO pets (id, name, status, tags) VALUES (2, 'Whiskers', 'pending', 'cat');")
            cursor.execute(
                "INSERT INTO users (id, username, password, email, role_id) VALUES (101, 'testuser', 'testpass', 'test@example.com', 1);")
            cursor.execute(
                "INSERT INTO users (id, username, password, email, role_id) VALUES (102, 'user2', 'pass2', 'user2@example.com', 2);")
            cursor.execute("INSERT INTO orders (order_id, user_id, status) VALUES (1001, 101, 'pending');")
            cursor.execute("INSERT INTO orders (order_id, user_id, status) VALUES (1002, 102, 'completed');")
            cursor.execute("INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (1, 1, 50);")
            cursor.execute("INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (2, 2, 120);")
            cursor.execute("INSERT INTO roles (id, role_name) VALUES (1, 'Admin');")
            cursor.execute("INSERT INTO roles (id, role_name) VALUES (2, 'User');")
            conn.commit()
            conn.close()
            logger.info(f"Dummy SQLite database created at {dummy_db_path} with sample data.")
        except Exception as ex:
            logger.error(f"Error creating dummy DB during __main__ init: {ex}")

    main()
