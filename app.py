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
import re
from urllib.parse import urlparse
import uuid
import time
from datetime import datetime, date # Import datetime and date

# FIX: Increase the string conversion limit for large integers.
sys.set_int_max_str_digits(0) # 0 means unlimited

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.jmeter_generator import JMeterScriptGenerator
from utils.database_connector import DatabaseConnector, DatabaseConfig
from utils.swagger_parser import SwaggerParser, SwaggerEndpoint # Import SwaggerEndpoint
from utils.data_mapper import DataMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Custom JSON Encoder to handle non-serializable types like Pandas Timestamps and datetimes
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, bytes): # <-- ADDED THIS BLOCK to handle bytes objects
            try:
                return obj.decode('utf-8') # Decode bytes to string
            except UnicodeDecodeError:
                return obj.hex() # Fallback to hex if not utf-8 decodable
        return json.JSONEncoder.default(self, obj)


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


def _repair_truncated_json(json_string: str) -> str:
    """
    Attempts to repair a truncated JSON string by appending missing closing characters.
    This is a heuristic and might not fix all cases, but addresses common truncation issues
    like unterminated strings or incomplete objects/arrays.
    """
    stripped_string = json_string.strip()
    
    # 1. Remove markdown fences, if any (should ideally be handled before this, but as a safeguard)
    if stripped_string.startswith("```json"):
        stripped_string = stripped_string[len("```json"):].strip()
    if stripped_string.endswith("```"):
        stripped_string = stripped_string[:-len("```")].strip()

    # Try to parse the string as is. If it succeeds, return.
    try:
        json.loads(stripped_string)
        return stripped_string
    except json.JSONDecodeError:
        pass # Continue with repair

    # Track open characters and string state more carefully
    open_stack = [] # Use a stack for {'[':']', '{':'}'}
    in_string = False
    repaired_chars = []
    
    for char in stripped_string:
        if char == '"':
            # Toggle in_string state if it's not an escaped quote
            if not in_string or (in_string and repaired_chars and repaired_chars[-1] != '\\'):
                in_string = not in_string
        elif not in_string: # Only consider structural characters if not inside a string
            if char == '[':
                open_stack.append('[')
            elif char == '{':
                open_stack.append('{')
            elif char == ']':
                if open_stack and open_stack[-1] == '[':
                    open_stack.pop()
                # else: Mismatched or extra closing bracket, ignore for now, rely on final balancing
            elif char == '}':
                if open_stack and open_stack[-1] == '{':
                    open_stack.pop()
                # else: Mismatched or extra brace, ignore for now
        repaired_chars.append(char)

    repaired_output = "".join(repaired_chars)

    # If the string was unterminated, close it
    if in_string:
        repaired_output += '"'

    # Remove trailing commas that would cause parsing errors
    while repaired_output and repaired_output.endswith(','):
        # Only remove if the comma is immediately before a structural closer or end of string
        # e.g., "key": "value", -- valid in an object before another key or '}'
        # { "key": "value", -- invalid if followed by } or ]
        # This logic is complex. A simpler approach is to remove if it immediately precedes end or a bracket/brace.
        
        # Check if the comma is indeed problematic (e.g., not followed by a valid JSON element start)
        # This is a heuristic and might need further refinement for edge cases
        if repaired_output.endswith((']', '}')): # Trailing comma right before a closer, e.g. [1, ] or {"a":1,}
             # Find last non-whitespace, non-comma char
            last_valid_char_idx = len(repaired_output) - 1
            while last_valid_char_idx >= 0 and repaired_output[last_valid_char_idx] in [',', ' ', '\n', '\r', '\t']:
                last_valid_char_idx -= 1
            if last_valid_char_idx >= 0 and repaired_output[last_valid_char_idx] in ['[', '{']:
                repaired_output = repaired_output[:last_valid_char_idx + 1] # Trim from the comma
            else:
                repaired_output = repaired_output[:-1] # Remove just the last comma if no structural issue immediately before
        else:
            break # No more problematic commas at the end
    
    # Append missing closing characters based on the stack
    # Reverse the stack to close in correct order (innermost first)
    for opener in reversed(open_stack):
        if opener == '[':
            repaired_output += ']'
        elif opener == '{':
            repaired_output += '}'

    # Final aggressive trim for very malformed endings (e.g., `,}]` or `]]}}`)
    # This loop tries to make the JSON parsable even if it means losing some trailing data
    # It attempts to iteratively trim until it's parsable or no further simple trimming helps.
    for _ in range(5): # Limit attempts to prevent infinite loop
        try:
            json.loads(repaired_output)
            break # Successfully parsed, stop
        except json.JSONDecodeError as e:
            if not repaired_output: # Nothing left to trim
                break
            # Attempt to strip characters that are common causes of syntax errors at the end
            # This is a bit of a brute-force approach but effective for truncation
            if repaired_output.endswith(('"', "'", ',', '[', '{', ':', '\\')):
                repaired_output = repaired_output[:-1]
            elif repaired_output.endswith(']}') and not open_stack: # if it looks like ]} but stack is empty, it's malformed
                 repaired_output = repaired_output[:-2] # trim both
            elif repaired_output.endswith(']}') and len(open_stack) == 1 and open_stack[0] == '[': # if it's ]} and only [ is open, then it's probably malformed and we need to fix
                 repaired_output = repaired_output[:-1] # trim }
            elif repaired_output.endswith(']'):
                if open_stack and open_stack[-1] == '[':
                    repaired_output = repaired_output[:-1] # let the stack handle it
                elif not open_stack:
                    repaired_output = repaired_output[:-1] # Remove extra closing bracket
            elif repaired_output.endswith('}'):
                if open_stack and open_stack[-1] == '{':
                    repaired_output = repaired_output[:-1] # let the stack handle it
                elif not open_stack:
                    repaired_output = repaired_output[:-1] # Remove extra closing brace
            else:
                break # Can't fix further by simple trimming
            
    return repaired_output


def call_llm_for_scenario_plan(prompt: str, swagger_endpoints: List[SwaggerEndpoint],
                               db_tables_schema: Dict[str, List[Dict[str, Any]]], # Changed to Any for schema details
                               db_sampled_data: Dict[str, pd.DataFrame], # Added sampled data for LLM context
                               thread_group_users: int,
                               ramp_up_time: int,
                               loop_count: int,
                               api_key: str) -> Optional[List[Dict[str, Any]]]: # Now returns a list of scenario requests
    """
    Calls an LLM to generate a detailed, structured test plan (scenario plan)
    in JSON format, including parameter sourcing, assertions, and extractions.
    """

    swagger_summary = []
    for ep in swagger_endpoints:
        # Include a more concise summary of parameters and body schema for LLM
        params_info = []
        for p in ep.parameters:
            param_detail = {"name": p.get('name'), "in": p.get('in'), "type": p.get('type'), "required": p.get('required', False)}
            # Only include min/max/enum if they are present and relevant for dummy generation
            if 'enum' in p: param_detail['enum'] = p['enum']
            if 'format' in p: param_detail['format'] = p['format'] # Important for dates/etc.
            if p.get('type') in ['integer', 'number']:
                if 'minimum' in p: param_detail['minimum'] = p['minimum']
                if 'maximum' in p: param_detail['maximum'] = p['maximum']
            if p.get('type') == 'string':
                if 'minLength' in p: param_detail['minLength'] = p['minLength']
                if 'maxLength' in p: param_detail['maxLength'] = p['maxLength']
            params_info.append(param_detail)
        
        # For body schema, only provide top-level properties and types, not deep nesting
        body_schema_concise = {}
        if ep.body_schema and isinstance(ep.body_schema, dict):
            if ep.body_schema.get('type') == 'object' and 'properties' in ep.body_schema:
                for prop_name, prop_details in ep.body_schema['properties'].items():
                    body_schema_concise[prop_name] = {"type": prop_details.get('type')}
                    if 'format' in prop_details: body_schema_concise[prop_name]['format'] = prop_details['format']
                    if 'enum' in prop_details: body_schema_concise[prop_name]['enum'] = prop_details['enum']
            else: # Handle non-object root body schemas
                body_schema_concise = {"type": ep.body_schema.get('type')}
                if 'format' in ep.body_schema: body_schema_concise['format'] = ep.body_schema['format']
                if 'enum' in ep.body_schema: body_schema_concise['enum'] = ep.body_schema['enum']
        
        swagger_summary.append({
            "method": ep.method,
            "path": ep.path,
            "operationId": ep.operation_id,
            "summary": ep.summary, # Keep summary for LLM context
            "parameters": params_info,
            "body_schema_concise": body_schema_concise, # Use concise body schema
            # Removed full responses to shorten prompt, LLM can infer common extractions
        })

    db_schema_summary = {}
    for table_name, columns in db_tables_schema.items():
        column_details = []
        for col in columns:
            # Only include name and type for brevity
            col_info = {"name": col.get('name'), "type": col.get('type')}
            if col.get('pk'): col_info['pk'] = True
            if col.get('fk'): col_info['fk'] = col['fk']
            column_details.append(col_info) 
        db_schema_summary[table_name] = column_details
    
    # Provide only column names with a hint for sampled data availability
    sampled_data_summary_concise = {}
    for table_name, df in db_sampled_data.items():
        if not df.empty:
            sampled_data_summary_concise[table_name] = {"columns": df.columns.tolist(), "has_data": True}
        else:
            sampled_data_summary_concise[table_name] = {"columns": [], "has_data": False}

    llm_prompt = f"""
    You are an expert in performance testing and JMeter script generation.
    Your task is to design a detailed test scenario plan in JSON format.
    This plan should directly specify each API request, how its parameters and body fields are sourced (from CSV, from previous extractions, generated dummy data, or static values),
    what assertions should be applied, and what values should be extracted from responses.
    Be concise in descriptions and strictly adhere to the JSON schema.

    User's test scenario request: {prompt}

    Available Swagger Endpoints (with concise parameter and body schemas):
    {json.dumps(swagger_summary, indent=2)}

    Detailed Database Schema (includes column names, types, and key info):
    {json.dumps(db_schema_summary, indent=2)}

    Sampled Database Data (column names and data availability hint):
    {json.dumps(sampled_data_summary_concise, indent=2)}

    Current Application Settings:
    - Number of Users: {thread_group_users}
    - Ramp-up Time (seconds): {ramp_up_time}
    - Loop Count: {loop_count} (Use -1 for infinite)
    - Authentication Enabled: {st.session_state.enable_auth_flow}
    - Login Endpoint Path: {st.session_state.auth_login_endpoint_path}
    - Auth Token JSON Path: {st.session_state.auth_token_json_path}
    - Auth Header Name: {st.session_state.auth_header_name}
    - Auth Header Prefix: {st.session_state.auth_header_prefix}

    Design the test scenario as a JSON array of request objects. Each request object must follow this structure EXACTLY:
    ```json
    [
      {{
        "name": "Login_User", 
        "method": "POST",
        "path": "/user/login", 
        "description": "User login.",
        "parameters_and_body_fields": [
          {{ "name": "username", "in": "body", "source": "from_csv", "table_name": "users", "column_name": "username", "type": "string" }},
          {{ "name": "password", "in": "body", "source": "from_csv", "table_name": "users", "column_name": "password", "type": "string" }}
        ],
        "assertions": [
          {{ "type": "Response Code", "value": "200" }}
        ],
        "extractions": [
          {{ "json_path": "{st.session_state.auth_token_json_path}", "var_name": "authToken" }}
        ],
        "think_time_ms": 500
      }},
      {{
        "name": "Get_Pet_By_ID",
        "method": "GET",
        "path": "/pet/{{petId}}", 
        "description": "Fetch pet details.",
        "parameters_and_body_fields": [
          {{ "name": "petId", "in": "path", "source": "from_csv", "table_name": "pets", "column_name": "id", "type": "integer", "format": "int64" }},
          {{ "name": "{st.session_state.auth_header_name}", "in": "header", "source": "from_extraction", "source_request_name": "Login_User", "extracted_variable_name": "authToken", "prefix": "{st.session_state.auth_header_prefix}" }}
        ],
        "assertions": [
          {{ "type": "Response Code", "value": "200" }}
        ],
        "extractions": [],
        "think_time_ms": 100
      }}
    ]
    ```
    
    For each `parameters_and_body_fields` item, **adhere to these strict rules**:
    - `name`: Parameter/field name. For body fields in a JSON object, use dot notation for nesting (e.g., 'user.address.street').
    - `in`: `path`, `query`, `header`, `body`.
    - `source`: `from_csv`, `from_extraction`, `generate_dummy`, `static_value`.
    - **IF `source` IS `"from_csv"`**: You **MUST INCLUDE BOTH** `"table_name"` AND `"column_name"`. These MUST correspond to tables and columns in `Detailed Database Schema` and `Sampled Database Data`. The `"value"` field **MUST NOT** be present. Fields like `"source_request_name"`, `"extracted_variable_name"`, `"prefix"` **MUST NOT** be present.
    - **IF `source` IS `"static_value"`**: You **MUST INCLUDE ONLY** the `"value"` field (e.g., `"value": "some_static_text"`). Fields like `"table_name"`, `"column_name"`, `"source_request_name"`, `"extracted_variable_name"`, `"prefix"` **MUST NOT** be present.
    - **IF `source` IS `"from_extraction"`**: You **MUST INCLUDE BOTH** `"source_request_name"` AND `"extracted_variable_name"`. An optional `"prefix"` field can also be included. Fields like `"table_name"`, `"column_name"`, `"value"` **MUST NOT** be present.
    - **IF `source` IS `"generate_dummy"`**: You **MUST NOT** include `"table_name"`, `"column_name"`, `"source_request_name"`, `"extracted_variable_name"`, `"value"`, or `"prefix"`. Include original Swagger `type`, `format`, `enum`, `minimum`, `maximum`, `minLength`, `maxLength` to guide dummy data generation if available.

    Your response MUST be ONLY the JSON array described above, with no additional text, markdown, or conversational elements. **Ensure the JSON is always perfectly formed and complete, with all brackets and commas correctly placed and no trailing commas or incomplete structures. DO NOT include comments in the JSON output.**
    """

    protocol_segment = "https://"
    domain_segment = "generativelanguage.googleapis.com"
    path_segment = "/v1beta/models/gemini-2.0-flash:generateContent?key="

    temp_base_api_url = "".join([protocol_segment, domain_segment, path_segment])
    base_api_url_cleaned = _clean_url_string(temp_base_api_url)
    api_url = base_api_url_cleaned + api_key

    payload = {
        "contents": [{"role": "user", "parts": [{"text": llm_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "method": {"type": "STRING", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                        "path": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "parameters_and_body_fields": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "in": {"type": "STRING", "enum": ["path", "query", "header", "body"]},
                                    "source": {"type": "STRING", "enum": ["from_csv", "from_extraction", "generate_dummy", "static_value"]},
                                    # All these are optional here because they are conditionally required/forbidden by the 'source' value,
                                    # which we enforce via the prompt instructions and post-processing.
                                    "table_name": {"type": "STRING"},
                                    "column_name": {"type": "STRING"},
                                    "source_request_name": {"type": "STRING"},
                                    "extracted_variable_name": {"type": "STRING"},
                                    "value": {"type": "STRING"}, # This one is the problem child for from_csv
                                    "prefix": {"type": "STRING"},
                                    "type": {"type": "STRING"}, # Original Swagger type
                                    "format": {"type": "STRING"}, # Original Swagger format
                                    "enum": {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "minimum": {"type": "NUMBER"},
                                    "maximum": {"type": "NUMBER"},
                                    "minLength": {"type": "INTEGER"},
                                    "maxLength": {"type": "INTEGER"}
                                },
                                "required": ["name", "in", "source"]
                            }
                        },
                        "assertions": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "type": {"type": "STRING", "enum": ["Response Code", "Response Body Contains"]},
                                    "value": {"type": "STRING"}
                                },
                                "required": ["type", "value"]
                            }
                        },
                        "extractions": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "json_path": {"type": "STRING"},
                                    "var_name": {"type": "STRING"}
                                },
                                "required": ["json_path", "var_name"]
                            }
                        },
                        "think_time_ms": {"type": "INTEGER"}
                    },
                    "required": ["name", "method", "path"]
                }
            }
        }
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Check if result structure is as expected and parse the JSON string
        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            
            json_response_str = result['candidates'][0]['content']['parts'][0]['text']
            logger.info(f"Raw LLM JSON response before cleanup: {json_response_str}") # Log before cleanup

            # Attempt to clean up the response string: remove markdown code block fences
            if json_response_str.strip().startswith("```json"):
                json_response_str = json_response_str.strip()[len("```json"):].strip()
                if json_response_str.strip().endswith("```"):
                    json_response_str = json_response_str.strip()[:-len("```")].strip()
            
            try:
                parsed_json = json.loads(json_response_str)
                logger.info(f"Successfully parsed LLM JSON response.")
                return parsed_json
            except json.JSONDecodeError as e:
                # Attempt to repair the JSON string if parsing fails
                st.warning(f"Attempting to repair truncated JSON response. Original error: {e}")
                repaired_json_str = _repair_truncated_json(json_response_str)
                logger.info(f"Attempted to repair JSON string to: {repaired_json_str}")
                try:
                    parsed_json = json.loads(repaired_json_str)
                    st.success("Successfully repaired and parsed LLM JSON response.")
                    return parsed_json
                except json.JSONDecodeError as repair_e:
                    st.error(f"Failed to parse LLM's JSON response even after repair. This indicates a severe issue with the LLM's output format. Error: {repair_e}")
                    st.code(f"Problematic JSON string (after attempted cleanup and repair):\n{repaired_json_str}") # Show the problematic string to the user
                    logger.error(f"JSON parsing error after repair from LLM response: {repair_e}. Raw response after repair: {repaired_json_str}")
                    return None
        else:
            logger.error(f"LLM did not return a valid structured suggestion: {result}")
            return None
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
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM call for suggestion: {e}")
        logger.error(f"An unexpected error occurred: {e}")
        return None


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
    if 'db_sampled_data' not in st.session_state:
        st.session_state.db_sampled_data = {}
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
    if 'llm_structured_scenario' not in st.session_state: # Renamed from llm_suggestion
        st.session_state.llm_structured_scenario = None
    if 'enable_constant_timer' not in st.session_state:
        st.session_state.enable_constant_timer = False
    if 'constant_timer_delay_ms' not in st.session_state:
        st.session_state.constant_timer_delay_ms = 300
    if 'include_scenario_assertions' not in st.session_state:
        st.session_state.include_scenario_assertions = True
    if 'test_plan_name' not in st.session_state:
        st.session_state.test_plan_name = "Web Application Performance Test"
    if 'thread_group_name' not in st.session_state:
        st.session_state.thread_group_name = "Users"
    if 'select_all_endpoints' not in st.session_state:
        st.session_state.select_all_endpoints = False
    if 'jmx_content_download' not in st.session_state:
        st.session_state.jmx_content_download = None
    if 'csv_content_download' not in st.session_state:
        st.session_state.csv_content_download = None
    if 'mapping_metadata_download' not in st.session_state:
        st.session_state.mapping_metadata_download = None
    if 'full_swagger_spec_download' not in st.session_state:
        st.session_state.full_swagger_spec_download = None
    if 'enable_setup_teardown_thread_groups' not in st.session_state:
        st.session_state.enable_setup_teardown_thread_groups = False
    
    if 'enable_auth_flow' not in st.session_state:
        st.session_state.enable_auth_flow = False
    if 'auth_login_endpoint_path' not in st.session_state:
        st.session_state.auth_login_endpoint_path = "/user/login"
    if 'auth_login_method' not in st.session_state:
        st.session_state.auth_login_method = "POST"
    if 'auth_login_username_param' not in st.session_state:
        st.session_state.auth_login_username_param = "username"
    if 'auth_login_password_param' not in st.session_state:
        st.session_state.auth_login_password_param = "password"
    if 'auth_login_body_template' not in st.session_state:
        st.session_state.auth_login_body_template = '{"username": "${csv_users_username}", "password": "${csv_users_password}"}'
    if 'auth_token_json_path' not in st.session_state:
        st.session_state.auth_token_json_path = "$.access_token"
    if 'auth_header_name' not in st.session_state:
        st.session_state.auth_header_name = "Authorization"
    if 'auth_header_prefix' not in st.session_state:
        st.session_state.auth_header_prefix = "Bearer "

    if 'db_host' not in st.session_state:
        st.session_state.db_host = ""
    if 'db_user' not in st.session_state:
        st.session_state.db_user = ""
    if 'db_password' not in st.session_state:
        st.session_state.db_password = ""
    if 'db_name' not in st.session_state:
        st.session_state.db_name = ""
    if 'db_port' not in st.session_state:
        st.session_state.db_port = ""
    if 'db_type_selected' not in st.session_state:
        st.session_state.db_type_selected = "SQLite"

    if 'loop_count_input_specific' not in st.session_state:
        st.session_state.loop_count_input_specific = 1

    # Helper function to recursively build JSON body based on schema and LLM-suggested mappings
    def _build_recursive_json_body_with_llm_guidance(
        schema: Dict[str, Any], 
        llm_param_fields: List[Dict[str, Any]], # LLM's suggested parameter/body field sourcing
        current_path_segments: List[str], 
        extracted_vars: Dict[str, str], # Already JMeter variable format
        endpoint_key: str # For DataMapper calls if needed
    ) -> Any:
        """
        Recursively builds a JSON body (or part of it) based on schema definitions,
        prioritizing LLM-suggested sourcing strategies and integrating extracted variables
        and DataMapper for CSV/dummy generation.
        """
        logger.debug(f"Entering _build_recursive_json_body_with_llm_guidance for path: {'.'.join(current_path_segments)}, schema_type: {schema.get('type')}")

        if not isinstance(schema, dict):
            logger.debug(f"Schema is not a dictionary: {schema}. Returning as is.")
            return schema 

        schema_type = schema.get('type', 'object') 

        if schema_type == 'object':
            body_obj = {}
            properties = schema.get('properties', {})
            
            for prop_name, prop_details in properties.items():
                full_param_name_dot_path = ".".join(current_path_segments + [prop_name])
                
                # Find LLM's explicit instruction for this specific field
                # For body fields, the 'name' in LLM instruction should match the full_param_name_dot_path
                llm_instruction = next((item for item in llm_param_fields if item.get('name') == full_param_name_dot_path and item.get('in') == 'body'), None)

                value_set = False

                if llm_instruction:
                    source = llm_instruction.get("source")
                    if source == "from_csv":
                        table_name = llm_instruction.get("table_name")
                        column_name = llm_instruction.get("column_name")
                        if table_name and column_name:
                            jmeter_var = f"${{csv_{table_name}_{column_name}}}"
                            body_obj[prop_name] = jmeter_var
                            value_set = True
                            logger.debug(f"Body field '{full_param_name_dot_path}' populated from LLM CSV instruction: {jmeter_var}")
                        else:
                            # Fallback if table_name or column_name is missing for from_csv
                            logger.warning(f"LLM suggested 'from_csv' for {full_param_name_dot_path} but missing table_name/column_name. Falling back to dummy.")
                            body_obj[prop_name] = DataMapper._generate_dummy_value(llm_instruction)
                            value_set = True
                    elif source == "from_extraction":
                        extracted_var_name = llm_instruction.get("extracted_variable_name")
                        prefix = llm_instruction.get("prefix", "")
                        if extracted_var_name in extracted_vars: # Check if the extracted variable is available
                             resolved_value = f"{prefix}{extracted_vars[extracted_var_name]}"
                             body_obj[prop_name] = resolved_value
                             value_set = True
                             logger.debug(f"Body field '{full_param_name_dot_path}' populated from LLM extraction instruction: {resolved_value}")
                        else:
                             # Fallback to a placeholder or dummy if extraction source not found
                             logger.warning(f"LLM suggested 'from_extraction' for {prop_name} but '{extracted_var_name}' not found in extracted_variables_map. Falling back to dummy.")
                             body_obj[prop_name] = DataMapper._generate_dummy_value(llm_instruction) # Use llm_instruction as it contains schema details
                             value_set = True
                    elif source == "static_value":
                        body_obj[prop_name] = llm_instruction.get("value")
                        value_set = True
                        logger.debug(f"Body field '{full_param_name_dot_path}' populated from LLM static instruction: {llm_instruction.get('value')}")
                    elif source == "generate_dummy":
                        body_obj[prop_name] = DataMapper._generate_dummy_value(llm_instruction) # Pass LLM instruction which has swagger details
                        value_set = True
                        logger.debug(f"Body field '{full_param_name_dot_path}' populated from LLM dummy instruction: {body_obj[prop_name]}")
                
                # If LLM didn't provide specific instruction or it failed, fall back
                if not value_set:
                    # Recursively build for nested objects/arrays
                    if prop_details.get('type') == 'object':
                        body_obj[prop_name] = _build_recursive_json_body_with_llm_guidance(
                            prop_details, llm_param_fields, current_path_segments + [prop_name], extracted_vars, endpoint_key
                        )
                        logger.debug(f"Body field '{full_param_name_dot_path}' populated by recursive call (object).")
                    elif prop_details.get('type') == 'array':
                        if 'items' in prop_details:
                            array_item_path_segments = current_path_segments + [prop_name, "_item"]
                            
                            item_schema = prop_details['items'] # Get the schema for array items
                            
                            generated_item = _build_recursive_json_body_with_llm_guidance(
                                item_schema, llm_param_fields, array_item_path_segments, extracted_vars, endpoint_key
                            )
                            body_obj[prop_name] = [generated_item] # Generate one item for the array as a sample
                            logger.debug(f"Body field '{full_param_name_dot_path}' (array) populated with one recursive item: {body_obj[prop_name]}")
                        else:
                            body_obj[prop_name] = [] # Empty array if no items schema
                            logger.debug(f"Body field '{full_param_name_dot_path}' (array) populated as empty (no item schema).")
                    else: # Primitive type (string, integer, boolean, number etc.)
                        # Fallback to DataMapper's default dummy generation if no LLM instruction or other source
                        body_obj[prop_name] = DataMapper._generate_dummy_value(prop_details) # Pass prop_details for smarter dummy generation
                        logger.debug(f"Body field '{full_param_name_dot_path}' (primitive, fallback) populated with dummy: {body_obj[prop_name]}")

            return body_obj
        
        elif schema_type == 'array':
            logger.debug(f"Schema is an array. Path: {'.'.join(current_path_segments)}")
            # Find LLM's explicit instruction for the root array itself (if body is an array)
            root_array_full_path = ".".join(current_path_segments)
            llm_instruction = next((item for item in llm_param_fields if item.get('name') == root_array_full_path and item.get('in') == 'body'), None)

            if llm_instruction and llm_instruction.get("source") == "static_value":
                try:
                    return json.loads(llm_instruction.get("value")) # Assume static value for array is a JSON string of array
                except json.JSONDecodeError:
                    logger.warning(f"Static value for array body at {root_array_full_path} is not valid JSON array: {llm_instruction.get('value')}")
                    return []
            
            if 'items' in schema:
                # Generate a single item for the array as a sample based on its item schema
                generated_item = _build_recursive_json_body_with_llm_guidance(
                    schema['items'], llm_param_fields, current_path_segments + ["_item"], extracted_vars, endpoint_key
                )
                logger.debug(f"Root array body populated with one recursive item: {generated_item}")
                return [generated_item]
            else:
                logger.debug("Root array body populated as empty (no item schema).")
                return [] 

        # Handle primitive types directly at the current level (e.g., if body is just a string or integer)
        elif schema_type in ['string', 'integer', 'boolean', 'number']:
            # Corrected f-string: removed extra single quote
            logger.debug(f"Schema is a primitive type: {schema_type}. Path: {'.'.join(current_path_segments)}")
            # For root-level primitive body, LLM instruction's name should match current_path_segments (empty string for root)
            root_primitive_name = ".".join(current_path_segments) if current_path_segments else "" # For root body, name ""
            llm_instruction = next((item for item in llm_param_fields if item.get('name') == root_primitive_name and item.get('in') == 'body'), None)

            if llm_instruction:
                source = llm_instruction.get("source")
                if source == "static_value":
                    val = llm_instruction.get('value')
                    if schema_type == 'integer':
                        try: return int(val)
                        except (ValueError, TypeError): return val
                    elif schema_type == 'boolean':
                        return str(val).lower() == 'true'
                    elif schema_type == 'number':
                        try: return float(val)
                        except (ValueError, TypeError): return val
                    else: # string
                        return val
                elif source == "generate_dummy":
                    return DataMapper._generate_dummy_value(llm_instruction) # Pass LLM instruction for full details
                elif source == "from_csv":
                    table_name = llm_instruction.get("table_name")
                    column_name = llm_instruction.get("column_name")
                    if table_name and column_name:
                        return f"${{csv_{table_name}_{column_name}}}"
                    else:
                        logger.warning(f"LLM suggested 'from_csv' for root primitive body but missing table_name/column_name. Falling back to dummy.")
                        return DataMapper._generate_dummy_value(schema) # Fallback to dummy
                elif source == "from_extraction":
                    extracted_var_name = llm_instruction.get("extracted_variable_name")
                    prefix = llm_instruction.get("prefix", "")
                    if extracted_var_name in extracted_vars:
                        return f"{prefix}{extracted_vars[extracted_var_name]}"
                    else:
                        logger.warning(f"LLM suggested 'from_extraction' for root primitive body but '{extracted_var_name}' not found in extracted_variables_map. Falling back to dummy.")
                        return DataMapper._generate_dummy_value(schema) # Fallback to dummy
            else:
                # Fallback to DataMapper's default dummy generation if no LLM instruction
                return DataMapper._generate_dummy_value(schema) # Pass the original schema for dummy generation
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

        uploaded_swagger_file = st.file_uploader(
            "Upload Parsed Swagger Endpoints (JSON)",
            type="json",
            key="upload_swagger_endpoints_file",
            help="Upload a previously downloaded JSON file containing parsed Swagger endpoints."
        )

        if uploaded_swagger_file is not None:
            try:
                uploaded_data = json.load(uploaded_swagger_file)
                # Convert dicts back to SwaggerEndpoint objects
                st.session_state.swagger_endpoints = [SwaggerEndpoint(**ep_dict) for ep_dict in uploaded_data]
                st.session_state.current_swagger_url = "Uploaded from file" # Indicate source
                
                # Mock a SwaggerParser for full_swagger_spec access if needed by other components
                if 'swagger_parser' not in st.session_state or not isinstance(st.session_state.swagger_parser, SwaggerParser):
                    st.session_state.swagger_parser = SwaggerParser("https://mock.swagger.url") # Use a dummy URL
                    st.session_state.swagger_parser.swagger_data = {"basePath": "/v2", "paths": {}} # Minimal mock data
                    # Populate paths and definitions based on uploaded data if possible
                    for ep in st.session_state.swagger_endpoints:
                        if ep.path not in st.session_state.swagger_parser.swagger_data['paths']:
                            st.session_state.swagger_parser.swagger_data['paths'][ep.path] = {}
                        st.session_state.swagger_parser.swagger_data['paths'][ep.path][ep.method.lower()] = {
                            "operationId": ep.operation_id,
                            "summary": ep.summary,
                            "parameters": ep.parameters,
                            "responses": ep.responses,
                            "requestBody": {"content": {"application/json": {"schema": ep.body_schema}}} if ep.body_schema else None
                        }

                st.success(f"Loaded {len(st.session_state.swagger_endpoints)} API endpoints from uploaded file.")
            except Exception as e:
                st.error(f"Error loading Swagger endpoints from file: {e}")
                st.session_state.swagger_endpoints = []
        
        if (st.button("Fetch Swagger", key="fetch_swagger") or 
           (swagger_url and swagger_url != st.session_state.current_swagger_url and not st.session_state.swagger_endpoints)) \
           and uploaded_swagger_file is None: # Only fetch if no file uploaded
            if swagger_url:
                with st.spinner("Fetching Swagger specification..."):
                    try:
                        parser = SwaggerParser(swagger_url)
                        endpoints = parser.extract_endpoints()
                        st.session_state.swagger_endpoints = endpoints
                        st.session_state.swagger_parser = parser
                        st.session_state.current_swagger_url = swagger_url
                        
                        st.session_state.selected_endpoint_keys = []
                        st.session_state.scenario_requests_configs = []
                        st.session_state.jmx_content_download = None
                        st.session_state.csv_content_download = None
                        st.session_state.mapping_metadata_download = None
                        st.session_state.full_swagger_spec_download = None
                        st.session_state.mappings = {}
                        st.session_state.llm_structured_scenario = None # Clear LLM suggestion on new swagger
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
                        st.session_state.llm_structured_scenario = None
            else:
                st.warning("Please enter a Swagger JSON URL.")

        if st.session_state.swagger_endpoints:
            if st.session_state.swagger_parser and st.session_state.swagger_parser.swagger_data:
                st.session_state.full_swagger_spec_download = json.dumps(
                    st.session_state.swagger_parser.get_full_swagger_spec(), indent=2
                )
            
            # Download button for parsed swagger endpoints
            st.download_button(
                label="Download Parsed Swagger Endpoints (JSON)",
                data=json.dumps([ep.to_dict() for ep in st.session_state.swagger_endpoints], indent=2).encode("utf-8"),
                file_name="parsed_swagger_endpoints.json",
                mime="application/json",
                key="download_parsed_swagger_endpoints",
                help="Download the internally parsed Swagger endpoints for re-use."
            )


    with col2:
        st.header("🗄️ Database Configuration")

        st.session_state.db_type_selected = st.selectbox(
            "Database Type",
            ["SQLite", "MySQL", "PostgreSQL", "SQL Server"],
            key="db_type_select",
            index=["SQLite", "MySQL", "PostgreSQL", "SQL Server"].index(st.session_state.db_type_selected)
        )

        db_loaded_from_file = False
        uploaded_db_schema_file = st.file_uploader(
            "Upload DB Schema (JSON)",
            type="json",
            key="upload_db_schema_file",
            help="Upload a previously downloaded JSON file containing database table schemas."
        )
        if uploaded_db_schema_file is not None:
            try:
                st.session_state.db_tables_schema = json.load(uploaded_db_schema_file)
                st.session_state.db_tables = list(st.session_state.db_tables_schema.keys())
                st.success(f"Loaded database schema for {len(st.session_state.db_tables)} tables from file.")
                db_loaded_from_file = True
            except Exception as e:
                st.error(f"Error loading DB schema from file: {e}")
                st.session_state.db_tables_schema = {}
                st.session_state.db_tables = []

        uploaded_db_sampled_data_file = st.file_uploader(
            "Upload Sampled DB Data (JSON)",
            type="json",
            key="upload_db_sampled_data_file",
            help="Upload a previously downloaded JSON file containing sampled database data."
        )
        if uploaded_db_sampled_data_file is not None:
            try:
                raw_sampled_data = json.load(uploaded_db_sampled_data_file)
                st.session_state.db_sampled_data = {
                    table_name: pd.DataFrame(data) 
                    for table_name, data in raw_sampled_data.items()
                }
                st.success(f"Loaded sampled data for {len(st.session_state.db_sampled_data)} tables from file.")
                db_loaded_from_file = True
            except Exception as e:
                st.error(f"Error loading sampled DB data from file: {e}")
                st.session_state.db_sampled_data = {}


        if st.session_state.db_type_selected == "SQLite":
            db_file_path = st.text_input(
                "SQLite Database File Path",
                value="database/petstore.db",
                help="Path to your SQLite database file (e.g., database/petstore.db)"
            )
            db_config = DatabaseConfig(db_type="sqlite", file_path=db_file_path) 
            
        else: # MySQL, PostgreSQL, or SQL Server
            st.warning(f"Note: For {st.session_state.db_type_selected}, ensure `pyodbc` (for SQL Server) and ODBC drivers are installed in your Python environment outside Canvas. Otherwise, schema and data will be dummy.")
            db_host_input = st.text_input("DB Host", value=st.session_state.db_host, key="db_host_input")
            db_user_input = st.text_input("DB Username", value=st.session_state.db_user, key="db_user_input")
            db_password_input = st.text_input("DB Password", type="password", value=st.session_state.db_password, key="db_password_input")
            db_name_input = st.text_input("DB Name", value=st.session_state.db_name, key="db_name_input")
            db_port_input = st.text_input("DB Port (optional)", value=st.session_state.db_port, key="db_port_input")

            port_val = None
            if db_port_input:
                try:
                    port_val = int(db_port_input)
                except ValueError:
                    st.warning("Invalid port number. Please enter an integer or leave blank.")
                    port_val = None

            db_config = DatabaseConfig(
                db_type=st.session_state.db_type_selected.lower().replace(" ", ""),
                host=db_host_input,
                username=db_user_input,
                password=db_password_input,
                database=db_name_input,
                port=port_val
            )

            st.session_state.db_host = db_host_input
            st.session_state.db_user = db_user_input
            st.session_state.db_password = db_password_input
            st.session_state.db_name = db_name_input
            st.session_state.db_port = port_val


        if st.button("Connect Database", key="connect_db") and not db_loaded_from_file:
            if st.session_state.db_type_selected == "SQLite" and not db_file_path:
                st.error("Please enter a SQLite database file path.")
                st.session_state.db_tables = []
                st.session_state.db_connector = None
                st.session_state.db_tables_schema = {}
                st.session_state.db_sampled_data = {}
                st.session_state.mappings = {}
            elif st.session_state.db_type_selected != "SQLite" and not (db_config.host and db_config.username and db_config.database):
                 st.error(f"Please provide Host, Username, and Database for {st.session_state.db_type_selected} connection.")
                 st.session_state.db_tables = []
                 st.session_state.db_connector = None
                 st.session_state.db_tables_schema = {}
                 st.session_state.db_sampled_data = {}
                 st.session_state.mappings = {}
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
                                sampled_data[table] = connector.preview_data(table, limit=3)
                            
                            st.session_state.db_tables_schema = tables_schema
                            st.session_state.db_sampled_data = sampled_data
                            
                            st.session_state.mappings = {} 
                            st.session_state.jmx_content_download = None
                            st.session_state.csv_content_download = None
                            st.session_state.mapping_metadata_download = None
                            st.session_state.llm_structured_scenario = None # Clear LLM suggestion on new db connection

                            st.success(f"Connected to {connector.config.db_type.upper()}! Found {len(tables)} tables.")
                            
                            # Display schema behind a clickable expander
                            if st.session_state.db_tables_schema:
                                with st.expander("View Detailed Database Schema"):
                                    st.json(st.session_state.db_tables_schema)

                            if st.session_state.db_type_selected == "SQLite" and not os.path.exists(db_file_path):
                                if st.button("Create Dummy SQLite DB", key="create_dummy_db_on_connect"):
                                    try:
                                        os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
                                        conn = sqlite3.connect(db_file_path)
                                        cursor = conn.cursor()
                                        cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS pets (
                                                id INTEGER PRIMARY KEY,
                                                name TEXT NOT NULL,
                                                status TEXT NOT NULL,
                                                tags TEXT
                                            );
                                        """)
                                        cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS users (
                                                id INTEGER PRIMARY KEY,
                                                username TEXT NOT NULL,
                                                password TEXT NOT NULL,
                                                email TEXT,
                                                role_id INTEGER
                                            );
                                        """)
                                        cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS orders (
                                                order_id INTEGER PRIMARY KEY,
                                                user_id INTEGER,
                                                status TEXT
                                            );
                                        """)
                                        cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS inventory_items (
                                                item_id INTEGER PRIMARY KEY,
                                                product_id INTEGER,
                                                quantity INTEGER
                                            );
                                        """)
                                        cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS roles (
                                                id INTEGER PRIMARY KEY,
                                                role_name TEXT
                                            );
                                        """)
                                        cursor.execute("INSERT INTO pets (id, name, status, tags) VALUES (1, 'Buddy', 'available', 'dog,friendly');")
                                        cursor.execute("INSERT INTO pets (id, name, status, tags) VALUES (2, 'Whiskers', 'pending', 'cat');")
                                        cursor.execute("INSERT INTO users (id, username, password, email, role_id) VALUES (101, 'testuser', 'testpass', 'test@example.com', 1);")
                                        cursor.execute("INSERT INTO users (id, username, password, email, role_id) VALUES (102, 'user2', 'pass2', 'user2@example.com', 2);")
                                        cursor.execute("INSERT INTO orders (order_id, user_id, status) VALUES (1001, 101, 'pending');")
                                        cursor.execute("INSERT INTO orders (order_id, user_id, status) VALUES (1002, 102, 'completed');")
                                        cursor.execute("INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (1, 1, 50);")
                                        cursor.execute("INSERT INTO inventory_items (item_id, product_id, quantity) VALUES (2, 2, 120);")
                                        cursor.execute("INSERT INTO roles (id, role_name) VALUES (1, 'Admin');")
                                        cursor.execute("INSERT INTO roles (id, role_name) VALUES (2, 'User');")
                                        conn.commit()
                                        conn.close()
                                        st.success(f"Dummy SQLite database created at {db_file_path} with sample data.")
                                        st.rerun()
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
        
        if st.session_state.db_tables_schema:
            st.download_button(
                label="Download DB Schema (JSON)",
                data=json.dumps(st.session_state.db_tables_schema, indent=2).encode("utf-8"),
                file_name="db_schema.json",
                mime="application/json",
                key="download_db_schema",
                help="Download the fetched database schema for re-use."
            )
        if st.session_state.db_sampled_data:
            # Convert DataFrames to dicts for JSON serialization
            sampled_data_for_download = {
                table_name: df.to_dict(orient='records') 
                for table_name, df in st.session_state.db_sampled_data.items()
            }
            st.download_button(
                label="Download Sampled DB Data (JSON)",
                data=json.dumps(sampled_data_for_download, indent=2, cls=CustomJSONEncoder).encode("utf-8"), # Use custom encoder
                file_name="db_sampled_data.json",
                mime="application/json",
                key="download_db_sampled_data",
                help="Download the sampled database data for re-use."
            )

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
        
        if st.button("Get AI Script Design", key="get_ai_design_btn"):
            if not st.session_state.swagger_endpoints:
                st.error("Please fetch Swagger specification or upload parsed endpoints first to get AI suggestions.")
            elif not st.session_state.db_tables_schema or not st.session_state.db_sampled_data:
                st.error("Please connect to database or upload schema/sampled data first to get AI suggestions for mappings.")
            elif not st.session_state.gemini_api_key.strip():
                st.error("Please provide your Gemini API Key to get AI suggestions.")
            else:
                with st.spinner("Getting AI detailed script design..."):
                    st.session_state.mappings = DataMapper.suggest_mappings(
                        st.session_state.swagger_endpoints,
                        st.session_state.db_tables_schema,
                        st.session_state.db_sampled_data
                    )
                        
                    structured_scenario = call_llm_for_scenario_plan(
                        prompt=prompt,
                        swagger_endpoints=st.session_state.swagger_endpoints,
                        db_tables_schema=st.session_state.db_tables_schema,
                        db_sampled_data=st.session_state.db_sampled_data,
                        thread_group_users=st.session_state.num_users_input,
                        ramp_up_time=st.session_state.ramp_up_time_input,
                        loop_count=st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1,
                        api_key=st.session_state.gemini_api_key
                    )
                    
                    if structured_scenario:
                        st.session_state.llm_structured_scenario = structured_scenario
                        st.success("AI script design generated!")
                        st.subheader("AI-Generated JMeter Scenario Details (Streaming)")
                        
                        # Display requests in a streaming fashion
                        for i, req in enumerate(structured_scenario):
                            st.markdown(f"**Request {i+1}: {req.get('method')} {req.get('path')}** (Name: `{req.get('name')}`)")
                            st.write(f"Description: {req.get('description', 'No description provided.')}")
                            
                            if req.get('parameters_and_body_fields'):
                                st.markdown("Parameters/Body Fields:")
                                for param in req['parameters_and_body_fields']:
                                    source_detail = ""
                                    if param['source'] == 'from_csv':
                                        source_detail = f" (from DB CSV: `{param.get('table_name')}.{param.get('column_name')}`)"
                                    elif param['source'] == 'from_extraction':
                                        source_detail = f" (from Extraction: request `{param.get('source_request_name')}`, variable `{param.get('extracted_variable_name')}`)"
                                    elif param['source'] == 'static_value':
                                        source_detail = f" (Static Value: `{param.get('value')}`)"
                                    elif param['source'] == 'generate_dummy':
                                        source_detail = f" (Generated Dummy: type `{param.get('type')}`)"
                                    st.markdown(f"- `{param.get('name')}` (in `{param.get('in')}`): **`{param['source']}`**{source_detail}")
                            
                            if req.get('assertions'):
                                st.markdown("Assertions:")
                                for ass in req['assertions']:
                                    st.markdown(f"- Type: `{ass.get('type')}`, Value: `{ass.get('value')}`")
                            
                            if req.get('extractions'):
                                st.markdown("Extractions:")
                                for ext in req['extractions']:
                                    st.markdown(f"- JSONPath: `{ext.get('json_path')}` -> Var: `{ext.get('var_name')}`")
                            
                            if req.get('think_time_ms') is not None:
                                st.markdown(f"Think Time: `{req.get('think_time_ms')}` ms")
                            
                            st.markdown("---") # Separator for each request
                            time.sleep(0.1) # Simulate streaming delay
                            
                    else:
                        st.session_state.llm_structured_scenario = None
                        st.error("Failed to generate structured AI design.")

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
                value=st.session_state.loop_count_input_specific,
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
            type="password",
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

    st.header("2.3 Scenario Thread Grouping")
    st.session_state.enable_setup_teardown_thread_groups = st.checkbox(
        "Enable Setup and Teardown Thread Groups",
        value=st.session_state.enable_setup_teardown_thread_groups,
        key="enable_setup_teardown_thread_groups_checkbox",
        help="If checked, JMeter will generate separate Setup and Teardown Thread Groups for pre/post test actions."
    )

    st.header("3. Select Endpoints for Scenario")
    if st.session_state.swagger_endpoints:
        endpoint_options = [f"{ep.method} {ep.path}" for ep in st.session_state.swagger_endpoints]
        
        select_all_toggle = st.checkbox("Select All Endpoints (for AI initial design)", key="select_all_endpoints_checkbox")
        
        default_selected_endpoints_for_ai = []
        if select_all_toggle:
            default_selected_endpoints_for_ai = endpoint_options
        else:
            default_selected_endpoints_for_ai = st.session_state.selected_endpoint_keys


        selected_endpoint_keys_for_ai_input = st.multiselect(
            "Select API Endpoints to include in AI's initial scenario design (order does NOT reflect execution sequence, AI determines flow)",
            options=endpoint_options,
            default=default_selected_endpoints_for_ai,
            key="endpoint_selector_for_ai"
        )
        st.session_state.selected_endpoint_keys = selected_endpoint_keys_for_ai_input 

        if st.button("Refine Scenario Manually", key="refine_scenario_btn"):
            st.session_state.llm_structured_scenario = None # Clear AI generated scenario
            new_scenario_configs = []
            
            st.session_state.mappings = DataMapper.suggest_mappings(
                st.session_state.swagger_endpoints,
                st.session_state.db_tables_schema,
                st.session_state.db_sampled_data
            )

            extracted_variables_map = {} 

            if st.session_state.enable_auth_flow:
                login_endpoint = next((ep for ep in st.session_state.swagger_endpoints if ep.path == st.session_state.auth_login_endpoint_path and ep.method == st.session_state.auth_login_method), None)
                if login_endpoint:
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
                    st.warning(f"Login endpoint {st.session_state.auth_login_method} {st.session_state.auth_login_endpoint_path} not found in Swagger spec for manual config. Authentication flow might not work as expected.")


            full_swagger_dict = st.session_state.swagger_parser.get_full_swagger_spec()
            base_path_from_swagger = full_swagger_dict.get('basePath', '/')
            if not base_path_from_swagger.startswith('/'):
                base_path_from_swagger = '/' + base_path_from_swagger
            if base_path_from_swagger != '/' and base_path_from_swagger.endswith('/'):
                base_path_from_swagger = base_path_from_swagger.rstrip('/')

            for ep_key in st.session_state.selected_endpoint_keys:
                method_str, path_str = ep_key.split(' ', 1)
                method_key = method_str.lower()

                resolved_endpoint_data = full_swagger_dict.get('paths', {}).get(path_str, {}).get(method_key, {})

                if resolved_endpoint_data:
                    clean_path_name = re.sub(r'[^\w\s-]', '', path_str).replace('/', '_').strip('_')
                    operation_id = resolved_endpoint_data.get('operationId')
                    request_name = f"{method_str}_{operation_id or clean_path_name}"
                    jmeter_formatted_path = path_str 
                    
                    request_config = {
                        "endpoint_key": ep_key,
                        "name": request_name,
                        "method": method_str,
                        "path": "",
                        "parameters": {},
                        "headers": {},
                        "body": None,
                        "assertions": [],
                        "json_extractors": [],
                        "think_time": 0
                    }

                    if 'parameters' in resolved_endpoint_data:
                        for param in resolved_endpoint_data['parameters']:
                            if param.get('in') == 'path':
                                param_name = param['name']
                                jmeter_var_for_path_param = f"${{{param_name}}}"
                                mapping_info = st.session_state.mappings.get(ep_key, {}).get(param_name)
                                if mapping_info:
                                    if mapping_info['source'] == "DB Sample (CSV)":
                                        jmeter_var_for_path_param = mapping_info['value']
                                    elif mapping_info['source'] == "Generated Value":
                                        jmeter_var_for_path_param = mapping_info['value']
                                    else: 
                                        jmeter_var_for_path_param = str(mapping_info['value'])
                                elif param_name.lower() in extracted_variables_map:
                                    jmeter_var_for_path_param = extracted_variables_map[param_name.lower()]
                                jmeter_formatted_path = re.sub(r'\{' + re.escape(param_name) + r'\}', jmeter_var_for_path_param, jmeter_formatted_path)
                    
                    final_request_path_for_jmeter = jmeter_formatted_path
                    if base_path_from_swagger != '/' and final_request_path_for_jmeter.startswith(base_path_from_swagger):
                        final_request_path_for_jmeter = final_request_path_for_jmeter[len(base_path_from_swagger):]
                        if not final_request_path_for_jmeter.startswith('/'):
                            final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                        if final_request_path_for_jmeter == "//":
                            final_request_path_for_jmeter = "/"
                    if final_request_path_for_jmeter and not final_request_path_for_jmeter.startswith('/'):
                         final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                    request_config["path"] = final_request_path_for_jmeter

                    if 'parameters' in resolved_endpoint_data:
                        for param in resolved_endpoint_data['parameters']:
                            if param.get('in') == 'header':
                                header_name = param['name']
                                mapped_value = None
                                mapping_info = st.session_state.mappings.get(ep_key, {}).get(header_name)
                                if mapping_info:
                                    mapped_value = mapping_info['value']
                                elif header_name.lower() in extracted_variables_map:
                                    mapped_value = extracted_variables_map[header_name.lower()]
                                if mapped_value:
                                    request_config['headers'][header_name] = str(mapped_value)
                                elif param.get('required'):
                                    request_config['headers'][header_name] = f"dummy_{header_name}"

                    if st.session_state.enable_auth_flow and ep_key != f"{st.session_state.auth_login_method} {st.session_state.auth_login_endpoint_path}":
                        if "authtoken" in extracted_variables_map:
                            request_config["headers"][st.session_state.auth_header_name] = f"{st.session_state.auth_header_prefix}{extracted_variables_map['authtoken']}"
                        else:
                            st.warning(f"Authentication flow enabled but auth token not found for {ep_key}. Check login configuration.")

                    if 'parameters' in resolved_endpoint_data:
                        for param in resolved_endpoint_data['parameters']:
                            if param.get('in') == 'query':
                                if param['name'].lower() in extracted_variables_map:
                                    request_config['parameters'][param['name']] = extracted_variables_map[param['name'].lower()]
                                else:
                                    mapping_info = st.session_state.mappings.get(ep_key, {}).get(param['name'])
                                    if mapping_info:
                                        request_config['parameters'][param['name']] = mapping_info['value']
                                    else:
                                        request_config['parameters'][param['name']] = "<<NO_MATCH_FOUND>>"

                    if method_key in ["post", "put", "patch"]:
                        content_type_header = "application/json"
                        if resolved_endpoint_data.get('consumes'):
                            content_type_header = resolved_endpoint_data['consumes'][0]
                        elif 'requestBody' in resolved_endpoint_data and 'content' in resolved_endpoint_data['requestBody']:
                            first_content_type = next(iter(resolved_endpoint_data['requestBody'].get('content', {})), None)
                            if first_content_type:
                                content_type_header = first_content_type
                        request_config["headers"]["Content-Type"] = content_type_header

                        request_body_schema = None
                        if 'requestBody' in resolved_endpoint_data:
                            content_types = resolved_endpoint_data['requestBody'].get('content', {})
                            if content_types:
                                if 'application/json' in content_types:
                                    request_body_schema = content_types['application/json'].get('schema')
                                else:
                                    request_body_schema = next(iter(content_types.values())).get('schema')
                    else:
                        for param in resolved_endpoint_data.get('parameters', []):
                            if param.get('in') == 'body' and 'schema' in param:
                                request_body_schema = param['schema']
                                break
                        
                        dummy_llm_param_fields = []
                        if ep_key in st.session_state.mappings:
                            for param_path, mapping_data in st.session_state.mappings[ep_key].items():
                                if 'in' in mapping_data and mapping_data['in'] == 'body': # This checks the 'in' property of mapping_data itself
                                    dummy_llm_param_fields.append({
                                        "name": param_path, # Use full path for nested params
                                        "in": "body",
                                        "source": mapping_data['source'],
                                        "value": mapping_data['value'],
                                        "table_name": mapping_data.get('table_name'),
                                        "column_name": mapping_data.get('column_name'),
                                        "type": mapping_data.get('type'),
                                        "format": mapping_data.get('format'), # Pass format too for dummy generation
                                        "enum": mapping_data.get('enum'),
                                        "minimum": mapping_data.get('minimum'),
                                        "maximum": mapping_data.get('maximum'),
                                        "minLength": mapping_data.get('minLength'),
                                        "maxLength": mapping_data.get('maxLength')
                                    })
                                elif '.' in param_path and (param_path.split('.')[0] in ['body', 'requestBody']): # Handle cases where root might be 'body' or 'requestBody'
                                    # This is a bit of a hack to get the nested path right for the _build_recursive_json_body_with_llm_guidance
                                    # It implies that the 'name' in mapping for body would already be 'body.nested_field'
                                    dummy_llm_param_fields.append({
                                        "name": param_path,
                                        "in": "body",
                                        "source": mapping_data['source'],
                                        "value": mapping_data['value'],
                                        "table_name": mapping_data.get('table_name'),
                                        "column_name": mapping_data.get('column_name'),
                                        "type": mapping_data.get('type'),
                                        "format": mapping_data.get('format'),
                                        "enum": mapping_data.get('enum'),
                                        "minimum": mapping_data.get('minimum'),
                                        "maximum": mapping_data.get('maximum'),
                                        "minLength": mapping_data.get('minLength'),
                                        "maxLength": mapping_data.get('maxLength')
                                    })

                        if request_body_schema:
                            try:
                                generated_body = _build_recursive_json_body_with_llm_guidance(
                                    request_body_schema,
                                    dummy_llm_param_fields, # This should be dummy_llm_param_fields if manual refine needs it.
                                                                        # LLM design already passes proper fields.
                                    [], 
                                    extracted_variables_map,
                                    ep_key
                                )
                                if isinstance(generated_body, (dict, list, int, float, bool)):
                                    request_config["body"] = json.dumps(generated_body, indent=2)
                                else:
                                    request_config["body"] = str(generated_body)
                                    
                            except Exception as e:
                                logger.error(f"Error building recursive JSON body for {ep_key} during manual refine: {e}", exc_info=True)
                                request_config["body"] = "{\n  \"message\": \"Error building dynamic body for bodySchema from full spec\"\n}"
                        else:
                            request_config["body"] = "{\n  \"message\": \"auto-generated dummy body (no schema found in full spec)\"\n}"

                    if st.session_state.include_scenario_assertions:
                        request_config['assertions'].append({"type": "Response Code", "value": "200"})
                    
                    for status_code, response_obj in resolved_endpoint_data.get('responses', {}).items():
                        if status_code.startswith('2'):
                            resp_schema = None
                            if 'schema' in response_obj:
                                resp_schema = response_obj['schema']
                            elif 'content' in response_obj:
                                # For OpenAPI 3.0+ responses using 'content'
                                content_types_resp = response_obj['content']
                                if 'application/json' in content_types_resp:
                                    resp_schema = content_types_resp['application/json'].get('schema')
                                else:
                                    # Try to get schema from any content type if JSON not found
                                    resp_schema = next(iter(content_types_resp.values()), {}).get('schema')
                            
                            if resp_schema and 'properties' in resp_schema:
                                for prop_name, prop_details in resp_schema['properties'].items():
                                    if (prop_name.lower() == 'id' and prop_details.get('type') in ['string', 'integer']) or \
                                       (prop_name.lower() == 'username' and prop_details.get('type') == 'string'):
                                        var_base_name = prop_name.replace('_', '').lower()
                                        correlated_var_name = f"{operation_id or clean_path_name}{var_base_name.capitalize()}"
                                        request_config['json_extractors'].append({
                                            "json_path_expr": f"$.{prop_name}",
                                            "var_name": correlated_var_name
                                        })
                                        extracted_variables_map[prop_name.lower()] = f"${{{correlated_var_name}}}"

                            if resp_schema and resp_schema.get('type') == 'array' and 'items' in resp_schema:
                                item_schema = resp_schema['items']
                                if 'properties' in item_schema:
                                    for prop_name, prop_details in item_schema['properties'].items():
                                        if (prop_name.lower() == 'id' and prop_details.get('type') in ['string', 'integer']) or \
                                           (prop_name.lower() == 'username' and prop_details.get('type') == 'string'):
                                            var_base_name = prop_name.replace('_', '').lower()
                                            correlated_var_name = f"{operation_id or clean_path_name}First{var_base_name.capitalize()}"
                                            request_config['json_extractors'].append({
                                                "json_path_expr": f"$[0].{prop_name}",
                                                "var_name": correlated_var_name
                                            })
                                            extracted_variables_map[prop_name.lower()] = f"${{{correlated_var_name}}}"
                            
                    new_scenario_configs.append(request_config)
                else:
                    logger.warning(f"Could not find fully resolved endpoint data for {ep_key}. Skipping.")
                    
            st.session_state.scenario_requests_configs = new_scenario_configs
            st.rerun()
        
        if st.session_state.llm_structured_scenario:
            st.markdown("---")
            # The subheader is moved here to appear only after successful AI generation.
            st.subheader("AI-Designed JMeter Scenario Details (Streaming)")
            llm_designed_configs = []
            
            st.session_state.mappings = DataMapper.suggest_mappings(
                st.session_state.swagger_endpoints,
                st.session_state.db_tables_schema,
                st.session_state.db_sampled_data
            )

            extracted_variables_map = {}

            full_swagger_dict = st.session_state.swagger_parser.get_full_swagger_spec()
            base_path_from_swagger = full_swagger_dict.get('basePath', '/')
            if not base_path_from_swagger.startswith('/'):
                base_path_from_swagger = '/' + base_path_from_swagger
            if base_path_from_swagger != '/' and base_path_from_swagger.endswith('/'):
                base_path_from_swagger = base_path_from_swagger.rstrip('/')

            for llm_req_design in st.session_state.llm_structured_scenario:
                method_str = llm_req_design.get('method')
                path_str = llm_req_design.get('path')
                request_name = llm_req_design.get('name', f"{method_str}_{path_str.replace('/','_').strip('_')}")
                think_time = llm_req_design.get('think_time_ms', 0)

                ep_key = f"{method_str} {path_str}"
                resolved_endpoint_data = full_swagger_dict.get('paths', {}).get(path_str, {}).get(method_str.lower(), {})

                if not resolved_endpoint_data:
                    st.warning(f"AI suggested endpoint {ep_key} not found in Swagger spec. Skipping.")
                    continue

                jmeter_formatted_path = path_str
                request_config = {
                    "endpoint_key": ep_key,
                    "name": request_name,
                    "method": method_str,
                    "path": "", 
                    "parameters": {}, 
                    "headers": {},
                    "body": None,
                    "assertions": [],
                    "json_extractors": [],
                    "think_time": think_time
                }

                body_fields_for_recursive_builder = []
                for param_field in llm_req_design.get('parameters_and_body_fields', []):
                    param_name = param_field['name']
                    param_in = param_field['in']
                    source_strategy = param_field['source']
                    
                    resolved_value = None

                    if param_in == 'body':
                        # For body fields, we collect them and process with the recursive builder
                        body_fields_for_recursive_builder.append(param_field)
                        continue # Skip to next param_field, as body is processed later
                    
                    if source_strategy == "from_csv":
                        table_name = param_field.get('table_name')
                        column_name = param_field.get('column_name')
                        if table_name and column_name:
                            resolved_value = f"${{csv_{table_name}_{column_name}}}"
                        else:
                            logger.warning(f"LLM suggested 'from_csv' for {param_name} but missing table_name/column_name. Falling back to dummy.")
                            resolved_value = DataMapper._generate_dummy_value(param_field)
                    elif source_strategy == "from_extraction":
                        extracted_var_name = param_field.get('extracted_variable_name')
                        prefix = param_field.get('prefix', '')
                        if extracted_var_name in extracted_variables_map:
                             resolved_value = f"{prefix}{extracted_variables_map[extracted_var_name]}"
                        else:
                             logger.warning(f"LLM suggested 'from_extraction' for {param_name} but '{extracted_var_name}' not found in extracted_variables_map. Falling back to dummy.")
                             resolved_value = DataMapper._generate_dummy_value(param_field)
                    elif source_strategy == "generate_dummy":
                        resolved_value = DataMapper._generate_dummy_value(param_field)
                    elif source_strategy == "static_value":
                        resolved_value = param_field.get('value')
                    
                    if param_in == 'path':
                        jmeter_formatted_path = re.sub(r'\{' + re.escape(param_name) + r'\}', str(resolved_value), jmeter_formatted_path)
                    elif param_in == 'query':
                        request_config['parameters'][param_name] = str(resolved_value)
                    elif param_in == 'header':
                        request_config['headers'][param_name] = str(resolved_value)


                final_request_path_for_jmeter = jmeter_formatted_path
                if base_path_from_swagger != '/' and final_request_path_for_jmeter.startswith(base_path_from_swagger):
                    final_request_path_for_jmeter = final_request_path_for_jmeter[len(base_path_from_swagger):]
                    if not final_request_path_for_jmeter.startswith('/'):
                        final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                    if final_request_path_for_jmeter == "//":
                        final_request_path_for_jmeter = "/"
                if final_request_path_for_jmeter and not final_request_path_for_jmeter.startswith('/'):
                    final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                request_config["path"] = final_request_path_for_jmeter

                if method_str.lower() in ["post", "put", "patch"]:
                    content_type_header = "application/json"
                    if resolved_endpoint_data.get('consumes'):
                        content_type_header = resolved_endpoint_data['consumes'][0]
                    elif 'requestBody' in resolved_endpoint_data and 'content' in resolved_endpoint_data['requestBody']:
                        first_content_type = next(iter(resolved_endpoint_data['requestBody'].get('content', {})), None)
                        if first_content_type:
                            content_type_header = first_content_type
                    request_config["headers"]["Content-Type"] = content_type_header

                    request_body_schema = None
                    if 'requestBody' in resolved_endpoint_data:
                        content_types = resolved_endpoint_data['requestBody'].get('content', {})
                        if content_types:
                            if 'application/json' in content_types:
                                request_body_schema = content_types['application/json'].get('schema')
                            else:
                                request_body_schema = next(iter(content_types.values())).get('schema')
                    else:
                        for param in resolved_endpoint_data.get('parameters', []):
                            if param.get('in') == 'body' and 'schema' in param:
                                request_body_schema = param['schema']
                                break
                    
                    if request_body_schema:
                        try:
                            generated_body = _build_recursive_json_body_with_llm_guidance(
                                request_body_schema, 
                                body_fields_for_recursive_builder,
                                [], 
                                extracted_variables_map,
                                ep_key
                            )
                            if isinstance(generated_body, (dict, list, int, float, bool)):
                                request_config["body"] = json.dumps(generated_body, indent=2)
                            else:
                                request_config["body"] = str(generated_body)
                        except Exception as e:
                            logger.error(f"Error building recursive JSON body for {ep_key} from LLM design: {e}", exc_info=True)
                            request_config["body"] = "{\n  \"message\": \"Error building dynamic body from AI design\"\n}"
                    else:
                        request_config["body"] = "{\n  \"message\": \"auto-generated dummy body (no schema found for AI design)\"\n}"

                for assertion_data in llm_req_design.get('assertions', []):
                    request_config['assertions'].append(assertion_data)

                for extractor_data in llm_req_design.get('extractions', []):
                    request_config['json_extractors'].append({
                        "json_path_expr": extractor_data['json_path'],
                        "var_name": extractor_data['var_name']
                    })
                    extracted_variables_map[extractor_data['var_name'].lower()] = f"${{{extractor_data['var_name']}}}"

                llm_designed_configs.append(request_config)
            
            st.session_state.scenario_requests_configs = llm_designed_configs
            st.success("Scenario configured based on AI design!")

        st.markdown("---")
        st.subheader("Current Scenario Configuration")
        if st.session_state.scenario_requests_configs:
            for i, config in enumerate(st.session_state.scenario_requests_configs):
                st.code(f"Request {i+1}: {config['method']} {config['path']} (Name: {config['name']})")
                with st.expander(f"View Details for {config['name']}"):
                    st.json(config)
        else:
            st.info("No scenario configured yet. Use 'Get AI Script Design' or 'Refine Scenario Manually'.")


    st.markdown("---")

    if st.button("Generate JMeter Script", key="generate_button_final", type="primary"):
        if not st.session_state.swagger_endpoints:
            st.error("Please fetch Swagger specification or upload parsed endpoints and ensure endpoints are parsed.")
            return
        if not st.session_state.scenario_requests_configs:
            st.error("Please generate an AI script design or refine the scenario manually before generating JMeter script.")
            return
        if not st.session_state.db_tables_schema or not st.session_state.db_sampled_data:
            st.error("Please connect to database or upload schema/sampled data if your scenario relies on DB data.")

        st.info("Generating JMeter script... Please wait.")

        try:
            scenario_plan = {"requests": st.session_state.scenario_requests_configs}
            
            num_users = st.session_state.num_users_input
            ramp_up_time = st.session_state.ramp_up_time_input
            loop_count = st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1
            
            global_constant_timer_delay = st.session_state.constant_timer_delay_ms if st.session_state.enable_constant_timer else 0

            csv_data_for_jmeter = {}
            csv_headers = set()

            for endpoint_key, params_map in st.session_state.mappings.items():
                for param_name, mapping_info in params_map.items():
                    if mapping_info['source'] == "DB Sample (CSV)":
                        # The JMeter variable name format expected by DataMapper is csv_<table_name>_<column_name>
                        # Ensure this is correctly extracted/formed from mapping_info
                        if 'table_name' in mapping_info and 'column_name' in mapping_info:
                            jmeter_var_name_raw = f"csv_{mapping_info['table_name']}_{mapping_info['column_name']}"
                        else:
                            # Fallback if table_name or column_name is missing from mapping_info
                            # This should ideally not happen if DataMapper.suggest_mappings is robust
                            logger.warning(f"Mapping info for {param_name} (from DB Sample) missing table_name/column_name. Cannot generate CSV.")
                            continue # Skip this mapping for CSV generation


                        if mapping_info['table_name'] in st.session_state.db_sampled_data and \
                           mapping_info['column_name'] in st.session_state.db_sampled_data[mapping_info['table_name']].columns:
                            
                            if jmeter_var_name_raw not in csv_data_for_jmeter:
                                csv_data_for_jmeter[jmeter_var_name_raw] = st.session_state.db_sampled_data[mapping_info['table_name']][mapping_info['column_name']].tolist()
                                csv_headers.add(jmeter_var_name_raw)
                            else:
                                if len(csv_data_for_jmeter[jmeter_var_name_raw]) < len(st.session_state.db_sampled_data[mapping_info['table_name']][mapping_info['column_name']]):
                                    csv_data_for_jmeter[jmeter_var_name_raw] = st.session_state.db_sampled_data[mapping_info['table_name']][mapping_info['column_name']].tolist()


            generated_csv_content = None
            if csv_headers and csv_data_for_jmeter:
                csv_headers_list = sorted(list(csv_headers))
                generated_csv_content = ",".join(csv_headers_list) + "\n"
                
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
            
            # Use the swagger_parser to get the base URL
            current_full_swagger_spec_dict = {}
            if st.session_state.swagger_parser and st.session_state.swagger_parser.swagger_data:
                parsed_url_from_swagger_data = urlparse(st.session_state.swagger_parser.swagger_data.get('host', st.session_state.current_swagger_url))
                protocol = st.session_state.swagger_parser.swagger_data.get('schemes', ['https'])[0]
                domain = parsed_url_from_swagger_data.hostname
                port = parsed_url_from_swagger_data.port if parsed_url_from_swagger_data.port else ""
                base_path_for_http_defaults = st.session_state.swagger_parser.swagger_data.get('basePath', '/')
                if not base_path_for_http_defaults.startswith('/'):
                    base_path_for_http_defaults = '/' + base_path_for_http_defaults
                if base_path_for_http_defaults != '/' and base_path_for_http_defaults.endswith('/'):
                    base_path_for_http_defaults = base_path_for_http_defaults.rstrip('/')
                current_full_swagger_spec_dict = st.session_state.swagger_parser.get_full_swagger_spec()
            else: # Fallback if swagger_parser is not set or data missing
                parsed_url = urlparse(swagger_url) # Use the input URL as fallback
                protocol = parsed_url.scheme
                domain = parsed_url.hostname
                port = parsed_url.port if parsed_url.port else ""
                base_path_for_http_defaults = "/"


            jmx_content, _ = generator.generate_jmx(
                app_base_url=swagger_url, # This is just a string, not used for parsing anymore
                thread_group_users=num_users,
                ramp_up_time=ramp_up_time,
                loop_count=loop_count,
                scenario_plan=scenario_plan, 
                csv_data_to_include=generated_csv_content,
                global_constant_timer_delay=global_constant_timer_delay,
                test_plan_name=st.session_state.test_plan_name, 
                thread_group_name=st.session_state.thread_group_name,
                http_defaults_protocol=protocol,
                http_defaults_domain=domain,
                http_defaults_port=port,
                http_defaults_base_path=base_path_for_http_defaults,
                full_swagger_spec=current_full_swagger_spec_dict,
                enable_setup_teardown_thread_groups=st.session_state.enable_setup_teardown_thread_groups
            )

            st.session_state.jmx_content_download = jmx_content
            st.session_state.csv_content_download = generated_csv_content
            st.session_state.mapping_metadata_download = json.dumps(st.session_state.mappings, indent=2)

            st.success("JMeter script generated successfully!")

        except Exception as e:
            st.error(f"An error occurred during script generation: {e}")
            logger.error(f"Error in main app execution: {e}", exc_info=True)

    st.subheader("Download Generated Files")
    if st.session_state.full_swagger_spec_download:
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
    elif st.session_state.jmx_content_download:
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
    if not os.path.exists("swagger.json"):
        dummy_swagger_content = """
{
  "swagger": "2.0",
  "info": {
    "description": "This is a sample server Petstore server.  You can find out more about Swagger at [http://swagger.io](http://swagger.io) or on [irc.freenode.net, #swagger](http://swagger.io/irc/).  For this sample, you can use the api key `special-key` to test the authorization filters.\",
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
                    ]
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
                    ]
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
                "type": "integer"
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
                }
              },
              "xml": {
                "name": "Tag"
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
                CREATE TABLE IF NOT EXISTS pets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tags TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT,
                    role_id INTEGER
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    status TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventory_items (
                    item_id INTEGER PRIMARY KEY,
                    product_id INTEGER,
                    quantity INTEGER
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    id INTEGER PRIMARY KEY,
                    role_name TEXT
                );
            """)
            cursor.execute("INSERT INTO pets (id, name, status, tags) VALUES (1, 'Buddy', 'available', 'dog,friendly');")
            cursor.execute("INSERT INTO pets (id, name, status, tags) VALUES (2, 'Whiskers', 'pending', 'cat');")
            cursor.execute("INSERT INTO users (id, username, password, email, role_id) VALUES (101, 'testuser', 'testpass', 'test@example.com', 1);")
            cursor.execute("INSERT INTO users (id, username, password, email, role_id) VALUES (102, 'user2', 'pass2', 'user2@example.com', 2);")
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
