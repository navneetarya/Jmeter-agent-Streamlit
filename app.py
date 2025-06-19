import streamlit as st
import requests
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
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


def _match_swagger_path_with_generated_path(swagger_endpoints: List[SwaggerEndpoint], method: str, generated_path: str) -> Tuple[Optional[SwaggerEndpoint], Dict[str, str]]:
    """
    Attempts to match a generated path from LLM to an actual Swagger endpoint,
    extracting path parameters if the generated path contains concrete values.
    Returns the matched SwaggerEndpoint and a dict of extracted path params.
    """
    extracted_params = {}
    
    for ep in swagger_endpoints:
        if ep.method.upper() != method.upper():
            continue

        # Convert Swagger path to a regex pattern
        swagger_path_regex_pattern = ep.path
        path_param_names = []
        
        # Identify path parameters and build regex
        # Replace {paramName} with a named capture group for regex
        for param in ep.parameters:
            if param.get('in') == 'path':
                param_name = param['name']
                # Escape existing curly braces in the swagger path, then replace param placeholders
                swagger_path_regex_pattern = swagger_path_regex_pattern.replace(f"{{{param_name}}}", f"(?P<{param_name}>[^/]+)")
                path_param_names.append(param_name)
        
        # Ensure the regex matches the full path
        swagger_path_regex_pattern = "^" + swagger_path_regex_pattern + "$"
        
        match = re.match(swagger_path_regex_pattern, generated_path)
        
        if match:
            # Extract concrete values for path parameters
            for p_name in path_param_names:
                if p_name in match.groupdict():
                    extracted_params[p_name] = match.group(p_name)
            return ep, extracted_params
            
    return None, {}


def call_llm_for_scenario_plan(llm_prompt_input: str, swagger_endpoints: List[SwaggerEndpoint], # Renamed prompt to llm_prompt_input
                               db_tables_schema: Dict[str, List[Dict[str, Any]]], # Changed to Any for schema details
                               db_sampled_data: Dict[str, pd.DataFrame], # Added sampled data for LLM context
                               thread_group_users: int,
                               ramp_up_time: int,
                               loop_count: int,
                               include_assertions_flag: bool, # New flag for assertions
                               enable_auth_flow_flag: bool, # New flag for auth flow
                               api_key: str) -> Optional[List[Dict[str, Any]]]: # Now returns a list of scenario requests
    """
    Calls an LLM to generate a detailed, structured test plan (scenario plan)
    in JSON format, including parameter sourcing, assertions, and extractions.
    """

    protocol_segment = "https://"
    domain_segment = "generativelanguage.googleapis.com"
    path_segment = "/v1beta/models/gemini-2.0-flash:generateContent?key="

    temp_base_api_url = "".join([protocol_segment, domain_segment, path_segment])
    base_api_url_cleaned = _clean_url_string(temp_base_api_url)
    api_url = base_api_url_cleaned + api_key

    # Filter swagger_endpoints based on user prompt for specific endpoints or methods
    filtered_swagger_endpoints = []
    lower_prompt = llm_prompt_input.lower() # Use the input prompt here

    # Default to all if no specific filter is found
    found_specific_filter = False

    # Improved logic to find multiple specific endpoints in the prompt
    # Look for patterns like "GET /path" or "POST /another/path"
    # This regex is more robust for extracting multiple endpoint paths
    specific_endpoint_patterns = re.findall(r'(get|post|put|delete|patch)\s+([/\w-]+(?:/[{\w}]+)*)', lower_prompt)
    
    if specific_endpoint_patterns:
        st.info(f"Detected specific endpoint requests in prompt: {specific_endpoint_patterns}")
        for method, path in specific_endpoint_patterns:
            for ep in swagger_endpoints:
                # Normalize paths for comparison (e.g., /v1/detailsbyappcds vs /v1/detailsbyappcds)
                normalized_ep_path = ep.path.rstrip('/')
                normalized_req_path = path.rstrip('/')
                # Simple exact match for now, could be extended to use path parameters like /{id}
                if ep.method.lower() == method.lower() and normalized_ep_path == normalized_req_path:
                    filtered_swagger_endpoints.append(ep)
                    found_specific_filter = True
        if not filtered_swagger_endpoints and specific_endpoint_patterns:
            st.warning("No matching endpoints found in Swagger spec for the requested paths in the prompt. Proceeding with all available endpoints.")
            filtered_swagger_endpoints = swagger_endpoints # Fallback if specific endpoints not found
    elif "only include get endpoints" in lower_prompt:
        filtered_swagger_endpoints = [ep for ep in swagger_endpoints if ep.method == "GET"]
        found_specific_filter = True
    elif "only include post endpoints" in lower_prompt:
        filtered_swagger_endpoints = [ep for ep in swagger_endpoints if ep.method == "POST"]
        found_specific_filter = True
    elif "design scripts for specific endpoints:" in lower_prompt:
        # Extract specific paths from the prompt using regex
        specific_paths = re.findall(r'/\w+/\w+/[a-zA-Z0-9/]+', lower_prompt) # More general regex for paths
        if specific_paths:
            for path in specific_paths:
                # Need to iterate through all methods for the specified path
                for ep in swagger_endpoints:
                    # Match path exactly (after potential stripping of method)
                    if path.strip() in ep.path: # Using 'in' for flexibility, could be exact match too
                        filtered_swagger_endpoints.append(ep)
            found_specific_filter = True
    elif "/v1/detailsbyappcds" in lower_prompt: # Explicitly checking for the user's requested endpoint
        for ep in swagger_endpoints:
            if ep.path == "/v1/detailsbyappcds" and ep.method == "GET":
                filtered_swagger_endpoints.append(ep)
        found_specific_filter = True
    elif "/api/v1/application/minthreadcount" in lower_prompt: # Explicitly checking for the user's requested endpoint
        for ep in swagger_endpoints:
            if ep.path == "/api/v1/application/minthreadcount" and ep.method == "GET":
                filtered_swagger_endpoints.append(ep)
        found_specific_filter = True


    if not found_specific_filter: # If no explicit filtering instruction, include all
        filtered_swagger_endpoints = swagger_endpoints

    # Now, use filtered_swagger_endpoints for swagger_summary
    swagger_summary = []
    for ep in filtered_swagger_endpoints:
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
            "path": ep.path, # Crucial: Send the template path to the LLM
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

    # Construct the final prompt using the input prompt from the UI
    # UPDATED: Enhanced LLM prompt for assertions, headers, and CSV variables
    final_llm_prompt = f"""
    {llm_prompt_input}

    **Input Context:**
    -   **Swagger Endpoints Summary:** ```json\n{json.dumps(swagger_summary, indent=2)}\n```
    -   **Detailed Database Schema:** ```json\n{json.dumps(db_schema_summary, indent=2)}\n```
    -   **Sampled Database Data Summary (column names and data availability):** ```json\n{json.dumps(sampled_data_summary_concise, indent=2)}\n```
    -   **Thread Group Users:** `{thread_group_users}`
    -   **Ramp-up Time (seconds):** `{ramp_up_time}`
    -   **Loop Count:** `{loop_count}` (Note: -1 means infinite)
    -   **Include Assertions in Scenario:** `{include_assertions_flag}` (If true, add response_code 200 assertion to each sampler)
    -   **Authentication Flow Enabled:** `{enable_auth_flow_flag}` (If true, and not a login request, add Authorization header with Bearer ${{auth_token}}). Assume `auth_token` is extracted from a prior login request.

    **CRITICAL REQUIREMENTS FOR EACH `http_sampler`:**
    * **Headers**: ALWAYS include a `headers` array.
        * For any sampler with a `body` (e.g., POST, PUT, PATCH requests), it MUST include `{{\"name\": \"Content-Type\", \"value\": \"application/json\"}}`.
        * For all samplers *except* the designated login request, if `Authentication Flow Enabled` is `true`, it MUST include `{{\"name\": \"Authorization\", \"value\": \"Bearer ${{auth_token}}\"}}`.
    * **Assertions**: ALWAYS include an `assertions` array.
        * It MUST include a `{{\"type\": \"response_code\", \"pattern\": \"XXX\"}}` assertion, where XXX is "200" for GET, PUT, PATCH requests, "201" for POST requests, and "204" for DELETE requests.
        * For GET, POST, PUT, and PATCH requests, it MUST also include a `{{\"type\": \"text_response\", \"pattern\": \"[relevant_key_field]\"}}` assertion. The `[relevant_key_field]` should be a primary identifier or a common success message expected in the response body (e.g., `ApplicationCD`, `clientId`, `authProviderCd`, `roleName`, `schemeName`, `portalCd`, `mappingId`, `status`, `updated`).

    """
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": final_llm_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "test_plan": {
                        "type": "OBJECT",
                        "properties": {
                            "thread_group": {
                                "type": "OBJECT",
                                "properties": {
                                    "num_threads": {"type": "INTEGER"},
                                    "ramp_up": {"type": "INTEGER"}
                                },
                                "required": ["num_threads", "ramp_up"]
                            },
                            "csv_data_set_config": {
                                "type": "OBJECT",
                                "properties": {
                                    "filename": {"type": "STRING"},
                                    "variable_names": {"type": "ARRAY", "items": {"type": "STRING"}}
                                },
                                "required": ["filename", "variable_names"]
                            },
                            "http_samplers": { # Changed to array of http_sampler objects
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "name": {"type": "STRING"},
                                        "method": {"type": "STRING", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                                        "path": {"type": "STRING"}, # LLM should provide template path
                                        "path_params": { # NEW: Explicit path parameters
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "name": {"type": "STRING"},
                                                    "value": {"type": "STRING"}
                                                },
                                                "required": ["name", "value"]
                                            }
                                        },
                                        "query_params": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "name": {"type": "STRING"},
                                                    "value": {"type": "STRING"}
                                                },
                                                "required": ["name", "value"]
                                            }
                                        },
                                        "body": {"type": "STRING"}, 
                                        "content_type": {"type": "STRING"},
                                        "headers": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "name": {"type": "STRING"},
                                                    "value": {"type": "STRING"}
                                                },
                                                "required": ["name", "value"]
                                            }
                                        },
                                        "assertions": { # Assertions moved inside http_sampler
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "type": {"type": "STRING", "enum": ["response_code", "text_response"]},
                                                    "pattern": {"type": "STRING"}
                                                },
                                                "required": ["type", "pattern"]
                                            }
                                        },
                                        "extractions": { # Extractions moved inside http_sampler
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "json_path": {"type": "STRING"},
                                                    "var_name": {"type": "STRING"}
                                                },
                                                "required": ["json_path", "var_name"]
                                            }
                                        }
                                    },
                                    "required": ["name", "method", "path"]
                                }
                            }
                        },
                        "required": ["thread_group", "csv_data_set_config", "http_samplers"] # Updated required field
                    }
                },
                "required": ["test_plan"]
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

                # If LLM generated body as a string, parse it into a Python object
                # This now needs to iterate through each http_sampler in the array
                if 'test_plan' in parsed_json and 'http_samplers' in parsed_json['test_plan'] and \
                   isinstance(parsed_json['test_plan']['http_samplers'], list):
                    
                    for sampler in parsed_json['test_plan']['http_samplers']:
                        if 'body' in sampler and isinstance(sampler['body'], str):
                            try:
                                sampler['body'] = json.loads(sampler['body'])
                                logger.info(f"Successfully parsed HTTP sampler '{sampler.get('name')}' body string to JSON object.")
                            except json.JSONDecodeError as body_parse_error:
                                logger.error(f"Failed to parse HTTP sampler '{sampler.get('name')}' body string from LLM response: {body_parse_error}")
                                pass # Body remains a string if it can't be parsed
                
                return [parsed_json] # Wrap the single object in a list
            except json.JSONDecodeError as e:
                # Attempt to repair the JSON string if parsing fails
                st.warning(f"Attempting to repair truncated JSON response. Original error: {e}")
                repaired_json_str = _repair_truncated_json(json_response_str)
                logger.info(f"Attempted to repair JSON string to: {repaired_json_str}")
                try:
                    parsed_json = json.loads(repaired_json_str)
                    st.success("Successfully repaired and parsed LLM JSON response.")

                    # If LLM generated body as a string after repair, parse it into a Python object
                    # This now needs to iterate through each http_sampler in the array
                    if 'test_plan' in parsed_json and 'http_samplers' in parsed_json['test_plan'] and \
                       isinstance(parsed_json['test_plan']['http_samplers'], list):
                        for sampler in parsed_json['test_plan']['http_samplers']:
                            if 'body' in sampler and isinstance(sampler['body'], str):
                                try:
                                    sampler['body'] = json.loads(sampler['body'])
                                    logger.info(f"Successfully parsed HTTP sampler '{sampler.get('name')}' body string to JSON object after repair.")
                                except json.JSONDecodeError as body_parse_error:
                                    logger.error(f"Failed to parse HTTP sampler '{sampler.get('name')}' body string from LLM response after repair: {body_parse_error}")
                                    pass # Body remains a string if it can't be parsed

                    return [parsed_json] # Wrap the single object in a list
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

    # Default LLM prompt for initial display
    # UPDATED: Adjusted the prompt to reflect the new expected JSON output structure
    # and added explicit instructions for headers, assertions, and CSV variables.
    # Emphasize that the path should be the OpenAPI template path.
    default_llm_prompt = """
    You are an INTELLIGENT AUTOMATION ENGINEER with expert-level experience in building dynamic, data-driven JMeter test plans using OpenAPI (Swagger), database schemas, and sample data inputs.

    Your task is to GENERATE a fully functional, dynamic JMeter test configuration in a **single JSON object** that describes a complete `test_plan`. This output will be used by another component to construct the full JMeter .jmx test plan.

    **Crucial Instructions:**
    * **Single Test Plan Output:** Your response MUST be a single JSON object with a top-level key `test_plan`. This `test_plan` object must contain `thread_group`, `csv_data_set_config`, and `http_samplers` (note: `http_samplers` is now an array).
    * **Strict Data Source Adherence:** You MUST ONLY use the provided `Swagger Endpoints Summary`, `Detailed Database Schema`, and `Sampled Database Data Summary` as your definitive sources of truth for all API definitions, database structures, and sample values. Do NOT invent data or refer to external knowledge.
    * **Endpoint Filtering & Multiple Samplers:** Your scenario MUST include ALL endpoints explicitly requested by the user in the `prompt`. If the `prompt` specifies multiple endpoints (e.g., "GET /api/v1/application/minthreadcount" and "GET /v1/detailsbyappcds"), you MUST generate a separate `http_sampler` object for EACH of these in the `http_samplers` array.
    * **Data Sourcing for `http_sampler`:**
        * **`path`**: For the `path` field, you MUST use the exact **OpenAPI template path** (e.g., `/pet/{petId}`, NOT `/pet/123` or any concrete value). This is crucial for correct mapping to Swagger definitions.
        * **`path_params`**: If the endpoint has path parameters (e.g., `{petId}` in the `path`), you MUST include them in this array with their `name` and `value`. If a value comes from CSV, use the JMeter variable format, e.g., `"${csv_users_appcd}"` or `"${csv_orders_order_id}"`. Otherwise, use a static value or a dummy value.
        * **`query_params`**: If parameters are in `query`, list them here with their `name` and `value`. If a value comes from CSV, use the JMeter variable format, e.g., `"${csv_users_username}"`. Otherwise, use a static value or a dummy value.
        * **`body`**: If the request has a JSON body, you MUST provide it as a **JSON string**. For example, if the body is `{"key": "value"}` then output `"{\"key\": \"value\"}"`. Values within this string can be static, or JMeter variables (e.g., `"${clientId}"`) if sourced from CSV.
        * **`content_type`**: String like "application/json". Use `null` if no body.
        * **`extractions` (Per Sampler)**: If dynamic data extraction is needed (e.g., extracting a token after login or an ID from a creation response), include `extractions` *within the relevant `http_sampler` object*. The `json_path` should be accurate, and `var_name` should be a descriptive JMeter variable name (e.g., `auth_token`, `pet_id`).
    * **CRITICAL REQUIREMENTS FOR EACH `http_sampler`:**
        * **Headers**: ALWAYS include a `headers` array.
            * For any sampler with a `body` (e.g., POST, PUT, PATCH requests), it MUST include `{"name": "Content-Type", "value": "application/json"}`.
            * For all samplers *except* the designated login request, if `Authentication Flow Enabled` is `true`, it MUST include `{"name": "Authorization", "value": "Bearer ${auth_token}"}`.
        * **Assertions**: ALWAYS include an `assertions` array.
            * It MUST include a `{"type": "response_code", "pattern": "XXX"}` assertion, where XXX is "200" for GET, PUT, PATCH requests, "201" for POST requests, and "204" for DELETE requests.
            * For GET, POST, PUT, and PATCH requests, it MUST also include a `{"type": "text_response", "pattern": "[relevant_key_field]"}` assertion. The `[relevant_key_field]` should be a primary identifier or a common success message expected in the response body (e.g., `ApplicationCD`, `clientId`, `authProviderCd`, `roleName`, `schemeName`, `portalCd`, `mappingId`, `status`, `updated`).
    * **Authentication Flow (`http_samplers` order):**
        * If `Authentication Flow Enabled` is `true`, your `http_samplers` array SHOULD start with a login request (e.g., `POST /user/login`) that includes an `extractions` definition to capture the `auth_token`. Subsequent authenticated requests MUST then include the `Authorization` header using this extracted token (`Bearer ${auth_token}`).
    * **CSV Data Set Configuration (`csv_data_set_config`):**
        * The `variable_names` array in `csv_data_set_config` MUST contain the exact names of ALL JMeter variables (e.g., `client_appcd`, `client_clientid`, `master_portalcd`, `master_authprovidercd`, `master_roleid`, `master_schemename`) that are used in your `http_samplers` and are sourced from CSV (e.g., `${csv_client_appcd}`). These names should directly correspond to the `table_name_column_name` format for clarity. The `filename` can be `data.csv`.

    **Generate JMeter test plan for ALL the following endpoints:**

    **Client**
    GET /v1/detailsbyappcds
    GET /api/v1/client
    POST /api/v1/client
    POST /v1/detailsbyappcd
    GET /v1/secrets/{appcd}
    GET /api/v1/client/{appcd}/secret

    **Master**
    GET /api/v1/authprovider/all
    GET /api/v1/role/all
    GET /api/v1/authprovider/{authProviderCd}/portal
    GET /api/v1/portal/{portalCd}/authprovider
    POST /api/v1/portal/{portalCd}/authprovider
    GET /api/v1/portal/{portalCd}/role
    POST /api/v1/portal/{portalCd}/role
    GET /api/v1/authenticationscheme/all
    GET /api/v1/authenticationscheme/{portalCd}
    POST /api/v1/authenticationscheme/{portalCd}
    GET /api/v1/portalroles/all
    GET /api/v1/mapping/portals

    **Output Expectations:**
    Your response MUST be ONLY the JSON object described below, with no additional text, markdown, or conversational elements. **Ensure the JSON is always perfectly formed and complete, with all brackets and commas correctly placed and no trailing commas or incomplete structures. DO NOT include comments in the JSON output.**

    ```json
    {
      "test_plan": {
        "thread_group": {
          "num_threads": 1,
          "ramp_up": 1
        },
        "csv_data_set_config": {
          "filename": "data.csv",
          "variable_names": ["client_appcd", "client_clientid", "master_portalcd", "master_authprovidercd", "master_roleid", "master_schemename", "users_username", "users_password"]
        },
        "http_samplers": [
          {
            "name": "Login_User",
            "method": "GET",
            "path": "/user/login",
            "path_params": [],
            "query_params": [
              {"name": "username", "value": "${csv_users_username}"},
              {"name": "password", "value": "${csv_users_password}"}
            ],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"}
            ],
            "extractions": [
              {"json_path": "$.token", "var_name": "auth_token"}
            ]
          },
          {
            "name": "Get_Client_Details_By_Appcds",
            "method": "GET",
            "path": "/v1/detailsbyappcds",
            "path_params": [],
            "query_params": [
              {"name": "appCD", "value": "${csv_client_appcd}"},
              {"name": "clientId", "value": "${csv_client_clientid}"}
            ],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "ApplicationCD"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Client_api_v1_client",
            "method": "GET",
            "path": "/api/v1/client",
            "path_params": [],
            "query_params": [
              {"name": "appCD", "value": "${csv_client_appcd}"},
              {"name": "clientId", "value": "${csv_client_clientid}"}
            ],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "appCD"}
            ],
            "extractions": []
          },
          {
            "name": "Add_Client_api_v1_client",
            "method": "POST",
            "path": "/api/v1/client",
            "path_params": [],
            "query_params": [],
            "body": "{\"ApplicationCD\": \"${csv_client_appcd}\", \"LegacyClientCD\": \"${csv_client_clientid}\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "201"},
              {"type": "text_response", "pattern": "clientId"}
            ],
            "extractions": []
          },
          {
            "name": "Add_Client_v1_detailsbyappcd",
            "method": "POST",
            "path": "/v1/detailsbyappcd",
            "path_params": [],
            "query_params": [],
            "body": "{\"ApplicationCD\": \"${csv_client_appcd}\", \"LegacyClientCD\": \"${csv_client_clientid}\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "201"},
              {"type": "text_response", "pattern": "ApplicationCD"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Test_Client_Secret_v1_secrets_appcd",
            "method": "GET",
            "path": "/v1/secrets/{appcd}",
            "path_params": [
              {"name": "appcd", "value": "${csv_client_appcd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Test_Client_Secret_api_v1_client_appcd_secret",
            "method": "GET",
            "path": "/api/v1/client/{appcd}/secret",
            "path_params": [
              {"name": "appcd", "value": "${csv_client_appcd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"}
            ],
            "extractions": []
          },
          {
            "name": "Get_All_Authentication_Provider",
            "method": "GET",
            "path": "/api/v1/authprovider/all",
            "path_params": [],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "authProviderCd"}
            ],
            "extractions": []
          },
          {
            "name": "Get_All_Role",
            "method": "GET",
            "path": "/api/v1/role/all",
            "path_params": [],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "roleName"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Authentication_Provider_Portal",
            "method": "GET",
            "path": "/api/v1/authprovider/{authProviderCd}/portal",
            "path_params": [
              {"name": "authProviderCd", "value": "${csv_master_authprovidercd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "portalCd"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Portal_Authentication_Provider",
            "method": "GET",
            "path": "/api/v1/portal/{portalCd}/authprovider",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "authProviderCd"}
            ],
            "extractions": []
          },
          {
            "name": "Add_Portal_Authentication_Provider",
            "method": "POST",
            "path": "/api/v1/portal/{portalCd}/authprovider",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": "{\"authProviderCd\": \"${csv_master_authprovidercd}\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "201"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Portal_Role",
            "method": "GET",
            "path": "/api/v1/portal/{portalCd}/role",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "roleId"}
            ],
            "extractions": []
          },
          {
            "name": "Add_Portal_Role",
            "method": "POST",
            "path": "/api/v1/portal/{portalCd}/role",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": "{\"roleId\": \"${csv_master_roleid}\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "201"}
            ],
            "extractions": []
          },
          {
            "name": "Get_All_Authentication_Scheme",
            "method": "GET",
            "path": "/api/v1/authenticationscheme/all",
            "path_params": [],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "schemeName"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Portal_Authentication_Scheme",
            "method": "GET",
            "path": "/api/v1/authenticationscheme/{portalCd}",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "schemeId"}
            ],
            "extractions": []
          },
          {
            "name": "Add_Portal_Authentication_Scheme",
            "method": "POST",
            "path": "/api/v1/authenticationscheme/{portalCd}",
            "path_params": [
              {"name": "portalCd", "value": "${csv_master_portalcd}"}
            ],
            "query_params": [],
            "body": "{\"schemeName\": \"${csv_master_schemename}\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "201"}
            ],
            "extractions": []
          },
          {
            "name": "Get_Portal_Roles_All",
            "method": "GET",
            "path": "/api/v1/portalroles/all",
            "path_params": [],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "portalRole"}
            ],
            "extractions": []
          },
          {
            "name": "Get_All_PortalAuthMapping",
            "method": "GET",
            "path": "/api/v1/mapping/portals",
            "path_params": [],
            "query_params": [],
            "body": null,
            "content_type": null,
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "mappingId"}
            ],
            "extractions": []
          }
        ]
      }
    }
    ```
    For each `path_params`, `query_params`, `body`, `headers`, `assertions`, and `extractions` item, **adhere to these strict rules**:
    -   **`path_params`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable like `${variable_name}`).
    -   **`query_params`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable like `${variable_name}`).
    -   **`body`**: A JSON string (e.g., `"{\"key\": \"value\"}"`). Values can be JMeter variables. Use `null` if no body.
    -   **`content_type`**: String like "application/json". Use `null` if no body.
    -   **`headers`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable).
    -   **`assertions`**: Array of objects. Each object must have `type` (string, "response_code" or "text_response") and `pattern` (string). This is now inside `http_sampler`.
    -   **`extractions`**: Array of objects. Each object must have `json_path` (string, JSONPath expression) and `var_name` (string, name of the JMeter variable to store the extracted value). This is now inside `http_sampler`.
    """

    if 'llm_prompt_text' not in st.session_state:
        st.session_state.llm_prompt_text = default_llm_prompt

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
        logger.debug(f"Entering _build_recursive_json_body_with_llm_guidance with: {'.'.join(current_path_segments)}, schema_type: {schema.get('type')}")

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
                        # PRIORITIZE table_name and column_name for from_csv, ignore 'value' from LLM if present
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
                    # PRIORITIZE table_name and column_name for from_csv, ignore 'value' from LLM if present
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
                        if parser.load_swagger_spec(): # Load the spec first
                            endpoints = parser.parse() # Corrected method call here
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
                        else:
                            st.error("Failed to load Swagger spec. Check URL and console for errors.")
                            st.session_state.swagger_endpoints = []
                            st.session_state.selected_endpoint_keys = []
                            st.session_state.scenario_requests_configs = []
                            st.session_state.jmx_content_download = None
                            st.session_state.csv_content_download = None
                            st.session_state.mapping_metadata_download = None
                            st.session_state.full_swagger_spec_download = None
                            st.session_state.mappings = {}
                            st.session_state.llm_structured_scenario = None
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
                    st.session_state.swagger_parser.swagger_data, indent=2 # Use swagger_data directly
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

        # Move the prompt text area here
        st.subheader("LLM Prompt Configuration")
        st.session_state.llm_prompt_text = st.text_area(
            "Edit the detailed prompt for the LLM",
            value="Please create jmeter script for end point mentioned below",
            height=100, # Increased height to show more of the detailed prompt
            help="Customize the instructions and examples provided to the LLM for scenario generation. This text is sent directly to the LLM."
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
                        llm_prompt_input=st.session_state.llm_prompt_text, # Pass the editable prompt from UI
                        swagger_endpoints=st.session_state.swagger_endpoints, # Will be filtered inside the function
                        db_tables_schema=st.session_state.db_tables_schema,
                        db_sampled_data=st.session_state.db_sampled_data,
                        thread_group_users=st.session_state.num_users_input,
                        ramp_up_time=st.session_state.ramp_up_time_input,
                        loop_count=st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1,
                        include_assertions_flag=st.session_state.include_scenario_assertions, # Pass assertions flag
                        enable_auth_flow_flag=st.session_state.enable_auth_flow, # Pass auth flow flag
                        api_key=st.session_state.gemini_api_key
                    )
                    
                    if structured_scenario:
                        st.session_state.llm_structured_scenario = structured_scenario
                        st.success("AI script design generated!")
                        st.subheader("AI-Generated JMeter Scenario Details (Streaming)")
                        
                        # Display requests in a streaming fashion
                        # UPDATED: Adjust display logic for new LLM output structure
                        if structured_scenario and len(structured_scenario) > 0 and 'test_plan' in structured_scenario[0]:
                            test_plan_data = structured_scenario[0]['test_plan']
                            
                            st.markdown("**Test Plan Overview:**")
                            st.json(test_plan_data) # Display the full structured response
                            
                            # You might want to extract and display specific parts more clearly
                            st.markdown("---")
                            st.markdown(f"**Thread Group:**")
                            st.write(f"Number of Threads: `{test_plan_data['thread_group']['num_threads']}`")
                            st.write(f"Ramp-up Time: `{test_plan_data['thread_group']['ramp_up']}`")

                            if 'csv_data_set_config' in test_plan_data:
                                st.markdown("---")
                                st.markdown(f"**CSV Data Set Config:**")
                                st.write(f"Filename: `{test_plan_data['csv_data_set_config']['filename']}`")
                                st.write(f"Variable Names: `{', '.join(test_plan_data['csv_data_set_config']['variable_names'])}`")
                            
                            st.markdown("---")
                            st.markdown(f"**HTTP Samplers:**")
                            if 'http_samplers' in test_plan_data and isinstance(test_plan_data['http_samplers'], list):
                                for i, sampler in enumerate(test_plan_data['http_samplers']):
                                    st.markdown(f"---")
                                    st.markdown(f"**Sampler {i+1}: {sampler.get('name', 'Unnamed Sampler')}**")
                                    st.write(f"Method: `{sampler['method']}`")
                                    st.write(f"Path: `{sampler['path']}`")
                                    # Display path_params if present
                                    if 'path_params' in sampler and sampler['path_params']:
                                        st.markdown("Path Parameters:")
                                        for pp in sampler['path_params']:
                                            st.write(f"- `{pp['name']}`: `{pp['value']}`")
                                    if 'query_params' in sampler and sampler['query_params']:
                                        st.markdown("Query Parameters:")
                                        for qp in sampler['query_params']:
                                            st.write(f"- `{qp['name']}`: `{qp['value']}`")
                                    if 'body' in sampler and sampler['body']:
                                        st.markdown("Body:")
                                        # Display the body as code, ensure it's a JSON object if parsed
                                        if isinstance(sampler['body'], (dict, list)):
                                            st.code(json.dumps(sampler['body'], indent=2))
                                        else: # It's a string (from LLM) or other primitive
                                            st.code(str(sampler['body']))
                                    if 'content_type' in sampler and sampler['content_type']:
                                        st.write(f"Content Type: `{sampler['content_type']}`")
                                    if 'headers' in sampler and sampler['headers']:
                                        st.markdown("Headers:")
                                        for header in sampler['headers']:
                                            st.write(f"- `{header['name']}`: `{header['value']}`")

                                    if 'assertions' in sampler and sampler['assertions']:
                                        st.markdown(f"**Assertions for this Sampler:**")
                                        for assertion in sampler['assertions']:
                                            st.write(f"- Type: `{assertion['type']}`, Pattern: `{assertion['pattern']}`")

                                    if 'extractions' in sampler and sampler['extractions']:
                                        st.markdown(f"**Extractions for this Sampler:**")
                                        for extraction in sampler['extractions']:
                                            st.write(f"- JSONPath: `{extraction['json_path']}` -> Var: `{extraction['var_name']}`")
                            else:
                                st.info("No HTTP Samplers found in AI-generated design.")
                            
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


            full_swagger_dict = st.session_state.swagger_parser.swagger_data # Use swagger_data directly
            base_path_from_swagger = full_swagger_dict.get('basePath', '/')
            if not base_path_from_swagger.startswith('/'):
                base_path_from_swagger = '/' + base_path_from_swagger
            if base_path_from_swagger != '/' and base_path_from_swagger.endswith('/'):
                base_path_from_swagger = base_path_from_swagger.rstrip('/')

            for ep_key in st.session_state.selected_endpoint_keys:
                method_str, path_str = ep_key.split(' ', 1)
                method_key = method_str.lower()

                # Find the actual SwaggerEndpoint object
                resolved_endpoint_obj = next((ep for ep in st.session_state.swagger_endpoints if ep.method == method_str and ep.path == path_str), None)

                if resolved_endpoint_obj:
                    # Use data from the resolved SwaggerEndpoint object
                    resolved_endpoint_data = resolved_endpoint_obj.to_dict() # Convert back to dict for consistency with old logic
                    
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
                            first_content_type = next(iter(content_types.values()), None) # Get content type from requestBody
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

            full_swagger_dict = st.session_state.swagger_parser.swagger_data # Use swagger_data directly
            base_path_from_swagger = full_swagger_dict.get('basePath', '/')
            if not base_path_from_swagger.startswith('/'):
                base_path_from_swagger = '/' + base_path_from_swagger
            if base_path_from_swagger != '/' and base_path_from_swagger.endswith('/'):
                base_path_from_swagger = base_path_from_swagger.rstrip('/')

            # UPDATED: The loop now processes a single test_plan object from the LLM response.
            # structured_scenario is now a list containing one item (the test_plan object)
            if st.session_state.llm_structured_scenario and len(st.session_state.llm_structured_scenario) > 0 and 'test_plan' in st.session_state.llm_structured_scenario[0]:
                llm_response_test_plan = st.session_state.llm_structured_scenario[0]['test_plan']
                
                # Now http_samplers is a list
                http_samplers_list = llm_response_test_plan.get('http_samplers', [])

                for http_sampler in http_samplers_list:
                    method_str = http_sampler.get('method')
                    # Use the path directly from LLM, it should be the template path as per new prompt
                    path_from_llm = http_sampler.get('path') 
                    request_name = http_sampler.get('name', f"{method_str}_{path_from_llm.replace('/','_').strip('_')}")
                    think_time = 0 

                    # UPDATED: Use the new matching function to find the Swagger endpoint
                    # The LLM is now instructed to provide the template path, so extracted_path_params
                    # will be empty if LLM follows instructions.
                    # This function still helps if LLM deviates and gives a concrete path.
                    resolved_endpoint_obj, _ = _match_swagger_path_with_generated_path(
                        st.session_state.swagger_endpoints, method_str, path_from_llm
                    )

                    if not resolved_endpoint_obj:
                        st.warning(f"AI suggested endpoint {method_str} {path_from_llm} not found in Swagger spec. Skipping.")
                        continue # Skip this sampler if endpoint not found
                    else:
                        resolved_endpoint_data = resolved_endpoint_obj.to_dict()

                        # Use the template path from resolved_endpoint_obj for JMeter generation
                        jmeter_formatted_path = resolved_endpoint_obj.path 
                        request_config = {
                            "endpoint_key": f"{method_str} {resolved_endpoint_obj.path}", # Use template path for key
                            "name": request_name,
                            "method": method_str,
                            "path": "", 
                            "parameters": {}, 
                            "headers": {}, # Initialize headers here
                            "body": None,
                            "assertions": [], # Initialize assertions here
                            "json_extractors": [],
                            "think_time": think_time # Use 0 or derive from thread_group later
                        }

                        # Handle path parameters from LLM response and substitute them into the path
                        for pp in http_sampler.get('path_params', []):
                            param_name = pp['name']
                            param_value = pp['value']
                            # Replace {param_name} in jmeter_formatted_path with the value provided by LLM
                            jmeter_formatted_path = jmeter_formatted_path.replace(f"{{{param_name}}}", param_value)


                        # Handle query parameters from LLM response
                        for qp in http_sampler.get('query_params', []):
                            request_config['parameters'][qp['name']] = qp['value']

                        # Handle headers from LLM response
                        # Directly assign headers from LLM output
                        for header in http_sampler.get('headers', []):
                            request_config['headers'][header['name']] = header['value']

                        # Add authentication header if enabled and not already provided by LLM for this request
                        # This covers cases where LLM might forget, or if it's a non-login authenticated request
                        if st.session_state.enable_auth_flow and request_config["endpoint_key"] != f"{st.session_state.auth_login_method} {st.session_state.auth_login_endpoint_path}":
                            auth_header_name_lower = st.session_state.auth_header_name.lower()
                            # Check if Authorization header is already present (case-insensitive)
                            if not any(h_name.lower() == auth_header_name_lower for h_name in request_config['headers'].keys()):
                                if "auth_token" in extracted_variables_map: # Check if the token was successfully extracted
                                    request_config["headers"][st.session_state.auth_header_name] = f"{st.session_state.auth_header_prefix}{extracted_variables_map['auth_token']}"
                                else:
                                    # Fallback for auth token, perhaps a dummy or warning
                                    request_config["headers"][st.session_state.auth_header_name] = f"{st.session_state.auth_header_prefix}<<AUTH_TOKEN_MISSING>>"
                                    st.warning(f"Authentication flow enabled for {request_config['endpoint_key']}, but auth token not found or extracted. Using placeholder.")


                        # Handle body from LLM response (now it's a string, needs parsing)
                        if 'body' in http_sampler and http_sampler['body'] is not None:
                            if isinstance(http_sampler['body'], str):
                                try:
                                    # Attempt to parse the body string to JSON object
                                    request_config["body"] = json.loads(http_sampler['body'])
                                except json.JSONDecodeError:
                                    # If it's not a valid JSON string, keep it as is (might be plain text)
                                    request_config["body"] = http_sampler['body']
                            else:
                                # If LLM didn't give a string (e.g. if schema was Object before), use as is
                                request_config["body"] = http_sampler['body']
                            
                            # Ensure content type is set if body is present
                            if 'content_type' in http_sampler and http_sampler['content_type']:
                                request_config["headers"]["Content-Type"] = http_sampler['content_type']
                            elif 'Content-Type' not in request_config['headers']: # Add default if not already set by LLM
                                request_config["headers"]["Content-Type"] = "application/json" # Default for JSON body


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

                        # Handle assertions from LLM response (now specific to this sampler)
                        for assertion_data in http_sampler.get('assertions', []):
                            if assertion_data['type'] == 'response_code':
                                request_config['assertions'].append({"type": "Response Code", "value": assertion_data['pattern']})
                            elif assertion_data['type'] == 'text_response':
                                request_config['assertions'].append({"type": "Response Body Contains", "value": assertion_data['pattern']})
                            else:
                                request_config['assertions'].append(assertion_data) # Add as is if new type

                        # Handle extractions from LLM response (now specific to this sampler)
                        for extractor_data in http_sampler.get('extractions', []):
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
            # UPDATED: Display logic for LLM-generated single test plan.
            # Now, st.session_state.scenario_requests_configs holds a list of requests (could be from LLM or manual).
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
            # The scenario_plan for JMeterGenerator needs to be a list of individual requests.
            # st.session_state.scenario_requests_configs already holds this list,
            # whether it came from LLM processing or manual refinement.
            scenario_plan = {"requests": st.session_state.scenario_requests_configs}
            
            num_users = st.session_state.num_users_input
            ramp_up_time = st.session_state.ramp_up_time_input
            loop_count = st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1
            
            global_constant_timer_delay = st.session_state.constant_timer_delay_ms if st.session_state.enable_constant_timer else 0

            csv_data_for_jmeter = {}
            csv_headers = set()

            # UPDATED: How CSV data is identified now needs to come from the `csv_data_set_config` from the LLM.
            # The existing `st.session_state.mappings` logic still applies for manual refinement,
            # but for LLM-generated output, we prefer the LLM's `csv_data_set_config`.
            if st.session_state.llm_structured_scenario and len(st.session_state.llm_structured_scenario) > 0 and 'test_plan' in st.session_state.llm_structured_scenario[0]:
                llm_csv_config = st.session_state.llm_structured_scenario[0]['test_plan'].get('csv_data_set_config')
                if llm_csv_config and llm_csv_config.get('variable_names'):
                    # Assume LLM variable names are in the format table_column for now
                    # This might need refinement based on how LLM actually names them.
                    for var_name_from_llm_csv_config in llm_csv_config['variable_names']:
                        # The user examples use 'clientId', 'appCD' directly as variable names.
                        # We need to map these back to actual database table.column if possible.
                        # This part of the integration is critical and may need further LLM refinement or manual mapping.
                        # For now, let's try to infer if they correspond to sampled data.
                        
                        found_match = False
                        # Try to find a match in sampled data using flexible matching
                        for table_name, df in st.session_state.db_sampled_data.items():
                            for col_name in df.columns:
                                # Prioritize exact match or case-insensitive match
                                if var_name_from_llm_csv_config.lower() == f"{table_name.lower()}_{col_name.lower()}": # Match table_column format
                                    jmeter_var_name_raw = f"csv_{table_name}_{col_name}"
                                    if jmeter_var_name_raw not in csv_data_for_jmeter:
                                        csv_data_for_jmeter[jmeter_var_name_raw] = df[col_name].tolist()
                                        csv_headers.add(jmeter_var_name_raw)
                                    found_match = True
                                    break # Found a column for this var_name
                            if found_match:
                                break # Found a table for this var_name
                        
                        if not found_match:
                            logger.warning(f"LLM suggested CSV variable '{var_name_from_llm_csv_config}' in csv_data_set_config but no direct or inferred match found in sampled database columns for CSV generation. This variable might not be correctly populated in CSV.")

            else: # Fallback to existing logic if LLM didn't provide new structure
                for endpoint_key, params_map in st.session_state.mappings.items():
                    for param_name, mapping_info in params_map.items():
                        if mapping_info['source'] == "DB Sample (CSV)":
                            if 'table_name' in mapping_info and 'column_name' in mapping_info:
                                jmeter_var_name_raw = f"csv_{mapping_info['table_name']}_{mapping_info['column_name']}"
                            else:
                                logger.warning(f"Mapping info for {param_name} (from DB Sample) missing table_name/column_name. Cannot generate CSV.")
                                continue

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
                current_full_swagger_spec_dict = st.session_state.swagger_parser.swagger_data # Use swagger_data directly
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
