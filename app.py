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

from utils.jmeter_generator import JMeterScriptGenerator # Import the class
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
    Attempts to repair a truncated or malformed JSON string by iteratively trimming
    from the end and adding common missing closing characters.
    Also attempts to escape unescaped internal quotes and newlines within string contexts,
    and handles specific cases of problematic backslashes.
    """
    stripped_string = json_string.strip() # Start with stripping whitespace

    # --- Initial Cleanup: Aggressively extract the main JSON block ---
    # Find the first occurrence of '{' or '[' and the last '}' or ']'
    json_start_match = re.search(r'[\{\[]', stripped_string)
    json_end_match = re.search(r'[\}\]]', stripped_string[::-1]) # Search reversed string

    if json_start_match and json_end_match:
        start_index = json_start_match.start()
        end_index = len(stripped_string) - json_end_match.start()
        stripped_string = stripped_string[start_index:end_index]
        logger.debug(f"Trimmed to potential JSON block: {stripped_string[:200]}...")

    # Strip markdown fences again, in case they were inside the extracted block
    if stripped_string.startswith("```json"):
        stripped_string = stripped_string[7:]
    if stripped_string.endswith("```"):
        stripped_string = stripped_string[:-3]
    stripped_string = stripped_string.strip() # Re-strip after markdown removal


    # --- Step 1: Pre-process for common JSON string literal issues ---
    # This loop attempts to fix unescaped newlines, unescaped quotes, and
    # problematic backslashes within what *should be* JSON string values.
    processed_chars = []
    in_string = False
    i = 0
    while i < len(stripped_string):
        char = stripped_string[i]

        if char == '\\':
            # Handle potential escape sequences. If it's a valid JSON escape, append as is.
            # Otherwise, escape the backslash itself.
            if i + 1 < len(stripped_string):
                next_char = stripped_string[i+1]
                # Common JSON valid escape characters
                if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    processed_chars.append(char)
                    processed_chars.append(next_char)
                    i += 1 # Skip next char
                else:
                    # Not a standard JSON escape, escape the backslash itself if in a string
                    if in_string:
                        processed_chars.append('\\\\')
                    else:
                        processed_chars.append(char) # Outside string, keep as is for now
            else:
                # Dangling backslash at the very end, escape it
                if in_string:
                    processed_chars.append('\\\\')
                else:
                    processed_chars.append(char)
        elif char == '"':
            # Determine if this quote is a string delimiter or an unescaped internal quote.
            # An unescaped quote within a string needs to be escaped.
            # A quote following an odd number of backslashes is already escaped.
            j = i - 1
            backslashes_count = 0
            while j >= 0 and stripped_string[j] == '\\':
                backslashes_count += 1
                j -= 1
            
            if backslashes_count % 2 == 0: # This quote is NOT escaped
                if in_string: # We are inside a string, so this is an unescaped internal quote
                    processed_chars.append('\\"') # Escape it
                else: # We are outside a string, so this is a string delimiter
                    processed_chars.append(char)
                    in_string = not in_string # Toggle state
            else: # This quote IS escaped by preceding backslashes
                processed_chars.append(char) # Keep it as is (already escaped in source)
        elif in_string:
            # Inside a string, escape control characters and other problematic ones
            if char == '\n':
                processed_chars.append('\\n')
            elif char == '\r':
                processed_chars.append('\\r')
            elif char == '\t':
                processed_chars.append('\\t')
            elif char == '`': # Backticks are not standard JSON, escape them
                processed_chars.append('\\`')
            elif ord(char) < 32: # Escape other non-printable ASCII control characters
                processed_chars.append(f'\\u{ord(char):04x}')
            else:
                processed_chars.append(char)
        else:
            # Not in a string, not a backslash or quote, append as is
            processed_chars.append(char)
        i += 1
    
    repaired_output = "".join(processed_chars)
    logger.debug(f"After string literal pre-processing: {repaired_output}")

    # If after processing, we are still conceptually "in a string", close it.
    # This handles cases where the entire JSON string might have been truncated mid-string.
    if in_string and not repaired_output.endswith('"'): # Check if the last char is not a quote
        repaired_output += '"'
        logger.debug(f"Appended missing closing quote: {repaired_output}")

    # --- Step 2: Attempt to parse. If successful, return. ---
    try:
        json.loads(repaired_output)
        logger.debug("Successfully parsed after string literal repair.")
        return repaired_output
    except json.JSONDecodeError as e:
        logger.debug(f"Still failing after string literal repair: {e}. Proceeding with structural repair.")
        pass # Continue with structural repair

    # --- Step 3: Structural Repair (balancing brackets and braces) and Final Trimming ---
    max_trim_attempts = 100 # Safety limit to prevent infinite loops
    for attempt in range(max_trim_attempts):
        try:
            json.loads(repaired_output)
            logger.debug(f"JSON successfully parsed after {attempt} final trims.")
            return repaired_output
        except json.JSONDecodeError as e:
            logger.debug(f"JSONDecodeError during final trim attempt {attempt}: {e}. Current length: {len(repaired_output)}")
            
            # Specific handling for "Expecting ',' delimiter" and other syntax errors
            if e.pos is not None and e.pos < len(repaired_output):
                # Aggressively truncate the string just before the problematic character
                # This is a heuristic to cut off the source of the delimiter/syntax error
                repaired_output = repaired_output[:e.pos]
                logger.warning(f"Truncated string at error position {e.pos} due2 to JSON syntax issue. New length: {len(repaired_output)}")
                
                # After truncation, re-balance structural elements for the new shorter string
                temp_stack = []
                for char_val in repaired_output:
                    if char_val == '{': temp_stack.append('{')
                    elif char_val == '[': temp_stack.append('[')
                    # Only pop if matching opener is found, otherwise it's an extraneous closer in the truncated part
                    elif char_val == '}':
                        if temp_stack and temp_stack[-1] == '{': temp_stack.pop()
                    elif char_val == ']':
                        if temp_stack and temp_stack[-1] == '[': temp_stack.pop()
                
                # Append missing closers
                for opener in reversed(temp_stack):
                    if opener == '{': repaired_output += '}'
                    elif opener == '[': repaired_output += ']'
                logger.debug(f"Re-balanced after truncation for syntax error. New end: {repaired_output[-50:] if len(repaired_output) > 50 else repaired_output}")
                
                # Try parsing again immediately with the modified string
                continue 

            # Fallback general trimming if e.pos is not useful or after targeted truncation
            original_len = len(repaired_output)
            
            if not repaired_output: # Handle empty string after previous truncations
                 logger.debug("Repaired output is empty, cannot trim further.")
                 break
            
            # Attempt to remove common problematic trailing characters
            if repaired_output.endswith((',', '[', '{', ':', '\\', '"', '`')):
                repaired_output = repaired_output.rstrip(',[:{\\"`')
                logger.debug(f"Trimmed problematic trailing non-structural chars. New end: {repaired_output[-50:] if len(repaired_output) > 50 else repaired_output}")
            elif original_len > 1 and repaired_output[-2] == ',' and repaired_output.endswith(('}', ']')):
                # Handle trailing comma just before a valid closing bracket/brace
                repaired_output = repaired_output[:-2] + repaired_output[-1] # Remove comma, keep closer
                logger.debug(f"Trimmed trailing comma before closer. New end: {repaired_output[-50:] if len(repaired_output) > 50 else repaired_output}")
            elif repaired_output.endswith((' ', '\n', '\r', '\t')):
                repaired_output = repaired_output.rstrip(' \n\r\t')
                logger.debug(f"Trimmed trailing whitespace. New end: {repaired_output[-50:] if len(repaired_output) > 50 else repaired_output}")
            else:
                # Last resort: remove one character if no specific pattern matched and it's still invalid
                if repaired_output: # Ensure string is not empty before slicing
                    repaired_output = repaired_output[:-1]
                logger.debug(f"General one-char trim. New end: {repaired_output[-50:] if len(repaired_output) > 50 else repaired_output}")

            if len(repaired_output) == original_len:
                logger.debug("No further progress in trimming. Breaking.")
                break

    logger.warning("Failed to fully repair JSON after multiple attempts. Returning best effort string.")
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


def call_llm_for_scenario_plan(llm_prompt_input: str,
                               swagger_endpoints: List[SwaggerEndpoint],
                               db_tables_schema: Dict[str, List[Dict[str, Any]]],
                               db_sampled_data: Dict[str, pd.DataFrame],
                               thread_group_users: int,
                               ramp_up_time: int,
                               loop_count: int,
                               include_assertions_flag: bool,
                               enable_auth_flow_flag: bool,
                               api_key: str) -> Optional[Dict[str, Any]]: # This function returns structured JSON
    """
    Calls the LLM API to get a structured JSON scenario plan.
    """
    import json
    import re
    import requests

    # Gemini Flash API URL
    protocol_segment = "https://"
    domain_segment = "generativelanguage.googleapis.com"
    path_segment = "/v1beta/models/gemini-2.0-flash:generateContent?key="
    api_url = f"{protocol_segment}{domain_segment}{path_segment}{api_key}"

    # Step 1: Match endpoints from prompt
    specific_endpoint_patterns = re.findall(r'(?i)(get|post|put|delete|patch)\s*\n?\s*([/\w\-{{}}]+)', llm_prompt_input)
    filtered_swagger_endpoints = []

    for method, path in specific_endpoint_patterns:
        for ep in swagger_endpoints:
            if ep.method.lower() == method.lower() and ep.path.rstrip('/') == path.rstrip('/'):
                filtered_swagger_endpoints.append(ep)

    if not filtered_swagger_endpoints:
        filtered_swagger_endpoints = swagger_endpoints
        st.warning("⚠️ No specific match found in Swagger. Using all available endpoints.")

    # Step 2: Generate DB param mappings
    suggested_mappings = DataMapper.suggest_mappings(
        filtered_swagger_endpoints, db_tables_schema, db_sampled_data
    )

    # Step 3: Trim large context inputs
    swagger_summary = json.dumps([ep.to_dict() for ep in filtered_swagger_endpoints], indent=2)[:15000]
    db_schema_summary = json.dumps(db_tables_schema, indent=2)[:8000]
    sampled_data_summary = json.dumps({
        k: df.head(3).to_dict(orient="records") for k, df in db_sampled_data.items()
    }, indent=2)[:5000]
    suggested_mapping_summary = json.dumps(suggested_mappings, indent=2)[:5000]

    # The prompt for LLM to generate structured JSON scenario
    final_llm_prompt_for_json_output = (
        f"{llm_prompt_input}\n\n" # This will include the user's base prompt
        f"### 🔍 Input Context:\n"
        f"- Swagger Endpoints:\n{swagger_summary}\n\n"
        f"- Database Schema:\n{db_schema_summary}\n\n"
        f"- Sampled DB Data:\n{sampled_data_summary}\n\n"
        f"- Suggested Parameter Mappings:\n{suggested_mapping_summary}\n\n"
        f"""### 📌 Mapping Instructions:
            - Use the provided Swagger parameter names and the DB schema + sample data to intelligently map values.
            - If a parameter like `clientId`, `ApplicationCD`, `authProviderCd`, etc., appears in both the Swagger and DB schema/sample, assume they are linked.
            - Match body or query parameters to the most relevant column names from the DB schema or sample data.
            - Prefer DB column values for:
                - Query parameters
                - Path variables (like `{{appcd}}`)
                - Request bodies
            - If a field maps to a CSV, use a placeholder like `${{csv_Client_ClientId}}`, `${{csv_DeletedUser_ApplicationCD}}`, etc.
            ✅ Follow the suggested_mappings below to assign accurate values:"""
        f"- Thread Group Users: {thread_group_users}\n"
        f"- Ramp-up Time (s): {ramp_up_time}\n"
        f"- Loop Count: {loop_count}\n"
        f"- Assertions Included: {include_assertions_flag}\n"
        f"- Auth Flow Enabled: {enable_auth_flow_flag}\n\n"
        f"""### 🔁 Extraction Instructions (Generic for Any Endpoint):

            - For **every endpoint**, check if the response body contains any identifiable or reusable field such as:
                - IDs (e.g., `clientId`, `mappingId`, `applicationCd`)
                - Tokens (e.g., `auth_token`, `accessToken`)
                - Secrets (e.g., `clientSecret`, `secretValue`)
                - Any other unique identifier or key field (e.g., `portalCd`, `schemeId`, `status`, `trackingId`)

            - If such fields are present in the response JSON, you **must include an `extractions` array** inside that same `http_sampler`.

            - Use this format:
            "extractions": [
              {{
                "type": "json_path",
                "expression": "$.fieldName",
                "variable": "fieldName"
              }}
            ]

            ✅ Best Practices:
            - Do this for **any POST, PUT, PATCH, GET** endpoint where relevant response fields are returned.
            - It is okay if the extracted value is **not used immediately or not reused at all** — still include it.
            - Use the extracted variable in subsequent requests **only when applicable** by referencing `${{fieldName}}`.
            - **Do not duplicate extractors** for the same field in multiple samplers.
            - Always prefer contextual names like:
                - `detailsbyappcd_ApplicationCD`
                - `clientApi_LegacyClientCD`
                - `secrets_api_client_secret`
        """
        f"""### ♻️ Variable Extraction and Reuse Guidelines:
            - If a sampler returns a field like `clientId`, `authToken`, `mappingId`, `applicationCd`, etc., extract it and save it into a variable using `extractions`.
            - Store it using `"extractions"` inside the **same http_sampler** object.
            - Once extracted, use the variable using `${{variable}}` syntax in any **future request**, even if it's not the immediate next one.
            - The variable may be used:
                - Immediately (in the next request)
                - After 1 or more intermediate samplers
                - Or not used at all (still keep the extractor)
                - DO NOT repeat the extraction if the variable is already extracted earlier.
                - Use descriptive variable names (e.g., `clientId`, `auth_token`, `mappingId`) and maintain consistency.
                ✅ Example usage:
                1. Extract token in login response:
                "extractions": [{{"type": "json_path", "expression": "$.token", "variable": "auth_token"}}]"""
        "\n\n"
        "### ✅ Expected Output:\n"
        "Return ONLY a valid JSON object in the following format:\n"
        "{\n"
        "  \"test_plan\": {\n"
        "    \"thread_group\": {\n"
        "      \"num_threads\": 1,\n"
        "      \"ramp_up\": 1\n"
        "    },\n"
        "    \"csv_data_set_config\": {\n"
        "      \"filename\": \"ClientUpdate.csv\",\n"
        "      \"variable_names\": [\"clientId\", \"newName\"]\n"
        "    },\n"
        "    \"http_samplers\": [\n"
        "      {\n"
        "        \"name\": \"PUT /v1/update-client\",\n"
        "        \"method\": \"PUT\",\n"
        "        \"path\": \"/v1/update-client\",\n"
        "        \"body\": \"{\\\"clientId\\\": \\\"${clientId}\\\", \\\"newName\\\": \\\"${newName}\\\"}\",\n"
        "        \"content_type\": \"application/json\",\n"
        "        \"headers\": [\n"
        "          {\"name\": \"Authorization\", \"value\": \"Bearer ${auth_token}\"},\n"
        "          {\"name\": \"Content-Type\", \"value\": \"application/json\"}\n"
        "        ],\n"
        "        \"extractions\": [\n"
        "          {\"json_path\": \"$.clientId\", \"var_name\": \"update_client_clientId\"}\n"
        "        ],\n"
        "        \"assertions\": [\n"
        "          {\"type\": \"response_code\", \"pattern\": \"200\"},\n"
        "          {\"type\": \"text_response\", \"pattern\": \"updated\"}\n"
        "        ]\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n\n"
        "⚠️ Each http_sampler object must include an \"assertions\" key:\n"
        "- It must contain at least a response code assertion like:\n"
        "  {\"type\": \"response_code\", \"pattern\": \"200\"}\n\n"
        "✅ Place this \"assertions\" array INSIDE each http_sampler, not as a global field.\n"
        "- **Double Quotes for Property Names**: ALL property names (keys) MUST be enclosed in double quotes (e.g., `\"name\": \"value\"`, NOT `name: \"value\"`).\n"
        "DO NOT include markdown code blocks. Respond only with pure JSON object.\n"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": final_llm_prompt_for_json_output}], # Use the JSON-focused prompt
                "role": "user"
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        gemini_output = response.json()
        response_text = gemini_output.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        if not response_text:
            st.error("❌ Gemini returned empty output. Try reducing Swagger size or simplify the prompt.")
            return None

        st.info("ℹ️ Gemini raw response:")
        st.code(response_text, language="json")

        # Strip markdown if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        # Parse JSON and validate structure
        try:
            # First attempt to parse directly
            test_plan_obj = json.loads(response_text)
            if isinstance(test_plan_obj, dict) and "test_plan" in test_plan_obj:
                st.success("✅ Parsed JMeter-style test plan from Gemini.")
                return test_plan_obj # Return the dictionary directly
            else:
                st.error("⚠️ LLM output does not contain a valid 'test_plan' structure after initial parse.")
                st.code(response_text)
                return None
        except json.JSONDecodeError as e:
            st.warning(f"Initial JSON parsing failed with: {e}. Attempting repair...")
            logger.warning(f"Initial JSON parsing failed with: {e}. Raw response text (truncated for log): {response_text[:500]}...")

            repaired_response_text = _repair_truncated_json(response_text)

            try:
                test_plan_obj = json.loads(repaired_response_text)
                if isinstance(test_plan_obj, dict) and "test_plan" in test_plan_obj:
                    st.session_state.llm_structured_scenario = test_plan_obj # Set session state on successful repair
                    st.success("✅ Parsed JMeter-style test plan from Gemini after repair.")
                    return test_plan_obj
                else:
                    st.error("⚠️ LLM output does not contain a valid 'test_plan' structure even after repair.")
                    st.code(repaired_response_text)
                    return None
            except json.JSONDecodeError as repair_e:
                st.error("❌ JSON parsing failed even after repair.")
                st.code(repaired_response_text) # Show the result of the repair attempt
                st.exception(repair_e) # Show the new error if repair failed
                return None

    except requests.RequestException as e:
        st.error(f"❌ Gemini API call failed: {str(e)}")
        return None

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

def _init_dummy_db():
    """Initializes a dummy SQLite database if it doesn't exist."""
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
            logger.error(f"Error creating dummy DB during init: {ex}")


def main():
    _init_dummy_db() # Call the dummy DB initialization here

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
        st.session_state.csv_content_download = None # Changed to store a dict of filenames to content
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

    # Default LLM prompt for initial display (removed f-prefix for backslash compatibility)
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
        * **`body`**: If the request has a JSON body, you MUST provide it as a **JSON string**. For example, if the body is `{"key": "value"}` then output `"{\\"key\\": \\"value\\"}"`. Values within this string can be static, or JMeter variables (e.g., `"${clientId}"`) if sourced from CSV.
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
    Your response MUST be ONLY a valid JSON object. **Ensure the JSON is always perfectly formed and complete, with all brackets and commas correctly placed and no trailing commas or incomplete structures. DO NOT include comments in the JSON output.**

    ```json
    {
      "test_plan": {
        "thread_group": {
          "num_threads": 1,
          "ramp_up": 1
        },
        "csv_data_set_config": {
          "filename": "ClientUpdate.csv",
          "variable_names": ["clientId", "newName"]
        },
        "http_samplers": [
          {
            "name": "PUT /v1/update-client",
            "method": "PUT",
            "path": "/v1/update-client",
            "body": "{\\\"clientId\\\": \\\"${clientId}\\\", \\\"newName\\\": \\\"${newName}\\\"}",
            "content_type": "application/json",
            "headers": [
              {"name": "Authorization", "value": "Bearer ${auth_token}"},
              {"name": "Content-Type", "value": "application/json"}
            ],
            "extractions": [
              {"json_path": "$.clientId", "var_name": "update_client_clientId"}
            ],
            "assertions": [
              {"type": "response_code", "pattern": "200"},
              {"type": "text_response", "pattern": "updated"}
            ]
          }
        ]
      }
    }
    ```
    For each `path_params`, `query_params`, `body`, `headers`, `assertions`, and `extractions` item, **adhere to these strict rules**:
    -   **`path_params`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable like `${variable_name}`).
    -   **`query_params`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable like `${variable_name}`).
    -   **`body`**: A JSON string (e.g., `"{\\"key\\": \\"value\\"}"`). Values can be JMeter variables. Use `null` if no body.
    -   **`content_type`**: String like "application/json". Use `null` if no body.
    -   **`headers`**: Array of objects. Each object must have `name` (string) and `value` (string, can be a JMeter variable).
    -   **`assertions`**: Array of objects. Each object must have `type` (string, "response_code" or "text_response") and `pattern` (string). This is now inside `http_sampler`.
    -   **`extractions`**: Array of objects. Each object must have `json_path` (string, JSONPath expression) and `var_name` (string, name of the JMeter variable to store the extracted value). This is now inside `http_sampler`.
    - **Double Quotes for Property Names**: ALL property names (keys) MUST be enclosed in double quotes (e.g., `\"name\": \"value\"`, NOT `name: \"value\"`).
    DO NOT include markdown code blocks. Respond only with pure JSON object.
"""

    if 'llm_prompt_text' not in st.session_state:
        st.session_state.llm_prompt_text = default_llm_prompt


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
                                    _init_dummy_db() # Call the encapsulated function here
                                    st.success(f"Dummy SQLite database created at {db_file_path} with sample data.")
                                    st.rerun()
                                        
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
            value=st.session_state.llm_prompt_text,
            height=600, # Increased height to show more of the detailed prompt
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
                        if structured_scenario and 'test_plan' in structured_scenario: # Check structured_scenario directly
                            test_plan_data = structured_scenario['test_plan'] # Access directly
                            
                            st.markdown("**Test Plan Overview:**")
                            st.json(test_plan_data) # Display the full structured response
                            
                            st.markdown("---")
                            st.markdown(f"**Thread Group:**")
                            st.write(f"Number of Threads: `{test_plan_data['thread_group']['num_threads']}`")
                            st.write(f"Ramp-up Time: `{test_plan_data['thread_group']['ramp_up']}`")

                            # Check for the single 'csv_data_set_config' first as per prompt instruction
                            llm_csv_config = test_plan_data.get('csv_data_set_config') 
                            if isinstance(llm_csv_config, list) and llm_csv_config:
                                llm_csv_config = llm_csv_config[0] # Take the first item if it's a list
                            
                            # Collect all CSV data set configs from LLM response (handling multiple, if LLM outputs them)
                            all_llm_csv_configs = []
                            if isinstance(llm_csv_config, dict) and llm_csv_config.get('variable_names') and llm_csv_config.get('filename'):
                                all_llm_csv_configs.append(llm_csv_config)
                            
                            for key, value in test_plan_data.items():
                                if key.startswith('csv_data_set_config_') and isinstance(value, dict) and 'variable_names' in value and 'filename' in value:
                                    if value not in all_llm_csv_configs: # Avoid duplicates if 'csv_data_set_config' was also dynamically named
                                         all_llm_csv_configs.append(value)

                            if all_llm_csv_configs:
                                st.markdown("---")
                                st.markdown(f"**CSV Data Set Configs:**")
                                for csv_conf_item in all_llm_csv_configs:
                                    st.write(f"- **Filename:** `{csv_conf_item['filename']}`, **Variable Names:** `{', '.join(csv_conf_item['variable_names'])}`")
                            else:
                                st.info("No CSV Data Set Config found in AI-generated design.")

                            st.markdown("---")
                            st.markdown(f"**HTTP Samplers:**")
                            if 'http_samplers' in test_plan_data and isinstance(test_plan_data['http_samplers'], list):
                                for i, sampler in enumerate(test_plan_data['http_samplers']):
                                    st.markdown(f"---")
                                    st.markdown(f"**Sampler {i + 1}: {sampler.get('name', 'Unnamed Sampler')}**")
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
                                        else:  # It's a string (from LLM) or other primitive
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
                                            st.write(
                                                f"- Type: `{assertion['type']}`, Pattern: `{assertion['pattern']}`")

                                    if 'extractions' in sampler and sampler['extractions']:
                                        st.markdown(f"**Extractions for this Sampler:**")
                                        # Corrected to check if 'extraction' is a dictionary
                                        for extraction in sampler['extractions']:
                                            if isinstance(extraction, dict):
                                                json_path = extraction.get('json_path')
                                                var_name = extraction.get('var_name')
                                                if json_path and var_name: # Only display if both are present
                                                    st.write(
                                                        f"- JSONPath: `{json_path}` -> Var: `{var_name}`")
                                                else:
                                                    logger.warning(f"Malformed extraction entry found and skipped: {extraction}")
                                            else:
                                                logger.warning(f"Non-dictionary extraction entry found and skipped: {extraction}")
                            else:
                                st.info("No HTTP Samplers found in AI-generated design.")
                            
                            time.sleep(0.1) # Simulate streaming delay
                            
                    else:
                        # This else block is now reached ONLY if structured_scenario is None
                        # after initial LLM call and repair attempts, implying a true failure.
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
            new_scenario_configs = []
            extracted_variables_map = {}

            # If AI-structured scenario exists, use it as the base for manual refinement
            if st.session_state.llm_structured_scenario and 'test_plan' in st.session_state.llm_structured_scenario:
                llm_response_test_plan = st.session_state.llm_structured_scenario['test_plan']
                http_samplers_list = llm_response_test_plan.get('http_samplers', [])

                for http_sampler in http_samplers_list:
                    # Deep copy the http_sampler to avoid modifying the original AI design in session state directly
                    current_sampler_config = json.loads(json.dumps(http_sampler)) 

                    # Populate extracted_variables_map from the current AI sampler's extractions
                    # Corrected to check if 'extractions' exists and is a list
                    if 'extractions' in current_sampler_config and isinstance(current_sampler_config['extractions'], list):
                        for extractor_data in current_sampler_config['extractions']:
                            if isinstance(extractor_data, dict) and 'var_name' in extractor_data:
                                extracted_variables_map[extractor_data['var_name'].lower()] = f"${{{extractor_data['var_name']}}}"
                    
                    new_scenario_configs.append(current_sampler_config)
            else:
                # Fallback to existing manual refinement logic if no AI scenario is present
                st.session_state.mappings = DataMapper.suggest_mappings(
                    st.session_state.swagger_endpoints,
                    st.session_state.db_tables_schema,
                    st.session_state.db_sampled_data
                )

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
                            auth_header_name_lower = st.session_state.auth_header_name.lower()
                            # Check if Authorization header is already present (case-insensitive)
                            if not any(h_name.lower() == auth_header_name_lower for h_name in request_config['headers'].keys()):
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
                                # For OpenAPI 3.0+
                                content_types = resolved_endpoint_data['requestBody'].get('content', {})
                                first_content_type = next(iter(content_types.keys()), None) # Get content type from requestBody
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
                                        resp_schema = next(iter(content_types_resp.values())).get('schema')
                                
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

            # structured_scenario is now a list containing one item (the test_plan object)
            if st.session_state.llm_structured_scenario and 'test_plan' in st.session_state.llm_structured_scenario: # Access directly
                llm_response_test_plan = st.session_state.llm_structured_scenario['test_plan'] # Access directly
                
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
                        # Corrected to handle potential missing 'json_path' key safely
                        for extractor_data in http_sampler.get('extractions', []):
                            if isinstance(extractor_data, dict):
                                json_path_expr = extractor_data.get('json_path')
                                var_name = extractor_data.get('var_name')
                                if json_path_expr and var_name:
                                    request_config['json_extractors'].append({
                                        "json_path_expr": json_path_expr,
                                        "var_name": var_name
                                    })
                                    extracted_variables_map[var_name.lower()] = f"${{{var_name}}}"
                                else:
                                    logger.warning(f"Malformed extraction entry from LLM and skipped: {extractor_data}")
                            else:
                                logger.warning(f"Non-dictionary extraction entry from LLM found and skipped: {extractor_data}")


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
            # Clear download states if preconditions are not met
            st.session_state.jmx_content_download = None
            st.session_state.csv_content_download = None
            st.session_state.mapping_metadata_download = None
            return
        if not st.session_state.llm_structured_scenario: # Changed from scenario_requests_configs to llm_structured_scenario
            st.error("Please generate an AI script design before generating JMeter script.")
            # Clear download states if preconditions are not met
            st.session_state.jmx_content_download = None
            st.session_state.csv_content_download = None
            st.session_state.mapping_metadata_download = None
            return
        if not st.session_state.db_tables_schema or not st.session_state.db_sampled_data:
            # Only block if scenario actually *requires* DB data (e.g., if LLM output used DB values)
            # For now, a warning is sufficient. If scenario_requests_configs has CSV vars, it might fail later.
            st.warning("Database schema or sampled data not loaded. JMeter script might not be fully data-driven if your scenario relies on DB data.")

        st.info("Generating JMeter script... Please wait.")

        try:
            # Call the generate_jmx_from_llm method from JMeterScriptGenerator
            jmx_content, csv_contents, mapping_metadata = JMeterScriptGenerator.generate_jmx_from_llm(
                llm_structured_scenario=st.session_state.llm_structured_scenario,
                swagger_endpoints=st.session_state.swagger_endpoints,
                db_tables_schema=st.session_state.db_tables_schema,
                db_sampled_data=st.session_state.db_sampled_data,
                test_plan_name=st.session_state.test_plan_name,
                thread_group_name=st.session_state.thread_group_name,
                num_users=st.session_state.num_users_input,
                ramp_up_time=st.session_state.ramp_up_time_input,
                loop_count=st.session_state.loop_count_input_specific if st.session_state.loop_count_option == "Specify iterations" else -1,
                global_constant_timer_delay=st.session_state.constant_timer_delay_ms,
                enable_auth_flow=st.session_state.enable_auth_flow,
                auth_login_endpoint_path=st.session_state.auth_login_endpoint_path,
                auth_login_method=st.session_state.auth_login_method,
                auth_login_body_template=st.session_state.auth_login_body_template,
                auth_token_json_path=st.session_state.auth_token_json_path,
                auth_header_name=st.session_state.auth_header_name,
                auth_header_prefix=st.session_state.auth_header_prefix,
                full_swagger_spec=st.session_state.swagger_parser.swagger_data if st.session_state.swagger_parser else {},
                enable_setup_teardown_thread_groups=st.session_state.enable_setup_teardown_thread_groups,
                current_swagger_url=st.session_state.current_swagger_url
            )

            st.session_state.jmx_content_download = jmx_content
            st.session_state.csv_content_download = csv_contents
            st.session_state.mapping_metadata_download = mapping_metadata

            if jmx_content:
                st.success("JMeter script generated successfully!")
            else:
                st.error("Failed to generate JMeter script.")

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
        for filename, content in st.session_state.csv_content_download.items():
            st.download_button(
                label=f"Download CSV Data ({filename})",
                data=content.encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                key=f"download_csv_{filename.replace('.', '_')}" # Unique key for each button
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
    main()
