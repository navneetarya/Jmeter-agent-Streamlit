import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import sys
import os
import re
from datetime import datetime, date
import io
import zipfile
import time
from urllib.parse import quote_plus

# Import utility classes
from utils.jmeter_generator import JMeterScriptGenerator
from utils.swagger_parser import SwaggerParser, SwaggerEndpoint
from utils.data_mapper import DataMapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)): return obj.isoformat()
        if isinstance(obj, bytes):
            try: return obj.decode('utf-8')
            except UnicodeDecodeError: return obj.hex()
        return json.JSONEncoder.default(self, obj)

def generate_csv_previews(table_to_columns: Dict[str, set], db_sampled_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    csv_files = {}
    for table, columns in table_to_columns.items():
        if table in db_sampled_data:
            df_table = db_sampled_data[table]
            columns_to_include = [col for col in columns if col in df_table.columns]
            if columns_to_include:
                csv_filename = f"{table}_data.csv"
                jmeter_df = pd.DataFrame()
                for col in columns_to_include:
                    jmeter_variable_name = f"csv_{table}_{col}"
                    jmeter_df[jmeter_variable_name] = df_table[col]
                csv_files[csv_filename] = jmeter_df.to_csv(index=False)
    return csv_files

# Main application
def main():
    st.set_page_config(page_title="AI JMeter Script Generator", page_icon="⚡", layout="wide")
    st.title("⚡ AI-Powered JMeter Script Generator")
    st.markdown("Upload your raw API spec and DB files, describe a scenario, and get a ready-to-run JMeter test package.")

    st.session_state.setdefault('designed_test_cases', [])
    st.session_state.setdefault('generated_csv_previews', {})
    st.session_state.setdefault('generated_artifacts', None)
    st.session_state.setdefault('ai_mappings', [])
    st.session_state.setdefault('relevant_endpoints', [])
    st.session_state.setdefault('full_swagger_spec', None)
    st.session_state.setdefault('validation_warnings', [])

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_swagger_file = st.file_uploader("Upload Raw Swagger/OpenAPI Spec (JSON)", type="json")
        uploaded_schema_file = st.file_uploader("Upload Database Schema (JSON)", type="json")
        uploaded_data_file = st.file_uploader("Upload Sampled Database Data (JSON)", type="json")
        api_key = st.text_input("Groq API Key", type="password", help="Get a free key from https://console.groq.com/keys")
        st.subheader("JMeter Test Plan Settings")
        st.session_state.test_plan_name = st.text_input("Test Plan Name", value="AI Generated Test Plan")
        st.session_state.thread_group_name = st.text_input("Thread Group Name", value="Simulated Users")
        st.subheader("Load Profile")
        num_users = st.number_input("Threads", 1, value=10)
        ramp_up_time = st.number_input("Ramp-up (s)", 0, value=10)
        loop_count = st.number_input("Loops", -1, value=1)

    st.header("Step 1: Identify Endpoints & Map Data")
    user_prompt = st.text_area("Enter your test scenario prompt:", value="Create a test plan for all GET endpoints.", height=100)

    if st.button("✨ 1. Analyze and Map Data"):
        if not all([uploaded_swagger_file, uploaded_schema_file, api_key, user_prompt]):
            st.warning("Please upload all three JSON files, provide an API key, and enter a prompt.")
        else:
            for key in ['designed_test_cases', 'generated_artifacts', 'ai_mappings', 'relevant_endpoints', 'validation_warnings']:
                if isinstance(st.session_state.get(key), list): st.session_state[key] = []
                else: st.session_state[key] = None
            
            with st.spinner("Step 1/3: Parsing Swagger and filtering endpoints..."):
                try:
                    uploaded_swagger_file.seek(0)
                    st.session_state.full_swagger_spec = json.load(uploaded_swagger_file)
                    parser = SwaggerParser(st.session_state.full_swagger_spec)
                    swagger_endpoints = parser.parse()
                    
                    prompt_lower = user_prompt.lower()
                    if "all get" in prompt_lower and "all post" in prompt_lower:
                        st.session_state.relevant_endpoints = [ep for ep in swagger_endpoints if ep.method in ['GET', 'POST']]
                    elif "all get" in prompt_lower:
                        st.session_state.relevant_endpoints = [ep for ep in swagger_endpoints if ep.method == 'GET']
                    elif "all post" in prompt_lower:
                        st.session_state.relevant_endpoints = [ep for ep in swagger_endpoints if ep.method == 'POST']
                    else: # Default to all endpoints
                        st.session_state.relevant_endpoints = swagger_endpoints
                    
                    if not st.session_state.relevant_endpoints: st.error("No relevant endpoints found based on your prompt."); st.stop()
                    
                    def find_schema_properties(schema, found_props):
                        if not isinstance(schema, dict) or 'properties' not in schema: return
                        for prop_name, details in schema['properties'].items():
                            if prop_name not in found_props:
                                found_props[prop_name] = {'name': prop_name, 'in': 'body', 'type': details.get('type')}
                            if details.get('type') == 'object': find_schema_properties(details, found_props)
                            elif details.get('type') == 'array': find_schema_properties(details.get('items', {}), found_props)

                    params_to_map_dict = {}
                    for ep in st.session_state.relevant_endpoints:
                        for param in ep.parameters:
                            if param.get('name') and param['name'] not in params_to_map_dict:
                                params_to_map_dict[param['name']] = param
                        if ep.body_schema: find_schema_properties(ep.body_schema, params_to_map_dict)

                    params_to_map = list(params_to_map_dict.values())
                    st.success(f"Found {len(st.session_state.relevant_endpoints)} endpoints and {len(params_to_map)} unique parameters.")

                except Exception as e: st.error(f"Endpoint analysis failed: {e}"); logger.error("Endpoint analysis", exc_info=True); st.stop()

            # --- START: INTELLIGENT SCHEMA PRUNING ---
            with st.spinner("Step 2/3: Pruning database schema for relevant tables..."):
                try:
                    uploaded_schema_file.seek(0)
                    full_db_schema = json.load(uploaded_schema_file)
                    
                    param_names = {p['name'].lower().replace('_', '') for p in params_to_map}
                    relevant_tables = set()

                    for table_name, columns in full_db_schema.items():
                        for column in columns:
                            col_name = column['name'].lower().replace('_', '')
                            # Check if any part of the param name is in the column name
                            if any(param_name in col_name for param_name in param_names):
                                relevant_tables.add(table_name)
                                break # Move to the next table once a match is found
                    
                    pruned_schema = {table: full_db_schema[table] for table in relevant_tables}
                    st.success(f"Pruned schema to {len(pruned_schema)} relevant tables (out of {len(full_db_schema)}).")

                except Exception as e: st.error(f"Schema pruning failed: {e}"); logger.error("Schema pruning", exc_info=True); st.stop()
            # --- END: INTELLIGENT SCHEMA PRUNING ---

            with st.spinner("Step 3/3: AI is mapping parameters to the pruned schema..."):
                try:
                    mapping_result = DataMapper.get_ai_powered_mappings(params_to_map, pruned_schema, api_key)
                    if mapping_result and "parameter_mappings" in mapping_result:
                        st.session_state.ai_mappings = mapping_result["parameter_mappings"]
                        st.success("AI data mapping complete!")
                    else: st.error("AI did not return valid mappings."); logger.error(f"Invalid map: {mapping_result}"); st.stop()
                except Exception as e: st.error(f"AI mapping failed: {e}"); logger.error("AI mapping", exc_info=True); st.stop()

    st.header("Step 2: Assemble & Review Test Plan")
    if st.session_state.ai_mappings:
        # The rest of the file remains exactly the same.
        with st.expander("View Final Aggregated AI Data Mapping Results"):
            st.dataframe(st.session_state.ai_mappings, use_container_width=True)

        def find_case_insensitive_column(df: pd.DataFrame, column_name: str) -> Optional[str]:
            for col in df.columns:
                if col.lower() == column_name.lower(): return col
            return None

        def get_value_from_heuristics(param_name: str, param_type: Optional[str] = None):
            name = param_name.lower()
            if any(k in name for k in ['uuid', 'guid']) or name.endswith('id'): return "${__UUID()}"
            if 'email' in name: return "test_" + "${__Random(1,99999)}" + "@example.com"
            if 'password' in name: return "Admin@123!"
            if any(k in name for k in ['phone', 'mobile']): return "+91" + "${__RandomString(10,0123456789)}"
            if 'pincode' in name or 'zipcode' in name: return "${__RandomString(6,0123456789)}"
            if 'token' in name or 'session' in name: return "${__UUID()}"
            if name.endswith('code') or name.endswith('cd'): return "CODE-" + "${__Random(100,999)}"
            if name == 'age': return "${__Random(18,60)}"
            if 'birthdate' in name or name == 'dob': return "${__timeShift(yyyy-MM-dd,,P-25Y)}"
            if any(k in name for k in ['date', 'timestamp', 'createdat', 'updatedat', '_on']): return "${__time(yyyy-MM-dd'T'HH:mm:ss.SSSZ)}"
            if any(k in name for k in ['name']): return "AutoUser-" + "${__Random(100,999)}"
            if 'description' in name or 'notes' in name or 'remark' in name: return "Generated by test automation."
            if name in ['lat', 'latitude']: return "28.6139"
            if name in ['lon', 'lng', 'longitude']: return "77.2090"
            if 'city' in name: return "Delhi"
            if 'country' in name: return "IN"
            if 'price' in name or 'amount' in name: return "${__Random(10,1000)}.50"
            if name.startswith('is') or name.startswith('has') or 'flag' in name: return "true"
            if 'status' in name or 'state' in name or 'type' in name: return "active"
            if 'role' in name or 'permission' in name: return "user"
            if 'url' in name or 'link' in name: return "https://example.com"
            if 'image' in name or 'avatar' in name: return "https://i.pravatar.cc/150"
            if 'version' in name: return "1.0.0"
            if param_type:
                if param_type == 'integer': return "${__Random(1,1000)}"
                if param_type == 'number': return "${__Random(1,1000)}.0"
                if param_type == 'boolean': return "true"
            return f"STATIC_{param_name}"
        
        def get_validated_intelligent_value(param, mapping_dict, db_data, table_to_columns, warnings_list):
            param_name = param['name']
            ai_map = mapping_dict.get(param_name, {})
            table, mapped_column = ai_map.get("mapped_table"), ai_map.get("mapped_column")
            
            if table and mapped_column and table in db_data:
                actual_column_name = find_case_insensitive_column(db_data[table], mapped_column)
                if actual_column_name:
                    value = f"${{csv_{table}_{actual_column_name}}}"
                    if table not in table_to_columns: table_to_columns[table] = set()
                    table_to_columns[table].add(actual_column_name)
                    return value
                else:
                    warnings_list.append(f"**`{param_name}`** → Mapped to `{table}.{mapped_column}`, but column not found. Falling back.")
            
            return get_value_from_heuristics(param_name, param.get('type'))

        def build_body_from_schema(schema, mapping_dict, db_data, table_to_columns, warnings_list):
            if not schema: return None
            schema_type = schema.get('type')
            
            if schema_type == 'object':
                body = {}
                if 'properties' not in schema: return body
                for prop_name, details in schema['properties'].items():
                    mock_param = {'name': prop_name, 'type': details.get('type')}
                    body[prop_name] = get_validated_intelligent_value(mock_param, mapping_dict, db_data, table_to_columns, warnings_list)
                return body
            
            if schema_type == 'array':
                items_schema = schema.get('items', {})
                items_type = items_schema.get('type')
                if items_type == 'object':
                    return [build_body_from_schema(items_schema, mapping_dict, db_data, table_to_columns, warnings_list)]
                else:
                    mock_param = {'name': 'array_item', 'type': items_type}
                    return [get_validated_intelligent_value(mock_param, mapping_dict, db_data, table_to_columns, warnings_list)]
            
            return None

        if st.button("🛠️ 2. Assemble Full Test Cases"):
            with st.spinner("Assembling final test cases and generating CSV previews..."):
                try:
                    st.session_state.validation_warnings = []
                    mapping_dict = {m['parameter_name']: m for m in st.session_state.ai_mappings}
                    uploaded_data_file.seek(0)
                    db_sampled_data = {k: pd.DataFrame(v) for k, v in json.load(uploaded_data_file).items()}
                    
                    final_test_cases = []
                    table_to_columns_for_csv = {}
                    
                    for ep in st.session_state.relevant_endpoints:
                        final_path = ep.path
                        query_params_dict = {}
                        
                        for param in ep.parameters:
                            value = get_validated_intelligent_value(param, mapping_dict, db_sampled_data, table_to_columns_for_csv, st.session_state.validation_warnings)
                            if param.get('in') == 'path': final_path = final_path.replace(f"{{{param['name']}}}", value)
                            elif param.get('in') == 'query': query_params_dict[param['name']] = value
                        
                        if query_params_dict:
                            query_parts = []
                            for k, v in query_params_dict.items():
                                val_str = str(v)
                                if '${' in val_str and '}' in val_str:
                                    query_parts.append(f"{k}={val_str}")
                                else:
                                    query_parts.append(f"{k}={quote_plus(val_str)}")
                            
                            query_string = "&".join(query_parts)
                            final_path += f"?{query_string}"

                        case = {"name": f"{ep.method} - {ep.path}", "method": ep.method, "path": final_path,
                                "headers": {}, "parameters": [], "body": None, "extractions": [],
                                "assertions": [{"type": "response_code", "pattern": "201" if ep.method == 'POST' else "200"}]}

                        if ep.method in ["POST", "PUT", "PATCH"] and ep.body_schema:
                            case["body"] = build_body_from_schema(ep.body_schema, mapping_dict, db_sampled_data, table_to_columns_for_csv, st.session_state.validation_warnings)
                        
                        if case.get("body"): case["headers"]["Content-Type"] = "application/json"
                        if ep.security_schemes: case["headers"]["Authorization"] = "Bearer ${authToken}"
                        
                        final_test_cases.append(case)
                    
                    st.session_state.designed_test_cases = final_test_cases
                    st.session_state.generated_csv_previews = generate_csv_previews(table_to_columns_for_csv, db_sampled_data)
                    st.success("Test cases assembled successfully!")
                except Exception as e: st.error(f"Assembly failed: {e}"); logger.error("Assembly", exc_info=True)

        if st.session_state.validation_warnings:
            st.warning("Mapping Validation Report")
            report_str = "The following AI mappings could not be validated against your sample data and fell back to heuristic generation:\n\n" + "\n".join(f"- {w}" for w in set(st.session_state.validation_warnings))
            st.markdown(report_str)

    st.header("Step 3: Generate and Download All Artifacts")
    if st.session_state.designed_test_cases:
        with st.expander("Preview Final Assembled Test Cases (JSON)"):
            st.json(st.session_state.designed_test_cases)
        with st.expander("Preview Generated CSVs"):
            if st.session_state.generated_csv_previews:
                for filename, content in st.session_state.generated_csv_previews.items():
                    st.subheader(f"`{filename}`"); st.dataframe(pd.read_csv(io.StringIO(content)))
            else: st.info("No CSV files were needed for this plan.")
    
    if st.button("🚀 Generate JMeter Plan and Artifacts", type="primary", disabled=(not st.session_state.designed_test_cases)):
        with st.spinner("Translating design into JMX, YAML, and CSV files..."):
            try:
                uploaded_data_file.seek(0)
                db_sampled_data = {k: pd.DataFrame(v) for k, v in json.load(uploaded_data_file).items()}
                
                jmx_content, csv_files, json_plan, yaml_plan = JMeterScriptGenerator.generate_jmx_and_artifacts(
                    designed_test_cases=st.session_state.designed_test_cases,
                    db_sampled_data=db_sampled_data,
                    test_plan_name=st.session_state.test_plan_name,
                    thread_group_name=st.session_state.thread_group_name,
                    num_users=num_users, ramp_up_time=ramp_up_time, loop_count=loop_count,
                    full_swagger_spec=st.session_state.full_swagger_spec
                )
                
                st.session_state.generated_artifacts = {"jmx": jmx_content, "csv": csv_files, "json": json_plan, "yaml": yaml_plan}
            except Exception as e: st.error(f"Generation failed: {e}"); logger.error("Generation", exc_info=True)

    if st.session_state.generated_artifacts:
        st.subheader("Preview Final Artifacts")
        tabs = st.tabs(["JSON Design", "JMX (XML)", "YAML Design"])
        with tabs[0]: st.code(st.session_state.generated_artifacts.get("json", ""), language="json")
        with tabs[1]: st.code(st.session_state.generated_artifacts.get("jmx", ""), language="xml")
        with tabs[2]: st.code(st.session_state.generated_artifacts.get("yaml", ""), language="yaml")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            if st.session_state.generated_artifacts.get("jmx"): zf.writestr("test_plan.jmx", st.session_state.generated_artifacts["jmx"])
            if st.session_state.generated_artifacts.get("csv"):
                for filename, content in st.session_state.generated_artifacts["csv"].items(): zf.writestr(filename, content)
            if st.session_state.generated_artifacts.get("yaml"): zf.writestr("test_plan_design.yaml", st.session_state.generated_artifacts["yaml"])
        st.download_button(label="📥 Download Test Package (.zip)", data=zip_buffer.getvalue(), file_name="jmeter_ai_test_package.zip", mime="application/zip")

if __name__ == "__main__":
    main()