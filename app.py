import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
import sys
import os
import re
from datetime import datetime, date
import io
import zipfile
import time

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

    # Initialize session state
    if 'designed_test_cases' not in st.session_state: st.session_state.designed_test_cases = []
    if 'generated_csv_previews' not in st.session_state: st.session_state.generated_csv_previews = {}
    if 'generated_artifacts' not in st.session_state: st.session_state.generated_artifacts = None
    if 'ai_mappings' not in st.session_state: st.session_state.ai_mappings = []
    if 'relevant_endpoints' not in st.session_state: st.session_state.relevant_endpoints = []
    if 'full_swagger_spec' not in st.session_state: st.session_state.full_swagger_spec = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        # UPDATED: Changed label to be more specific
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
            st.session_state.designed_test_cases = []
            st.session_state.generated_artifacts = None
            
            with st.spinner("Step 1/2: Filtering endpoints and collecting parameters..."):
                uploaded_swagger_file.seek(0)
                # --- THIS IS THE CORRECTED LOGIC ---
                st.session_state.full_swagger_spec = json.load(uploaded_swagger_file)
                if not isinstance(st.session_state.full_swagger_spec, dict):
                    st.error("Uploaded Swagger file is not a valid JSON object (dictionary). Please upload the raw Swagger/OpenAPI specification.")
                    st.stop()
                
                parser = SwaggerParser(st.session_state.full_swagger_spec)
                swagger_endpoints = parser.parse()
                
                relevant_endpoints = []
                prompt_lower = user_prompt.lower()
                if "all get" in prompt_lower: relevant_endpoints = [ep for ep in swagger_endpoints if ep.method == 'GET']
                elif "all post" in prompt_lower: relevant_endpoints = [ep for ep in swagger_endpoints if ep.method == 'POST']
                else: relevant_endpoints = swagger_endpoints
                
                if not relevant_endpoints: st.error("No relevant endpoints found based on your prompt."); st.stop()
                st.session_state.relevant_endpoints = relevant_endpoints
                
                params_to_map = []
                unique_param_names = set()
                for ep in relevant_endpoints:
                    for param in ep.parameters:
                        if param.get('name') and param['name'] not in unique_param_names:
                            params_to_map.append({"name": param['name'], "in": param.get('in', 'unknown')})
                            unique_param_names.add(param['name'])
                st.success(f"Found {len(relevant_endpoints)} endpoints and {len(params_to_map)} unique parameters to map.")

            with st.spinner("Step 2/2: AI is mapping data parameters..."):
                uploaded_schema_file.seek(0)
                db_schema = json.load(uploaded_schema_file)
                mapping_result = DataMapper.get_ai_powered_mappings(params_to_map, db_schema, api_key)
                if "parameter_mappings" in mapping_result:
                    st.session_state.ai_mappings = mapping_result["parameter_mappings"]
                    st.success("AI data mapping complete!")
                else: st.error("AI did not return a valid mapping structure."); st.stop()

    st.header("Step 2: Assemble & Review Test Plan")
    if st.session_state.ai_mappings:
        with st.expander("View AI Data Mapping Results"):
            st.dataframe(st.session_state.ai_mappings, use_container_width=True)

        if st.button("🛠️ 2. Assemble Full Test Cases"):
            with st.spinner("Assembling final test cases and generating CSV previews..."):
                mapping_dict = {m['parameter_name']: m for m in st.session_state.ai_mappings}
                final_test_cases = []
                table_to_columns_for_csv = {}
                
                for i, ep in enumerate(st.session_state.relevant_endpoints):
                    case_name = f"{ep.method} - {ep.path}"
                    case = {"name": case_name, "method": ep.method, "path": ep.path, "headers": {}, "parameters": [], "body": {}, "extractions": [], "assertions": [{"type": "response_code", "pattern": "200"}]}
                    if ep.method in ["POST", "PUT", "PATCH"]: case["headers"]["Content-Type"] = "application/json"
                    if ep.security_schemes: case["headers"]["Authorization"] = "Bearer ${authToken}"

                    for param in ep.parameters:
                        param_name = param['name']
                        mapping = mapping_dict.get(param_name, {})
                        table, column = mapping.get("mapped_table"), mapping.get("mapped_column")
                        value = f"STATIC_{param_name}"
                        if table and column:
                            value = f"${{csv_{table}_{column}}}"
                            if table not in table_to_columns_for_csv: table_to_columns_for_csv[table] = set()
                            table_to_columns_for_csv[table].add(column)
                        case['parameters'].append({"name": param_name, "value": value, "in": param.get('in')})
                    
                    for status_code, response_def in ep.responses.items():
                        if status_code.startswith('2') and response_def and response_def.get('schema') and response_def.get('schema').get('properties'):
                            for prop_name, prop_details in response_def['schema']['properties'].items():
                                if 'id' in prop_name.lower() or 'uid' in prop_name.lower():
                                    case['extractions'].append({"variable_name": f"{ep.method.lower()}_{prop_name}", "json_path": f"$.{prop_name}"})
                    final_test_cases.append(case)
                
                st.session_state.designed_test_cases = final_test_cases
                uploaded_data_file.seek(0)
                db_sampled_data = {k: pd.DataFrame(v) for k, v in json.load(uploaded_data_file).items()}
                st.session_state.generated_csv_previews = generate_csv_previews(table_to_columns_for_csv, db_sampled_data)
                st.success("Test cases assembled successfully!")
    else:
        st.info("Click 'Analyze and Map Data' to begin.")

    st.header("Step 3: Generate and Download All Artifacts")
    if st.session_state.designed_test_cases:
        with st.expander("Preview Final Assembled Test Cases (JSON)"):
            st.json({"designed_test_cases": st.session_state.designed_test_cases})
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
            except Exception as e:
                st.error(f"An error occurred during artifact generation: {e}")
                logger.error("Generation failed", exc_info=True)

    if st.session_state.generated_artifacts:
        st.subheader("Preview Final Artifacts")
        json_tab, jmx_tab, yaml_tab = st.tabs(["JSON Design", "JMX (XML)", "YAML Design"])
        with json_tab: st.code(st.session_state.generated_artifacts.get("json", ""), language="json")
        with jmx_tab: st.code(st.session_state.generated_artifacts.get("jmx", ""), language="xml")
        with yaml_tab: st.code(st.session_state.generated_artifacts.get("yaml", ""), language="yaml")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            if st.session_state.generated_artifacts.get("jmx"): zip_file.writestr("test_plan.jmx", st.session_state.generated_artifacts["jmx"])
            if st.session_state.generated_artifacts.get("csv"):
                for filename, content in st.session_state.generated_artifacts["csv"].items():
                    zip_file.writestr(filename, content)
            if st.session_state.generated_artifacts.get("yaml"): zip_file.writestr("test_plan_design.yaml", st.session_state.generated_artifacts["yaml"])
        st.download_button(label="📥 Download Test Package (.zip)", data=zip_buffer.getvalue(), file_name="jmeter_ai_test_package.zip", mime="application/zip")

if __name__ == "__main__":
    main()