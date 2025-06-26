import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import io
import zipfile
from copy import deepcopy
from urllib.parse import quote_plus, unquote
import re

from utils.jmeter_generator import JMeterScriptGenerator
from utils.swagger_parser import SwaggerParser
from utils.data_mapper import DataMapper

st.set_page_config(page_title="Intelligent JMeter Test Builder", page_icon="🤖", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_session_state():
    defaults = {
        'all_endpoints': [], 'scenario_steps': [], 'ai_mappings': {},
        'validation_warnings': [], 'generated_artifacts': None,
        'db_schema': None, 'db_sampled_data': None,
        'assembled_scenario': None, 'csv_files_preview': {},
        'correlation_suggestions': [], 'applied_correlations': {},
        'assertions_applied': False
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# All helper and core logic functions are stable and correct...
def find_case_insensitive_key(data_dict, target_key):
    if not target_key or not data_dict: return None
    for key in data_dict.keys():
        if key.lower() == target_key.lower(): return key
    return None

def find_case_insensitive_column_in_schema(schema, target_column):
    if not target_column or not schema: return None
    for col_def in schema:
        if 'name' in col_def and col_def['name'].lower() == target_column.lower(): return col_def['name']
    return None

def get_value_from_heuristics(param_name, param_type=None):
    name = param_name.lower().replace('_', '')
    if any(k in name for k in ['uuid', 'guid']) or name.endswith('id'): return "${__UUID()}"
    if 'email' in name: return "test_${__Random(1000,9999)}@example.com"
    if 'password' in name: return "Admin@123!"
    return f"auto_value_{param_name}"

def get_validated_intelligent_value(param, mapping_dict, db_schema, db_data, table_to_columns, warnings_list):
    param_name = param['name']
    ai_map = mapping_dict.get(param_name.lower(), {})
    suggested_table = ai_map.get("mapped_table"); suggested_column = ai_map.get("mapped_column")
    if suggested_table and suggested_column and db_schema and db_data:
        actual_table_name_schema = find_case_insensitive_key(db_schema, suggested_table)
        if actual_table_name_schema:
            table_schema_def = db_schema[actual_table_name_schema]
            actual_column_name_schema = find_case_insensitive_column_in_schema(table_schema_def, suggested_column)
            if actual_column_name_schema:
                actual_table_name_data = find_case_insensitive_key(db_data, actual_table_name_schema)
                if actual_table_name_data and actual_column_name_schema in db_data[actual_table_name_data].columns:
                    jmeter_variable_name = f"csv_{actual_table_name_data}_{actual_column_name_schema}"
                    value = f"${{{jmeter_variable_name}}}"
                    if actual_table_name_data not in table_to_columns: table_to_columns[actual_table_name_data] = set()
                    table_to_columns[actual_table_name_data].add(actual_column_name_schema)
                    return value
    return get_value_from_heuristics(param_name, param.get('type'))

def build_body_from_schema(schema, mapping_dict, db_schema, db_data, table_to_columns, warnings_list):
    if not schema or 'properties' not in schema: return None
    body = {}
    for prop_name, details in schema['properties'].items():
        mock_param = {'name': prop_name, 'type': details.get('type')}
        if details.get('type') == 'object':
            body[prop_name] = build_body_from_schema(details, mapping_dict, db_schema, db_data, table_to_columns, warnings_list)
        else:
            body[prop_name] = get_validated_intelligent_value(mock_param, mapping_dict, db_schema, db_data, table_to_columns, warnings_list)
    return body

def get_unique_parameters_for_scenario(scenario_steps):
    unique_params = {}
    def extract_from_schema(schema, prefix=''):
        if not schema or 'properties' not in schema: return
        for prop_name, details in schema.get('properties', {}).items():
            full_name = f"{prefix}{prop_name}"; full_name_lower = full_name.lower()
            if full_name_lower not in unique_params: unique_params[full_name_lower] = {'name': full_name, 'type': details.get('type')}
            if details.get('type') == 'object': extract_from_schema(details, prefix=f"{full_name}.")
    for endpoint in scenario_steps:
        for param in endpoint.get('parameters', []):
            param_name_lower = param['name'].lower()
            if param_name_lower not in unique_params: unique_params[param_name_lower] = {'name': param['name'], 'type': param.get('schema', {}).get('type')}
        if endpoint.get('body_schema'): extract_from_schema(endpoint['body_schema'])
    return list(unique_params.values())

def generate_correlation_suggestions():
    suggestions = []
    steps = st.session_state.scenario_steps
    for i, step in enumerate(steps):
        if step['method'] in ['POST', 'PUT'] and i + 1 < len(steps):
            next_step = steps[i+1]; path_params = re.findall(r'\{(\w+)\}', next_step['path'])
            for param in path_params:
                if 'id' in param.lower() or 'uid' in param.lower() or 'cd' in param.lower():
                    suggestions.append({"from_step": i, "to_step": i + 1, "variable_name": f"extracted_{param}", "json_path": f"$.{param}", "description": f"Extract **`{param}`** from **Step {i+1}**'s response to use in **Step {i+2}**'s URL."})
    st.session_state.correlation_suggestions = suggestions

def build_intelligent_scenario():
    endpoints = deepcopy(st.session_state.all_endpoints)
    method_order = {"POST": 0, "GET": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}
    sorted_endpoints = sorted(endpoints, key=lambda ep: (method_order.get(ep['method'], 99), ep['path'].count('/'), '{' in ep['path'], ep['path']))
    st.session_state.scenario_steps = sorted_endpoints
    generate_correlation_suggestions()
    st.session_state.ai_mappings = {}; st.session_state.assembled_scenario = None; st.session_state.assertions_applied = False

def main():
    initialize_session_state()
    st.title("🤖 Intelligent JMeter Test Builder")
    
    with st.sidebar:
        st.header("⚙️ Configuration"); uploaded_swagger_file = st.file_uploader("1. Swagger/OpenAPI Spec", type="json"); uploaded_schema_file = st.file_uploader("2. Database Schema", type="json"); uploaded_data_file = st.file_uploader("3. Sampled DB Data", type="json"); api_key = st.text_input("4. Groq API Key", type="password")
        if uploaded_swagger_file and not st.session_state.all_endpoints:
            with st.spinner("Parsing Swagger..."): st.session_state.all_endpoints = [ep.to_dict() for ep in SwaggerParser(json.load(uploaded_swagger_file)).parse()]
        if uploaded_schema_file and not st.session_state.db_schema: st.session_state.db_schema = json.load(uploaded_schema_file)
        if uploaded_data_file and not st.session_state.db_sampled_data: st.session_state.db_sampled_data = {t: pd.DataFrame(d) for t, d in json.load(uploaded_data_file).items()}
        st.divider(); st.header("Load Parameters"); target_host = st.text_input("Target Host", "sandbox-auth.livecareer.com"); num_users = st.number_input("Users", 1, value=10); ramp_up_time = st.number_input("Ramp-up (s)", 1, value=10); loop_count = st.number_input("Loops", -1, value=1); think_time = st.slider("Think Time (ms)", 0, 5000, 1000, 100)

    st.header("Step 1: Design the Test Scenario")
    build_mode = st.radio("Build Mode:", ["🤖 Auto-Build", "👨‍💻 Manual Build"], horizontal=True, label_visibility="collapsed")
    if build_mode == "🤖 Auto-Build":
        if st.button("🚀 Auto-Build Intelligent Scenario", disabled=not st.session_state.all_endpoints, type="primary"):
            build_intelligent_scenario(); st.rerun()
    else:
        with st.container(border=True, height=400):
            if st.session_state.all_endpoints:
                modules = {}
                for ep in st.session_state.all_endpoints:
                    try: module_name = ep['path'].split('/')[3] if ep['path'].count('/') > 2 else "general"
                    except IndexError: module_name = "general"
                    modules.setdefault(module_name.capitalize(), []).append(ep)
                scenario_endpoint_ids = {f"{e['method']}_{e['path']}" for e in st.session_state.scenario_steps}
                for module, endpoints in sorted(modules.items()):
                    with st.expander(f"🗂️ {module} ({len(endpoints)} endpoints)", expanded=True):
                        for endpoint in endpoints:
                            endpoint_id = f"{endpoint['method']}_{endpoint['path']}"
                            if endpoint_id not in scenario_endpoint_ids:
                                if st.button(f"➕ `{endpoint['method']}` {endpoint['path']}", key=f"add_{endpoint_id}", use_container_width=True):
                                    st.session_state.scenario_steps.append(deepcopy(endpoint)); generate_correlation_suggestions(); st.rerun()
    st.subheader("Current Scenario Steps")
    if not st.session_state.scenario_steps: st.info("Your scenario is empty. Use a build mode above to add steps.")
    else:
        for i, step in enumerate(st.session_state.scenario_steps):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([0.8, 0.05, 0.05, 0.1])
                method_color = {"GET": "blue", "POST": "green", "PUT": "orange", "DELETE": "red"}.get(step['method'], "grey")
                success_codes_str = ", ".join(step.get('success_codes', ['200']))
                c1.markdown(f"**Step {i+1}:** :{method_color}[`{step['method']}`] `{step['path']}` -- *Expects: `[{success_codes_str}]`*")
                if c2.button("⬆️", key=f"up_{i}", disabled=(i==0)): st.session_state.scenario_steps.insert(i-1, st.session_state.scenario_steps.pop(i)); generate_correlation_suggestions(); st.rerun()
                if c3.button("⬇️", key=f"down_{i}", disabled=(i==len(st.session_state.scenario_steps)-1)): st.session_state.scenario_steps.insert(i+1, st.session_state.scenario_steps.pop(i)); generate_correlation_suggestions(); st.rerun()
                if c4.button("🗑️", key=f"remove_{i}"): st.session_state.scenario_steps.pop(i); generate_correlation_suggestions(); st.rerun()

    st.divider()

    st.header("Step 2: Configure Logic & Assemble")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation")
        if st.session_state.scenario_steps:
            if st.session_state.correlation_suggestions:
                if st.button("✅ Apply All Correlations"):
                    for s in st.session_state.correlation_suggestions: st.session_state.applied_correlations[f"{s['from_step']}_{s['to_step']}_{s['variable_name']}"] = {**s}
                    st.rerun()
                # *** FIX: Restore individual correlation forms ***
                for i, s in enumerate(st.session_state.correlation_suggestions):
                    corr_id = f"{s['from_step']}_{s['to_step']}_{s['variable_name']}"
                    if corr_id in st.session_state.applied_correlations: st.success(f"✓ {s['description']}")
                    else:
                        with st.form(key=f"corr_form_{i}"):
                            st.markdown(s['description'])
                            sc1, sc2, sc3 = st.columns([2,2,1]); json_path = sc1.text_input("JSON Path", value=s['json_path'], key=f"jsp_{i}", label_visibility="collapsed"); var_name = sc2.text_input("Variable Name", value=s['variable_name'], key=f"vn_{i}", label_visibility="collapsed")
                            if sc3.form_submit_button("Apply"):
                                st.session_state.applied_correlations[corr_id] = {**s, "json_path": json_path, "variable_name": var_name}; st.rerun()
            else: st.info("No obvious correlations found.")
    with c2:
        st.subheader("Assertions")
        if st.session_state.scenario_steps:
            if st.session_state.assertions_applied: st.success("✓ Spec-driven assertions will be added.")
            else:
                if st.button("🎯 Apply Auto-Assertions"): st.session_state.assertions_applied = True; st.rerun()
    
    is_assemble_disabled = not st.session_state.scenario_steps or not st.session_state.db_schema or not api_key
    if st.button("Map Data & Assemble Scenario", disabled=is_assemble_disabled):
        with st.spinner("Mapping data and assembling..."):
            params_to_map = get_unique_parameters_for_scenario(st.session_state.scenario_steps)
            if params_to_map:
                ai_response = DataMapper.get_ai_powered_mappings(params_to_map, st.session_state.db_schema, api_key)
                st.session_state.ai_mappings = {item["parameter_name"].lower(): {**item} for item in ai_response.get("parameter_mappings", [])}
            st.session_state.validation_warnings.clear(); final_table_to_columns_map = {}; assembled_steps = []
            for i, step in enumerate(st.session_state.scenario_steps):
                assembled_step = deepcopy(step); assembled_step['extractions'] = []; assembled_step['assertions'] = []
                if st.session_state.assertions_applied: assembled_step['assertions'].append({"type": "ResponseAssertion", "codes": step.get('success_codes', ['200'])})
                for corr in st.session_state.applied_correlations.values():
                    if corr['from_step'] == i: assembled_step['extractions'].append({"type": "JSONExtractor", "refname": corr['variable_name'], "jsonpath": corr['json_path']})
                query_params, path_placeholders = [], {}
                for param in assembled_step.get('parameters', []):
                    correlated_value = next((f"${{{corr['variable_name']}}}" for corr in st.session_state.applied_correlations.values() if corr['to_step'] == i and param['name'].lower() in corr['variable_name'].lower()), None)
                    value = correlated_value or get_validated_intelligent_value(param, st.session_state.ai_mappings, st.session_state.db_schema, st.session_state.db_sampled_data, final_table_to_columns_map, st.session_state.validation_warnings)
                    if param.get('in') == 'path': path_placeholders[param['name']] = value
                    elif param.get('in') == 'query': query_params.append(f"{quote_plus(param['name'])}={quote_plus(value)}")
                temp_path = assembled_step['path']
                for p_name, p_val in path_placeholders.items(): temp_path = temp_path.replace(f"{{{p_name}}}", unquote(p_val))
                assembled_step['path'] = f"{temp_path}?{'&'.join(query_params)}" if query_params else temp_path
                if assembled_step.get('body_schema'):
                    assembled_step['body'] = build_body_from_schema(assembled_step['body_schema'], st.session_state.ai_mappings, st.session_state.db_schema, st.session_state.db_sampled_data, final_table_to_columns_map, st.session_state.validation_warnings)
                assembled_step['name'] = f"{step['method']} - {step['path']}"; assembled_step['headers'] = {"Content-Type": "application/json"}
                assembled_steps.append(assembled_step)
            st.session_state.assembled_scenario = {"Test Scenario": assembled_steps}
            st.session_state.csv_files_preview.clear()
            if st.session_state.db_sampled_data:
                for table, columns in final_table_to_columns_map.items():
                    df_table = st.session_state.db_sampled_data[table]
                    preview_df = pd.DataFrame()
                    for col in sorted(list(columns)): preview_df[f"csv_{table}_{col}"] = df_table[col]
                    st.session_state.csv_files_preview[f"{table}_data.csv"] = preview_df
            st.success("Assembly complete!")

    # *** FIX: Restore the preview sections ***
    if st.session_state.ai_mappings:
        with st.expander("View AI Data Mappings"):
            st.dataframe(pd.DataFrame([
                {"API Parameter": v.get("parameter_name"), "Mapped Table": v.get('mapped_table'), "Mapped Column": v.get('mapped_column')}
                for k, v in st.session_state.ai_mappings.items()
            ]), use_container_width=True)
            
    if st.session_state.csv_files_preview:
        with st.expander("📊 Preview of Generated CSV Files"):
            for filename, df in st.session_state.csv_files_preview.items():
                st.markdown(f"**`{filename}`**"); st.dataframe(df.head(), use_container_width=True)

    if st.session_state.assembled_scenario:
        with st.expander("🔬 View Final Assembled Test Case (JSON)"):
            st.json(st.session_state.assembled_scenario)

    st.divider()

    st.header("Step 4: Generate JMeter Package")
    is_generate_disabled = not st.session_state.assembled_scenario
    if st.button("🚀 Generate Full Test Plan", type="primary", disabled=is_generate_disabled, use_container_width=True):
        with st.spinner("Generating JMX, CSVs, and other artifacts..."):
            csv_files_for_zip = {filename: df.to_csv(index=False) for filename, df in st.session_state.csv_files_preview.items()}
            jmx_content, _, json_design, yaml_design = JMeterScriptGenerator.generate_jmx_and_artifacts(
                designed_scenarios=st.session_state.assembled_scenario, db_sampled_data=st.session_state.db_sampled_data,
                test_plan_name=f"{target_host} Load Test", thread_group_name=f"{num_users} Users",
                target_host=target_host, num_users=num_users, ramp_up_time=ramp_up_time, loop_count=loop_count, think_time=think_time
            )
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("test_plan.jmx", jmx_content); zf.writestr("design.json", json_design); zf.writestr("design.yaml", yaml_design)
                for filename, content in csv_files_for_zip.items(): zf.writestr(f"data/{filename}", content)
            st.session_state.generated_artifacts = zip_buffer.getvalue()
            st.success("Test plan package generated successfully!")

    if st.session_state.generated_artifacts:
        st.download_button("⬇️ Download JMeter Package (.zip)", st.session_state.generated_artifacts, "jmeter_test_package.zip", "application/zip", use_container_width=True)

if __name__ == "__main__":
    main()