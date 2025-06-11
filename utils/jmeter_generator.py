import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import random
import string
import json  # Import json module for parsing LLM parameter strings

# Import SwaggerEndpoint as it's used in type hints for generate_jmx
from utils.swagger_parser import SwaggerEndpoint

logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    def __init__(self):
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="2.9", jmeter="5.5")
        hash_tree = ET.SubElement(self.test_plan_root, "hashTree")

        self.test_plan = self._create_element(hash_tree, "TestPlan", attrib={
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": "Test Plan",
            "enabled": "true"
        })
        test_plan_hash_tree = self._create_element(self.test_plan, "hashTree")
        self.elements = test_plan_hash_tree  # This will be the main container for thread groups and other elements

        # Default HTTP Request Defaults
        self.add_http_request_defaults(self.elements)

        # CSV Data Set Config placeholder, added if needed later
        self.csv_config = None

    def _create_element(self, parent, tag, attrib=None):
        if attrib is None:
            attrib = {}
        return ET.SubElement(parent, tag, attrib)

    def _create_collection_prop(self, parent, name):
        prop = self._create_element(parent, "collectionProp", {"name": name})
        return prop

    def _create_string_prop(self, parent, name, value):
        prop = self._create_element(parent, "stringProp", {"name": name})
        prop.text = str(value)
        return prop

    def _create_bool_prop(self, parent, name, value):
        prop = self._create_element(parent, "boolProp", {"name": name})
        prop.text = "true" if value else "false"
        return prop

    def add_http_request_defaults(self, parent, protocol="https", domain="petstore.swagger.io", port=""):
        config = self._create_element(parent, "ConfigTestElement", {
            "guiclass": "HttpDefaultsGui",
            "testclass": "HttpDefaults",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        })

        # Correct structure: ConfigTestElement has ONE immediate hashTree child
        config_hash_tree = self._create_element(config, "hashTree")

        # All properties and elementProps should be children of this hashTree
        # 1. Arguments Element (elementProp)
        arguments_prop = self._create_element(config_hash_tree, "elementProp", {  # Corrected parent to config_hash_tree
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(arguments_prop, "Arguments.arguments")  # Empty collection for defaults

        # 2. String Properties
        self._create_string_prop(config_hash_tree, "HTTPSampler.protocol", protocol)  # Corrected parent
        self._create_string_prop(config_hash_tree, "HTTPSampler.domain", domain)  # Corrected parent
        self._create_string_prop(config_hash_tree, "HTTPSampler.port", str(port))  # Corrected parent
        self._create_string_prop(config_hash_tree, "HTTPSampler.connect_timeout", "")  # Corrected parent
        self._create_string_prop(config_hash_tree, "HTTPSampler.response_timeout", "")  # Corrected parent

        # 3. Boolean Properties
        self._create_bool_prop(config_hash_tree, "HTTPSampler.follow_redirects", True)  # Corrected parent
        self._create_bool_prop(config_hash_tree, "HTTPSampler.auto_redirects", False)  # Corrected parent
        self._create_bool_prop(config_hash_tree, "HTTPSampler.use_keepalive", True)  # Corrected parent
        self._create_bool_prop(config_hash_tree, "HTTPSampler.DO_MULTIPART_POST", False)  # Corrected parent
        self._create_bool_prop(config_hash_tree, "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART", True)  # Corrected parent
        self._create_bool_prop(config_hash_tree, "HTTPSampler.concurrentDwn", False)  # Corrected parent

    def add_thread_group(self, num_users, ramp_up_time, loop_count, parent_element):
        thread_group = self._create_element(parent_element, "ThreadGroup", {
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": "Users",
            "enabled": "true"
        })
        self_controller = self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"})
        self_controller.text = "continue"

        self._create_string_prop(thread_group, "ThreadGroup.num_threads", str(num_users))
        self._create_string_prop(thread_group, "ThreadGroup.ramp_time", str(ramp_up_time))
        self._create_bool_prop(thread_group, "ThreadGroup.scheduler", False)
        self._create_string_prop(thread_group, "ThreadGroup.duration", "")
        self._create_string_prop(thread_group, "ThreadGroup.delay", "")
        self._create_bool_prop(thread_group, "ThreadGroup.same_user_on_next_iteration", True)

        # Check if loop_count is -1 for infinite or a positive integer
        if loop_count == -1:
            main_controller = self._create_element(thread_group, "elementProp", {
                "name": "ThreadGroup.main_controller",
                "elementType": "LoopController",
                "guiclass": "LoopControlPanel",
                "testclass": "LoopController",
                "testname": "Loop Controller",
                "enabled": "true"
            })
            self._create_bool_prop(main_controller, "LoopController.continue_forever", True)
            self._create_string_prop(main_controller, "LoopController.loops", "-1")
        else:
            main_controller = self._create_element(thread_group, "elementProp", {
                "name": "ThreadGroup.main_controller",
                "elementType": "LoopController",
                "guiclass": "LoopControlPanel",
                "testclass": "LoopController",
                "testname": "Loop Controller",
                "enabled": "true"
            })
            self._create_bool_prop(main_controller, "LoopController.continue_forever", False)
            self._create_string_prop(main_controller, "LoopController.loops", str(loop_count))

        thread_group_hash_tree = self._create_element(thread_group, "hashTree")
        return thread_group_hash_tree

    def add_http_sampler(self, parent_element, name, method, path, parameters=None, body=None,
                         headers=None):  # Added headers parameter
        sampler = self._create_element(parent_element, "HTTPSamplerProxy", {
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })
        sampler_hash_tree = self._create_element(sampler, "hashTree")

        self._create_string_prop(sampler, "HTTPSampler.method", method)
        self._create_string_prop(sampler, "HTTPSampler.path", path)
        self._create_bool_prop(sampler, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(sampler, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(sampler, "HTTPSampler.use_keepalive", True)
        self._create_string_prop(sampler, "HTTPSampler.embedded_url_দোষ", "")
        self._create_string_prop(sampler, "HTTPSampler.connect_timeout", "")
        self._create_string_prop(sampler, "HTTPSampler.response_timeout", "")

        # Arguments for parameters (URL parameters or form data)
        arguments = self._create_element(sampler, "elementProp", {
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        collection_prop = self._create_collection_prop(arguments, "Arguments.arguments")

        if parameters:
            for param_name, param_value in parameters.items():
                arg_prop = self._create_element(collection_prop, "elementProp", {
                    "name": param_name,
                    "elementType": "HTTPArgument"
                })
                self._create_bool_prop(arg_prop, "HTTPArgument.always_encode", False)
                self._create_string_prop(arg_prop, "Argument.value", str(param_value))
                self._create_string_prop(arg_prop, "Argument.metadata", "=")
                self._create_string_prop(arg_prop, "Argument.name", param_name)

        # For POST/PUT requests with JSON bodies
        if method in ["POST", "PUT", "PATCH"] and body:
            post_data = self._create_element(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"})
            post_data.text = "true"

            body_data = self._create_element(sampler, "elementProp", {
                "name": "HTTPsampler.Arguments",
                "elementType": "Arguments",
                "guiclass": "HTTPArgumentsPanel",
                "testclass": "Arguments",
                "enabled": "true"
            })
            body_data_collection_prop = self._create_collection_prop(body_data, "Arguments.arguments")

            http_argument = self._create_element(body_data_collection_prop, "elementProp", {
                "name": "",
                "elementType": "HTTPArgument"
            })
            self._create_bool_prop(http_argument, "HTTPArgument.always_encode", False)
            self._create_string_prop(http_argument, "Argument.value", body)
            self._create_string_prop(http_argument, "Argument.metadata", "=")  # For raw body, metadata is just "="

        # Add Header Manager if headers are provided
        if headers:
            header_manager = self._create_element(sampler_hash_tree, "HeaderManager", {
                "guiclass": "HeaderPanel",
                "testclass": "HeaderManager",
                "testname": "HTTP Header Manager",
                "enabled": "true"
            })
            header_collection_prop = self._create_collection_prop(header_manager, "HeaderManager.headers")
            for header_name, header_value in headers.items():
                header_element = self._create_element(header_collection_prop, "elementProp", {
                    "name": "",
                    "elementType": "Header"
                })
                self._create_string_prop(header_element, "Header.name", header_name)
                self._create_string_prop(header_element, "Header.value", header_value)
            self._create_element(header_manager, "hashTree")  # Empty hash tree for HeaderManager

    def add_csv_data_config(self, parent_element, filename, variable_names, delimiter=",", quoted_data=False):
        csv_config_element = self._create_element(parent_element, "CSVDataSet", {
            "guiclass": "TestBeanGUI",
            "testclass": "CSVDataSet",
            "testname": "CSV Data Set Config",
            "enabled": "true"
        })
        self._create_string_prop(csv_config_element, "filename", filename)
        self._create_string_prop(csv_config_element, "variableNames", variable_names)
        self._create_string_prop(csv_config_element, "delimiter", delimiter)
        self._create_bool_prop(csv_config_element, "ignoreFirstLine", True)
        self._create_bool_prop(csv_config_element, "quotedData", quoted_data)
        self._create_bool_prop(csv_config_element, "recycle", True)
        self._create_bool_prop(csv_config_element, "stopThread", False)
        self._create_string_prop(csv_config_element, "shareMode", "shareMode.all")

        self._create_element(csv_config_element, "hashTree")  # Empty hash tree for CSV config
        self.csv_config = csv_config_element  # Store reference to add if needed

    def add_response_assertion(self, parent_element, name, response_field, test_type, pattern):
        assertion = self._create_element(parent_element, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": name,
            "enabled": "true"
        })
        assertion_hash_tree = self._create_element(assertion, "hashTree")

        self._create_collection_prop(assertion, "Asserion.test_strings")
        self._create_string_prop(assertion_hash_tree, "Assertion.custom_message", "")
        self._create_string_prop(assertion_hash_tree, "Assertion.test_field",
                                 response_field)  # "Response Data", "Response Code", etc.
        self._create_string_prop(assertion_hash_tree, "Assertion.test_type",
                                 test_type)  # "Contains", "Matches", "Equals", "Substring"
        self._create_string_prop(assertion_hash_tree, "Assertion.assume_success", "false")
        self._create_collection_prop(assertion_hash_tree, "Asserion.test_strings").text = pattern  # pattern to match

    def add_json_extractor(self, parent_element, name, json_path_expr, var_name, match_no="1",
                           default_value="NOT_FOUND"):
        extractor = self._create_element(parent_element, "JSONPostProcessor", {
            "guiclass": "JSONPostProcessorGui",
            "testclass": "JSONPostProcessor",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(extractor, "JSONPostProcessor.jsonPathExpr", json_path_expr)
        self._create_string_prop(extractor, "JSONPostProcessor.referenceNames", var_name)
        self._create_string_prop(extractor, "JSONPostProcessor.matchNumbers", match_no)
        self._create_string_prop(extractor, "JSONPostProcessor.defaultValues", default_value)
        self._create_string_prop(extractor, "JSONPostProcessor.scope", "body")  # Default scope to body

        self._create_element(extractor, "hashTree")  # Empty hash tree for extractor

    def generate_jmx(self, swagger_endpoints: List[SwaggerEndpoint],
                     mappings: Dict[str, Dict[str, str]],
                     thread_group_users: int,
                     ramp_up_time: int,
                     loop_count: int,
                     scenario_plan: Dict[str, Any],
                     database_connector: Any,  # DatabaseConnector instance
                     db_tables_schema: Dict[str, List[Dict[str, str]]]) -> (str, Optional[str]):

        # Initialize thread group
        thread_group_elements = self.add_thread_group(
            num_users=scenario_plan['thread_group'].get('num_users', thread_group_users),
            ramp_up_time=scenario_plan['thread_group'].get('ramp_up', ramp_up_time),
            loop_count=scenario_plan['thread_group'].get('loop_count', loop_count),
            parent_element=self.elements
        )

        csv_content = None
        csv_data_needed = False
        csv_columns = set()

        # Collect all mapped parameters that will need CSV data
        mapped_parameters_to_collect = {}  # { "table_name.column_name": set of request_data }

        for request_data in scenario_plan['requests']:
            endpoint_key = request_data['endpoint_key']
            parameters_from_llm = request_data.get('parameters', {})
            # FIX: Ensure parameters_from_llm is a dictionary.
            # If it was parsed from 'null', it will be None. Convert to empty dict.
            if parameters_from_llm is None:
                parameters_from_llm = {}
                logger.debug(f"Parameters for {endpoint_key} were None, defaulting to empty dict.")

            # Check if LLM outputted parameters as a string and parse them
            if isinstance(parameters_from_llm, str):
                try:
                    parameters_from_llm = json.loads(parameters_from_llm)
                    if parameters_from_llm is None:  # If string was 'null'
                        parameters_from_llm = {}
                    logger.debug(f"Parsed parameters string for {endpoint_key}: {parameters_from_llm}")
                except json.JSONDecodeError:
                    logger.error(
                        f"LLM generated invalid JSON string for parameters: {parameters_from_llm}. Skipping parameters for this request.")
                    parameters_from_llm = {}  # Fallback to empty dict

            # Loop through parameters to check for mapped data
            for param_name, param_value in parameters_from_llm.items():
                if param_value.startswith("${") and param_value.endswith("}"):
                    # This is a JMeter variable, likely from a mapping
                    # Extract the actual mapping (e.g., categories_id from ${categories_id})
                    jmeter_var_name = param_value[2:-1]  # Remove ${ and }
                    # JMeter variable names might use underscores instead of dots from mappings
                    original_mapping_format = jmeter_var_name.replace('_', '.')

                    # Find the actual mapping in our stored mappings
                    # This is a bit indirect, but we need to find the table.column
                    found_mapping = None
                    for ep_key, ep_mappings in mappings.items():
                        for map_param, map_target in ep_mappings.items():
                            if map_target == original_mapping_format:
                                found_mapping = map_target
                                break
                        if found_mapping:
                            break

                    if found_mapping:
                        table_name, column_name = found_mapping.split('.')
                        csv_columns.add(f"{table_name}.{column_name}")
                        csv_data_needed = True
                        logger.debug(f"Identified mapped parameter needing CSV data: {found_mapping}")
                    else:
                        logger.warning(
                            f"JMeter variable '{jmeter_var_name}' found in LLM plan but no matching original mapping '{original_mapping_format}' in DataMapper. Skipping CSV collection for this param.")

        if csv_data_needed and database_connector:
            csv_data = {}
            for col_identifier in csv_columns:
                table_name, column_name = col_identifier.split('.')
                try:
                    # Fetch all data for the relevant column
                    df = database_connector.preview_data(table_name, limit=None)  # Fetch all data
                    if not df.empty and column_name in df.columns:
                        # Use list to store values, as iterrows can be inefficient for large datasets
                        column_values = df[column_name].tolist()
                        csv_data[col_identifier] = column_values
                        logger.debug(f"Collected {len(column_values)} values for {col_identifier}")
                    else:
                        logger.warning(
                            f"Column '{column_name}' not found or table '{table_name}' is empty for CSV data.")
                except Exception as e:
                    logger.error(f"Error fetching data for CSV for {col_identifier}: {e}")

            if csv_data:
                # Determine max rows
                max_rows = 0
                if csv_data:
                    max_rows = max(len(v) for v in csv_data.values()) if csv_data else 0

                # Create CSV content
                csv_header = []
                csv_rows = []

                # Sort columns for consistent CSV output
                sorted_csv_columns = sorted(list(csv_data.keys()))
                csv_header = [col.replace('.', '_') for col in
                              sorted_csv_columns]  # JMeter variable names use underscores

                for i in range(max_rows):
                    row_values = []
                    for col_identifier in sorted_csv_columns:
                        values = csv_data.get(col_identifier, [])
                        row_values.append(str(values[i]) if i < len(values) else "")  # Handle uneven column lengths
                    csv_rows.append(",".join(row_values))

                csv_content = ",".join(csv_header) + "\n" + "\n".join(csv_rows)
                logger.info(f"Generated CSV content with {max_rows} rows for columns: {csv_header}")

                # Add CSV Data Set Config to the Thread Group
                csv_variable_names = ",".join(csv_header)  # Use JMeter variable names (underscores)
                self.add_csv_data_config(thread_group_elements, "data.csv", csv_variable_names)
            else:
                logger.info("No data collected for CSV config despite mapped parameters. CSV config will not be added.")

        for request_data in scenario_plan['requests']:
            endpoint_key = request_data['endpoint_key']
            method = request_data['method']
            path = request_data['path']
            name = request_data['name']
            body = request_data.get('body')
            # Extract headers from request_data, default to empty dict
            headers = request_data.get('headers', {})

            parameters_from_llm = request_data.get('parameters', {})
            # FIX: Ensure parameters_from_llm is a dictionary.
            # If it was parsed from 'null', it will be None. Convert to empty dict.
            if parameters_from_llm is None:
                parameters_from_llm = {}
                logger.debug(f"Parameters for {endpoint_key} were None, defaulting to empty dict.")

            # If it's a string (from LLM output), attempt to parse it again here if not already parsed
            if isinstance(parameters_from_llm, str):
                try:
                    parameters_from_llm = json.loads(parameters_from_llm)
                    if parameters_from_llm is None:  # If string was 'null'
                        parameters_from_llm = {}
                    logger.debug(f"Parsed parameters string for {endpoint_key}: {parameters_from_llm}")
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to re-parse parameters string during sampler creation for {endpoint_key}. Using empty dict.")
                    parameters_from_llm = {}

            # Replace mapped parameters with JMeter variable format using underscores
            processed_parameters = {}
            for param_name, param_value in parameters_from_llm.items():
                if param_value.startswith("${") and param_value.endswith("}"):
                    # This is already a JMeter variable from LLM, ensure underscore format
                    processed_parameters[param_name] = param_value.replace('.', '_')
                else:
                    processed_parameters[param_name] = param_value

            # Add HTTP Sampler, passing the extracted headers
            self.add_http_sampler(thread_group_elements, name, method, path, processed_parameters, body, headers)

            # Add simple response assertion (e.g., check for 200 OK)
            self.add_response_assertion(thread_group_elements, f"{name} - Status 200", "Response Code", "Equals", "200")

            # Example: Add JSON Extractor if the response might contain an ID for chaining
            # This is a heuristic and would need more sophisticated LLM guidance for real-world scenarios
            if method == "POST" and body and isinstance(body, str) and "id" in body:  # more robust check
                logger.debug(f"Adding JSON Extractor for {name} to capture 'id'.")
                self.add_json_extractor(thread_group_elements, f"{name} - Extract ID", "$.id",
                                        f"{name.replace(' ', '_')}_id")

        # Generate XML string
        rough_string = ET.tostring(self.test_plan_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_jmx = reparsed.toprettyxml(indent="  ")

        # JMeter JMX files often have a specific XML declaration and doctype
        # We can add these if missing or ensure they are correct
        final_jmx_content = f'<?xml version="1.0" encoding="UTF-8"?>\n' \
                            f'<!DOCTYPE jmeterTestPlan SYSTEM "jmeter.apache.org/dtd/jmeter_2_3.dtd">\n' \
                            + pretty_jmx.split('?>\n', 1)[-1]  # Remove old XML declaration if exists

        return final_jmx_content, csv_content

    # Placeholder for JMX content generation (replace with actual JMeter XML generation)
    def _generate_placeholder_jmx(self, endpoints, mappings, thread_group_users, ramp_up_time, loop_count) -> str:
        jmx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.5">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Generated Test Plan" enabled="true">
      <stringProp name="TestPlan.comments"></stringProp>
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
      <elementProp name="TestPlan.user_defined_variables" elementType="Arguments" guiclass="ArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
        <collectionProp name="Arguments.arguments"/>
      </elementProp>
      <stringProp name="TestPlan.user_define_classpath"></stringProp>
    </TestPlan>
    <hashTree>
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Users" enabled="true">
        <stringProp name="ThreadGroup.num_threads">{thread_group_users}</stringProp>
        <stringProp name="ThreadGroup.ramp_time">{ramp_up_time}</stringProp>
        <boolProp name="ThreadGroup.scheduler">false</boolProp>
        <stringProp name="ThreadGroup.duration"></stringProp>
        <stringProp name="ThreadGroup.delay"></stringProp>
        <boolProp name="ThreadGroup.same_user_on_next_iteration">true</boolProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControlPanel" testclass="LoopController" testname="Loop Controller" enabled="true">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <stringProp name="LoopController.loops">{loop_count}</stringProp>
        </elementProp>
      </ThreadGroup>
      <hashTree>
        <!-- HTTP Request Defaults -->
        <ConfigTestElement guiclass="HttpDefaultsGui" testclass="HttpDefaults" testname="HTTP Request Defaults" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">petstore.swagger.io</stringProp>
          <stringProp name="HTTPSampler.protocol">https</stringProp>
          <stringProp name="HTTPSampler.contentEncoding"></stringProp>
          <stringProp name="HTTPSampler.port"></stringProp>
          <stringProp name="HTTPSampler.proxyHost"></stringProp>
          <stringProp name="HTTPSampler.proxyPort"></stringProp>
          <stringProp name="HTTPSampler.proxyUser"></stringProp>
          <stringProp name="HTTPSampler.proxyPass"></stringProp>
          <stringProp name="HTTPSampler.send_chunked_post_body">false</stringProp>
          <stringProp name="HTTPSampler.connection_timeout"></stringProp>
          <stringProp name="HTTPSampler.response_timeout"></stringProp>
        </ConfigTestElement>
        <hashTree/>
        <!-- Samplers based on Endpoints -->
"""
        for endpoint in endpoints:
            # Simple placeholder for each endpoint
            jmx_content += f"""
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="{endpoint.method} {endpoint.path}" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments">
"""
            # Add parameters from mappings or just placeholders
            endpoint_key = f"{endpoint.method} {endpoint.path}"
            if endpoint_key in mappings:
                for param, mapped_col in mappings[endpoint_key].items():
                    jmx_content += f"""
              <elementProp name="{param}" elementType="HTTPArgument">
                <boolProp name="HTTPArgument.always_encode">false</boolProp>
                <stringProp name="Argument.value">${{{mapped_col.replace('.', '_')}}}</stringProp>
                <stringProp name="Argument.metadata">=</stringProp>
                <stringProp name="Argument.name">{param}</stringProp>
              </elementProp>
"""
            else:
                for param in endpoint.parameters:
                    jmx_content += f"""
              <elementProp name="{param['name']}" elementType="HTTPArgument">
                <boolProp name="HTTPArgument.always_encode">false</boolProp>
                <stringProp name="Argument.value">dummy_value</stringProp>
                <stringProp name="Argument.metadata">=</stringProp>
                <stringProp name="Argument.name">{param['name']}</stringProp>
              </elementProp>
"""
            jmx_content += f"""
            </collectionProp>
          </elementProp>
          <stringProp name="HTTPSampler.method">{endpoint.method}</stringProp>
          <stringProp name="HTTPSampler.path">{endpoint.path}</stringProp>
          <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
          <boolProp name="HTTPSampler.auto_redirects">false</boolProp>
          <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
          <boolProp name="HTTPSampler.postBodyRaw">false</boolProp>
          <elementProp name="HTTPSampler.HeaderTemplate" elementType="HeaderPanel" guiclass="HeaderPanel" testclass="HeaderPanel" testname="Headers" enabled="true">
            <collectionProp name="Header.headers"/>
          </elementProp>
        </HTTPSamplerProxy>
        <hashTree/>
"""

        jmx_content += """
      </hashTree>
    </hashTree>
</jmeterTestPlan>
"""
        return jmx_content
