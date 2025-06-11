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
        # Align jmeter version with the user's working JMX for better compatibility
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0", jmeter="5.2.1")
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
        # Call add_http_request_defaults here, which will add the ConfigTestElement
        # and its sibling hashTree to self.elements
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

        # All properties and elementProps should be direct children of the ConfigTestElement itself
        # This matches the structure seen in the user's working JMX file.

        # 1. Arguments Element (elementProp)
        arguments_prop = self._create_element(config, "elementProp", {
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(arguments_prop, "Arguments.arguments")  # Empty collection for arguments

        # 2. String Properties (direct children of config)
        self._create_string_prop(config, "HTTPSampler.protocol", protocol)
        self._create_string_prop(config, "HTTPSampler.domain", domain)
        self._create_string_prop(config, "HTTPSampler.port", str(port))
        self._create_string_prop(config, "HTTPSampler.contentEncoding", "")  # Added as per working JMX
        self._create_string_prop(config, "HTTPSampler.proxyHost", "")  # Added as per working JMX
        self._create_string_prop(config, "HTTPSampler.proxyPort", "")  # Added as per working JMX
        self._create_string_prop(config, "HTTPSampler.proxyUser", "")  # Added as per working JMX
        self._create_string_prop(config, "HTTPSampler.proxyPass", "")  # Added as per working JMX
        self._create_string_prop(config, "HTTPSampler.connect_timeout", "")
        self._create_string_prop(config, "HTTPSampler.response_timeout", "")

        # 3. Boolean Properties (direct children of config)
        # Order within string/bool props is less critical but following working JMX
        self._create_bool_prop(config, "HTTPSampler.send_chunked_post_body", False)  # Corrected to boolProp
        self._create_bool_prop(config, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(config, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(config, "HTTPSampler.use_keepalive", True)
        self._create_bool_prop(config, "HTTPSampler.DO_MULTIPART_POST", False)
        self._create_bool_prop(config, "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART", True)
        self._create_bool_prop(config, "HTTPSampler.concurrentDwn", False)  # Common JMeter default (if needed)

        # Crucial: An empty hashTree as a SIBLING to ConfigTestElement, not a child
        # This hashTree marks the end of the ConfigTestElement's scope/children
        self._create_element(parent, "hashTree")

    def add_thread_group(self, num_users, ramp_up_time, loop_count, parent_element):
        thread_group = self._create_element(parent_element, "ThreadGroup", {
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": "Users",
            "enabled": "true"
        })
        self._create_string_prop(thread_group, "ThreadGroup.on_sample_error", "continue")  # Direct string prop

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
                         headers=None) -> ET.Element:  # Added return type hint for sampler_hash_tree
        sampler = self._create_element(parent_element, "HTTPSamplerProxy", {
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })
        # This is the hashTree that contains children elements scoped to this sampler
        sampler_hash_tree = self._create_element(sampler, "hashTree")

        # Sampler properties (direct children of sampler)
        self._create_string_prop(sampler, "HTTPSampler.method", method)
        self._create_string_prop(sampler, "HTTPSampler.path", path)
        self._create_bool_prop(sampler, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(sampler, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(sampler, "HTTPSampler.use_keepalive", True)
        self._create_string_prop(sampler, "HTTPSampler.embedded_url_দোষ", "")
        self._create_string_prop(sampler, "HTTPSampler.connect_timeout", "")
        self._create_string_prop(sampler, "HTTPSampler.response_timeout", "")

        # Arguments for parameters (URL parameters or form data)
        arguments = self._create_element(sampler, "elementProp", {  # Child of sampler
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
            post_data = self._create_element(sampler, "boolProp",
                                             {"name": "HTTPSampler.postBodyRaw"})  # Child of sampler
            post_data.text = "true"

            body_data = self._create_element(sampler, "elementProp", {  # Child of sampler
                "name": "HTTPsampler.Arguments",  # This name is also used for raw body data
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
            header_manager = self._create_element(sampler_hash_tree, "HeaderManager",
                                                  {  # HeaderManager is a child of sampler_hash_tree
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
            # FIX: REMOVED THE INCORRECT HASH TREE: self._create_element(header_manager, "hashTree")
            # The HeaderManager does *not* have an internal hashTree.
            # Its hashTree for scoping children (e.g., Assertions scoped to HeaderManager) is its own hashTree
            # which is *not* what the ClassCastException was about. The error was due to the extra hashTree inside HeaderManager.

        return sampler_hash_tree  # Return the hash_tree so other elements can be added to it

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
            if parameters_from_llm is None:
                parameters_from_llm = {}
                logger.debug(f"Parameters for {endpoint_key} were None, defaulting to empty dict.")

            if isinstance(parameters_from_llm, str):
                try:
                    parameters_from_llm = json.loads(parameters_from_llm)
                    if parameters_from_llm is None:
                        parameters_from_llm = {}
                    logger.debug(f"Parsed parameters string for {endpoint_key}: {parameters_from_llm}")
                except json.JSONDecodeError:
                    logger.error(
                        f"LLM generated invalid JSON string for parameters: {parameters_from_llm}. Skipping parameters for this request.")
                    parameters_from_llm = {}

            # Loop through parameters to check for mapped data
            for param_name, param_value in parameters_from_llm.items():
                if param_value.startswith("${") and param_value.endswith("}"):
                    # This is a JMeter variable, likely from a mapping
                    jmeter_var_name = param_value[2:-1]  # Remove ${ and }
                    original_mapping_format = jmeter_var_name.replace('_', '.')

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
                    df = database_connector.preview_data(table_name, limit=None)  # Fetch all data
                    if not df.empty and column_name in df.columns:
                        column_values = df[column_name].tolist()
                        csv_data[col_identifier] = column_values
                        logger.debug(f"Collected {len(column_values)} values for {col_identifier}")
                    else:
                        logger.warning(
                            f"Column '{column_name}' not found or table '{table_name}' is empty for CSV data.")
                except Exception as e:
                    logger.error(f"Error fetching data for CSV for {col_identifier}: {e}")

            if csv_data:
                max_rows = 0
                if csv_data:
                    max_rows = max(len(v) for v in csv_data.values()) if csv_data else 0

                csv_header = []
                csv_rows = []

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
            headers = request_data.get('headers', {})

            parameters_from_llm = request_data.get('parameters', {})
            if parameters_from_llm is None:
                parameters_from_llm = {}
                logger.debug(f"Parameters for {endpoint_key} were None, defaulting to empty dict.")

            if isinstance(parameters_from_llm, str):
                try:
                    parameters_from_llm = json.loads(parameters_from_llm)
                    if parameters_from_llm is None:
                        parameters_from_llm = {}
                    logger.debug(f"Parsed parameters string for {endpoint_key}: {parameters_from_llm}")
                except json.JSONDecodeError:
                    logger.error(
                        f"LLM generated invalid JSON string for parameters: {parameters_from_llm}. Skipping parameters for this request.")
                    parameters_from_llm = {}

            processed_parameters = {}
            for param_name, param_value in parameters_from_llm.items():
                if param_value.startswith("${") and param_value.endswith("}"):
                    processed_parameters[param_name] = param_value.replace('.', '_')
                else:
                    processed_parameters[param_name] = param_value

            # Call add_http_sampler and get its internal hashTree
            current_sampler_hash_tree = self.add_http_sampler(thread_group_elements, name, method, path,
                                                              processed_parameters, body, headers)

            # Now, add assertions and extractors as children of current_sampler_hash_tree
            self.add_response_assertion(current_sampler_hash_tree, f"{name} - Status 200", "Response Code", "Equals",
                                        "200")

            if method == "POST" and body and isinstance(body, str) and "id" in body:
                logger.debug(f"Adding JSON Extractor for {name} to capture 'id'.")
                self.add_json_extractor(current_sampler_hash_tree, f"{name} - Extract ID", "$.id",
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
        # This placeholder logic is not currently used by generate_jmx, but kept for completeness
        jmx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.2.1">
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
          <boolProp name="HTTPSampler.send_chunked_post_body">false</boolProp>
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
            <collectionProp name="Arguments.arguments"/>
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
