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
        # Keeping 5.2.1 as it's the version from the user's working JMX reference,
        # and more likely to be compatible with older XStream if that's the issue.
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0",
                                         jmeter="5.6.3")  # Updated JMeter version

        # This is the outermost hashTree element, a sibling to the TestPlan itself
        self.root_hash_tree_container = ET.SubElement(self.test_plan_root, "hashTree")

        # TestPlan element: Child of the root_hash_tree_container
        self.test_plan = self._create_element(self.root_hash_tree_container, "TestPlan", attrib={
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": "Web Application Performance Test",  # User specified name
            "enabled": "true"
        })
        self._create_string_prop(self.test_plan, "TestPlan.comments",
                                 "A test plan to evaluate the performance of a web application under load.")  # User specified
        self._create_bool_prop(self.test_plan, "TestPlan.functional_mode", False)  # User specified
        self._create_bool_prop(self.test_plan, "TestPlan.tearDown_on_shutdown", True)  # User specified
        self._create_bool_prop(self.test_plan, "TestPlan.serialize_threadgroups", False)  # User specified

        # User Defined Variables for TestPlan (Exact element as specified)
        user_defined_variables = self._create_element(self.test_plan, "elementProp", {
            "name": "TestPlan.user_defined_variables",
            "elementType": "Arguments",
            "guiclass": "ArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(user_defined_variables, "Arguments.arguments")
        self._create_string_prop(self.test_plan, "TestPlan.user_define_classpath", "")

        # TestPlan's SIBLING hashTree - this will contain HTTP Defaults, Thread Groups, Listeners
        self.test_plan_children_hash_tree = ET.SubElement(self.root_hash_tree_container, "hashTree")

        # Default HTTP Request Defaults (add as child of test_plan_children_hash_tree)
        http_defaults = self.add_http_request_defaults(self.test_plan_children_hash_tree,
                                                       protocol="https",
                                                       domain="example.com",  # User specified domain
                                                       connect_timeout="5000",  # User specified 5s (in ms)
                                                       response_timeout="10000")  # User specified 10s (in ms)

        # SIBLING hashTree for HTTP Request Defaults
        self.http_defaults_children_hash_tree = ET.SubElement(self.test_plan_children_hash_tree, "hashTree")

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

    def add_http_request_defaults(self, parent, protocol="https", domain="petstore.swagger.io", port="",
                                  connect_timeout="", response_timeout=""):
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
        self._create_string_prop(config, "HTTPSampler.contentEncoding", "")  # User specified
        self._create_string_prop(config, "HTTPSampler.proxyHost", "")
        self._create_string_prop(config, "HTTPSampler.proxyPort", "")
        self._create_string_prop(config, "HTTPSampler.proxyUser", "")
        self._create_string_prop(config, "HTTPSampler.proxyPass", "")
        self._create_string_prop(config, "HTTPSampler.connect_timeout", connect_timeout)  # User specified
        self._create_string_prop(config, "HTTPSampler.response_timeout", response_timeout)  # User specified

        # 3. Boolean Properties (direct children of config)
        self._create_bool_prop(config, "HTTPSampler.send_chunked_post_body", False)
        self._create_bool_prop(config, "HTTPSampler.follow_redirects", True)  # User specified
        self._create_bool_prop(config, "HTTPSampler.auto_redirects", False)  # User specified
        self._create_bool_prop(config, "HTTPSampler.use_keepalive", True)  # User specified
        self._create_bool_prop(config, "HTTPSampler.DO_MULTIPART_POST", False)
        self._create_bool_prop(config, "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART", True)
        self._create_bool_prop(config, "HTTPSampler.concurrentDwn", False)

        # No internal hashTree for ConfigTestElement
        return config

    def add_thread_group(self, num_users, ramp_up_time, loop_count, parent_element):
        thread_group = self._create_element(parent_element, "ThreadGroup", {
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": "Main Users",  # User specified name
            "enabled": "true"
        })
        self._create_string_prop(thread_group, "ThreadGroup.on_sample_error", "continue")  # User specified

        self._create_string_prop(thread_group, "ThreadGroup.num_threads", str(num_users))  # User specified
        self._create_string_prop(thread_group, "ThreadGroup.ramp_time", str(ramp_up_time))  # User specified
        self._create_bool_prop(thread_group, "ThreadGroup.scheduler", False)
        self._create_string_prop(thread_group, "ThreadGroup.duration", "")
        self._create_string_prop(thread_group, "ThreadGroup.delay", "")
        self._create_bool_prop(thread_group, "ThreadGroup.same_user_on_next_iteration", True)  # User specified

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
            self._create_string_prop(main_controller, "LoopController.loops", str(loop_count))  # User specified

        # No internal hashTree for ThreadGroup here, it will be a sibling
        return thread_group

    # add_http_sampler now only creates the sampler itself, not its child hashTree
    def add_http_sampler(self, parent_element, name, method, path, parameters=None, body=None) -> ET.Element:
        sampler = self._create_element(parent_element, "HTTPSamplerProxy", {
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })

        # --- IMPORTANT: All direct properties of HTTPSamplerProxy must come BEFORE its SIBLING hashTree ---
        self._create_string_prop(sampler, "HTTPSampler.method", method)
        self._create_string_prop(sampler, "HTTPSampler.path", path)
        self._create_bool_prop(sampler, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(sampler, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(sampler, "HTTPSampler.use_keepalive", True)
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
            self._create_bool_prop(sampler, "HTTPSampler.postBodyRaw", True)

            body_data = self._create_element(sampler, "elementProp", {
                "name": "HTTPsampler.Arguments",  # This name is correct for raw body
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
            self._create_string_prop(http_argument, "Argument.metadata", "=")

        # No internal hashTree for Sampler, it will be a sibling
        return sampler

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

        # No internal hashTree for CSVDataSet, it will be a sibling
        return csv_config_element

    def add_response_assertion(self, parent_element, name, response_field, test_type, pattern):
        assertion = self._create_element(parent_element, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": name,
            "enabled": "true"
        })
        # ResponseAssertion properties
        test_strings_prop = self._create_collection_prop(assertion, "Assertion.test_strings")
        self._create_string_prop(test_strings_prop, "TestString", pattern)

        self._create_string_prop(assertion, "Assertion.custom_message", "")
        self._create_string_prop(assertion, "Assertion.test_field", response_field)
        self._create_string_prop(assertion, "Assertion.test_type", test_type)
        self._create_bool_prop(assertion, "Assertion.assume_success", False)
        self._create_string_prop(assertion, "Assertion.scope", "variable")
        self._create_bool_prop(assertion, "Assertion.override_existing_properties", False)

        # No internal hashTree for ResponseAssertion, it will be a sibling
        return assertion

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

        # No internal hashTree for JSONPostProcessor, it will be a sibling
        return extractor

    def add_header_manager(self, parent_element, headers: Dict[str, str], name="HTTP Header Manager"):
        header_manager = self._create_element(parent_element, "HeaderManager", {
            "guiclass": "HeaderPanel",
            "testclass": "HeaderManager",
            "testname": name,
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
        # No internal hashTree for HeaderManager, it will be a sibling
        return header_manager

    def add_view_results_tree_listener(self, parent_element, name="View Results Tree"):
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "ViewResultsFullVisualizer",  # Updated guiclass name
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        name_elem = self._create_element(obj_prop, "name")
        name_elem.text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
        # Corrected: Create direct elements for boolean flags instead of boolProp
        self._create_element(value_elem, "time").text = "true"
        self._create_element(value_elem, "latency").text = "true"
        self._create_element(value_elem, "timestamp").text = "true"
        self._create_element(value_elem, "success").text = "true"
        self._create_element(value_elem, "label").text = "true"
        self._create_element(value_elem, "code").text = "true"
        self._create_element(value_elem, "message").text = "true"
        self._create_element(value_elem, "threadName").text = "true"
        self._create_element(value_elem, "dataType").text = "true"
        self._create_element(value_elem, "encoding").text = "false"
        self._create_element(value_elem, "assertions").text = "true"
        self._create_element(value_elem, "subresults").text = "true"
        self._create_element(value_elem, "responseData").text = "false"
        self._create_element(value_elem, "samplerData").text = "false"
        self._create_element(value_elem, "xml").text = "false"
        # Corrected: fieldNames should also be a direct element, not stringProp
        self._create_element(value_elem, "fieldNames").text = "true"
        self._create_element(value_elem, "responseHeaders").text = "false"
        self._create_element(value_elem, "requestHeaders").text = "false"
        self._create_element(value_elem, "responseDataOnError").text = "false"
        self._create_element(value_elem, "saveAssertionResultsFailureMessage").text = "true"
        # Corrected: assertionsResultsToSave should also be a direct element, not stringProp
        self._create_element(value_elem, "assertionsResultsToSave").text = "0"
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        # No internal hashTree for ResultCollector, it will be a sibling
        return listener

    def add_summary_report_listener(self, parent_element, name="Summary Report"):
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "SummaryReport",  # Updated guiclass name
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        name_elem = self._create_element(obj_prop, "name")
        name_elem.text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
        # Corrected: Create direct elements for boolean flags instead of boolProp
        self._create_element(value_elem, "time").text = "true"
        self._create_element(value_elem, "latency").text = "true"
        self._create_element(value_elem, "timestamp").text = "true"
        self._create_element(value_elem, "success").text = "true"
        self._create_element(value_elem, "label").text = "true"
        self._create_element(value_elem, "code").text = "true"
        self._create_element(value_elem, "message").text = "true"
        self._create_element(value_elem, "threadName").text = "true"
        self._create_element(value_elem, "dataType").text = "true"
        self._create_element(value_elem, "encoding").text = "false"
        self._create_element(value_elem, "assertions").text = "true"
        self._create_element(value_elem, "subresults").text = "true"
        self._create_element(value_elem, "responseData").text = "false"
        self._create_element(value_elem, "samplerData").text = "false"
        self._create_element(value_elem, "xml").text = "false"
        # Corrected: fieldNames should also be a direct element, not stringProp
        self._create_element(value_elem, "fieldNames").text = "true"
        self._create_element(value_elem, "responseHeaders").text = "false"
        self._create_element(value_elem, "requestHeaders").text = "false"
        self._create_element(value_elem, "responseDataOnError").text = "false"
        self._create_element(value_elem, "saveAssertionResultsFailureMessage").text = "true"
        # Corrected: assertionsResultsToSave should also be a direct element, not stringProp
        self._create_element(value_elem, "assertionsResultsToSave").text = "0"
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        # No internal hashTree for ResultCollector, it will be a sibling
        return listener

    def generate_jmx(self, swagger_endpoints: List[SwaggerEndpoint],
                     mappings: Dict[str, Dict[str, str]],
                     thread_group_users: int,
                     ramp_up_time: int,
                     loop_count: int,
                     scenario_plan: Dict[str, Any],
                     database_connector: Any,  # DatabaseConnector instance
                     db_tables_schema: Dict[str, List[Dict[str, str]]]) -> (str, Optional[str]):

        # HTTP Request Defaults is already added in __init__ along with its sibling hashTree
        # self.http_defaults_children_hash_tree is the correct container for any children of HTTP Defaults (usually empty)

        # Step 1: Add Thread Group
        thread_group_element = self.add_thread_group(
            num_users=thread_group_users,
            ramp_up_time=ramp_up_time,
            loop_count=loop_count,
            parent_element=self.test_plan_children_hash_tree  # ThreadGroup is a child of TestPlan's children hashTree
        )
        # SIBLING hashTree for the Thread Group. This will contain all samplers, assertions, etc.
        thread_group_children_hash_tree = self._create_element(self.test_plan_children_hash_tree, "hashTree")

        csv_content = None
        # Collect data from DB for mapped parameters if database_connector is provided
        csv_data = {}
        csv_headers = []
        if database_connector and db_tables_schema:
            unique_mapped_columns = set()
            for endpoint_key, param_map in mappings.items():
                for param_name, db_column_ref in param_map.items():
                    unique_mapped_columns.add(db_column_ref)

            for col_ref in unique_mapped_columns:
                try:
                    table_name, column_name = col_ref.split('.')
                    if table_name in database_connector.get_tables():
                        df = database_connector.preview_data(table_name, limit=None)
                        if column_name in df.columns:
                            csv_data[col_ref] = df[column_name].tolist()
                            if col_ref not in csv_headers:
                                csv_headers.append(col_ref)
                except Exception as e:
                    logger.warning(f"Warning: Could not fetch data for {col_ref} from DB: {e}")

        # Create CSV Data Set Config if there's data to parameterize
        if csv_headers:
            csv_config_element = self.add_csv_data_config(
                parent_element=thread_group_children_hash_tree,  # Added as child of ThreadGroup's hashTree
                filename="data.csv",
                variable_names=",".join([col.replace('.', '_') for col in csv_headers])
            )
            self._create_element(thread_group_children_hash_tree, "hashTree")  # SIBLING hashTree for CSVDataSet

        # Generate CSV content for display
        if csv_headers and csv_data:
            csv_content = ",".join([col.replace('.', '_') for col in csv_headers]) + "\n"
            max_rows = 0
            if csv_data:
                max_rows = max(len(v) for v in csv_data.values())

            for i in range(max_rows):
                row_values = []
                for header in csv_headers:
                    values = csv_data.get(header, [])
                    row_values.append(str(values[i]) if i < len(values) else "")
                csv_content += ",".join(row_values) + "\n"

        # --- Add Homepage Request (HTTP GET to /) ---
        homepage_sampler = self.add_http_sampler(
            parent_element=thread_group_children_hash_tree,
            name="Homepage Request",
            method="GET",
            path="/",
            parameters={},
            body=None
        )
        # SIBLING hashTree for the Homepage Request sampler
        homepage_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree, "hashTree")

        # Add Assertion as a child of the sampler's sibling hashTree
        assertion = self.add_response_assertion(homepage_sampler_children_hash_tree, "Check Status 200",
                                                "Response Code", "Equals", "200")
        self._create_element(homepage_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the assertion

        # --- Add Create User Request (HTTP POST to /user) ---
        create_user_body = '{"username": "user${__Random(1,100,)}", "email": "user${__Random(1,100,)}@example.com"}'
        create_user_headers = {"Content-Type": "application/json"}
        create_user_sampler = self.add_http_sampler(
            parent_element=thread_group_children_hash_tree,
            name="Create User",
            method="POST",
            path="/user",
            parameters={},
            body=create_user_body
        )
        # SIBLING hashTree for the Create User sampler
        create_user_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree, "hashTree")

        # Add Header Manager
        if create_user_headers:
            header_manager = self.add_header_manager(create_user_sampler_children_hash_tree, create_user_headers,
                                                     name="HTTP Header Manager")
            self._create_element(create_user_sampler_children_hash_tree,
                                 "hashTree")  # SIBLING hashTree for HeaderManager

        # Add Assertion
        assertion = self.add_response_assertion(create_user_sampler_children_hash_tree, "Create User - Status 200",
                                                "Response Code", "Equals", "200")
        self._create_element(create_user_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the assertion

        # Add JSON Extractor
        extractor = self.add_json_extractor(create_user_sampler_children_hash_tree, "Extract User ID", "$.id",
                                            "user_id")
        self._create_element(create_user_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the extractor

        # --- Add Login User Request (HTTP POST to /login) ---
        login_user_body = '{"username": "user1", "password": "pass123"}'
        login_user_headers = {"Content-Type": "application/json"}
        login_user_sampler = self.add_http_sampler(
            parent_element=thread_group_children_hash_tree,
            name="Login User",
            method="POST",
            path="/login",
            parameters={},
            body=login_user_body
        )
        # SIBLING hashTree for the Login User sampler
        login_user_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree, "hashTree")

        # Add Header Manager
        if login_user_headers:
            header_manager = self.add_header_manager(login_user_sampler_children_hash_tree, login_user_headers,
                                                     name="HTTP Header Manager")
            self._create_element(login_user_sampler_children_hash_tree,
                                 "hashTree")  # SIBLING hashTree for HeaderManager

        # Add Assertion
        assertion = self.add_response_assertion(login_user_sampler_children_hash_tree, "Login User - Status 200",
                                                "Response Code", "Equals", "200")
        self._create_element(login_user_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the assertion

        # --- Add Get Pets By Status Request ---
        get_pets_parameters = {"status": "${pets_status}"}
        get_pets_sampler = self.add_http_sampler(
            parent_element=thread_group_children_hash_tree,
            name="Get Pets By Status",
            method="GET",
            path="/pet/findByStatus",
            parameters=get_pets_parameters,
            body=None
        )
        # SIBLING hashTree for the Get Pets By Status sampler
        get_pets_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree, "hashTree")

        # Add Assertion
        assertion = self.add_response_assertion(get_pets_sampler_children_hash_tree, "Get Pets By Status - Status 200",
                                                "Response Code", "Equals", "200")
        self._create_element(get_pets_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the assertion

        # --- Add other requests from scenario_plan (if any) ---
        if scenario_plan is None:
            scenario_plan = {"requests": []}

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

            current_sampler = self.add_http_sampler(thread_group_children_hash_tree, name, method, path,
                                                    processed_parameters, body)
            current_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree,
                                                                      "hashTree")  # SIBLING hashTree

            # Add Header Manager if headers are present for LLM-generated requests
            if headers:
                header_manager = self.add_header_manager(current_sampler_children_hash_tree, headers,
                                                         name="HTTP Header Manager")
                self._create_element(current_sampler_children_hash_tree,
                                     "hashTree")  # SIBLING hashTree for HeaderManager

            assertion = self.add_response_assertion(current_sampler_children_hash_tree, f"{name} - Status 200",
                                                    "Response Code", "Equals", "200")
            self._create_element(current_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for the assertion

            if method == "POST" and body and isinstance(body, str) and "id" in body:
                logger.debug(f"Adding JSON Extractor for {name} to capture 'id'.")
                extractor = self.add_json_extractor(current_sampler_children_hash_tree, f"{name} - Extract ID", "$.id",
                                                    f"{name.replace(' ', '_')}_id")
                self._create_element(current_sampler_children_hash_tree,
                                     "hashTree")  # SIBLING hashTree for the extractor

        # --- Add Listeners as per user's request ---
        # Listeners are direct children of TestPlan's children hash tree
        view_results_tree = self.add_view_results_tree_listener(self.test_plan_children_hash_tree)
        self._create_element(self.test_plan_children_hash_tree, "hashTree")  # SIBLING hashTree for View Results Tree

        summary_report = self.add_summary_report_listener(self.test_plan_children_hash_tree)
        self._create_element(self.test_plan_children_hash_tree, "hashTree")  # SIBLING hashTree for Summary Report

        # Generate XML string
        rough_string = ET.tostring(self.test_plan_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_jmx = reparsed.toprettyxml(indent="  ")

        # REMOVED: <!DOCTYPE jmeterTestPlan SYSTEM "jmeter.apache.org/dtd/jmeter_2_3.dtd">
        # This line is removed to prevent SAXParseException due to disallow-doctype-decl.
        final_jmx_content = pretty_jmx.split('?>\n', 1)[-1].strip()  # Remove old XML declaration and DOCTYPE if exists

        return final_jmx_content, csv_content
