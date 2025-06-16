import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import random
import string
import json  # Import json module for parsing LLM parameter strings
from urllib.parse import urlparse  # Import urlparse for parsing base URL

# Import SwaggerEndpoint as it's used in type hints for generate_jmx
from utils.swagger_parser import SwaggerEndpoint  # Keep this import, it's correct

logger = logging.getLogger(__name__)


class JMeterScriptGenerator:  # JMeterScriptGenerator class definition starts here
    def __init__(self, test_plan_name: str = "Web Application Performance Test", thread_group_name: str = "Users"):
        # Align jmeter version with the user's working JMX for better compatibility
        # Keeping 5.2.1 as it's the version from the user's working JMX reference,
        # and more likely to be compatible with older XStream if that's the issue.
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0",
                                         jmeter="5.6.3")  # Updated JMeter version

        # This is the outermost hashTree element, a sibling to the TestPlan itself
        self.root_hash_tree_container = ET.SubElement(self.test_plan_root, "hashTree")

        # TestPlan element: Child of the root_hash_tree_container
        self._add_comment(self.root_hash_tree_container, "This is the root element of the JMeter Test Plan.")
        self.test_plan = self._create_element(self.root_hash_tree_container, "TestPlan", attrib={
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": test_plan_name,  # User specified name
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

        # Default HTTP Request Defaults (will be populated dynamically later in generate_jmx)
        self.http_defaults_children_hash_tree = None  # Will be set after adding HTTP Defaults

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

    def _add_comment(self, parent, comment_text):
        """Adds an XML comment element to the parent."""
        comment = ET.Comment(comment_text)
        parent.append(comment)

    def add_http_request_defaults(self, parent, protocol: str, domain: str, port: str, base_path: str,
                                  connect_timeout: str = "5000", response_timeout: str = "10000"):
        self._add_comment(parent, "HTTP Request Defaults: Configures default settings for HTTP requests.")
        config = self._create_element(parent, "ConfigTestElement", {
            "guiclass": "HttpDefaultsGui",
            "testclass": "HttpDefaults",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        })

        arguments_prop = self._create_element(config, "elementProp", {
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(arguments_prop, "Arguments.arguments")

        self._create_string_prop(config, "HTTPSampler.protocol", protocol)
        self._create_string_prop(config, "HTTPSampler.domain", domain)
        self._create_string_prop(config, "HTTPSampler.port", str(port))
        self._create_string_prop(config, "HTTPSampler.contentEncoding", "")
        self._create_string_prop(config, "HTTPSampler.proxyHost", "")
        self._create_string_prop(config, "HTTPSampler.proxyPort", "")
        self._create_string_prop(config, "HTTPSampler.proxyUser", "")
        self._create_string_prop(config, "HTTPSampler.proxyPass", "")
        self._create_string_prop(config, "HTTPSampler.connect_timeout", connect_timeout)
        self._create_string_prop(config, "HTTPSampler.response_timeout", response_timeout)
        self._create_string_prop(config, "HTTPSampler.path", base_path)  # Set base path here

        self._create_bool_prop(config, "HTTPSampler.send_chunked_post_body", False)
        self._create_bool_prop(config, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(config, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(config, "HTTPSampler.use_keepalive", True)
        self._create_bool_prop(config, "HTTPSampler.DO_MULTIPART_POST", False)
        self._create_bool_prop(config, "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART", True)
        self._create_bool_prop(config, "HTTPSampler.concurrentDwn", False)

        return config

    def add_thread_group(self, num_users: int, ramp_up_time: int, loop_count: int, parent_element: ET.Element,
                         name: str):
        self._add_comment(parent_element,
                          f"Thread Group: Simulates {num_users} users over {ramp_up_time} seconds, looping {loop_count if loop_count != -1 else 'infinitely'}.")
        thread_group = self._create_element(parent_element, "ThreadGroup", {
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(thread_group, "ThreadGroup.on_sample_error", "continue")

        self._create_string_prop(thread_group, "ThreadGroup.num_threads", str(num_users))
        self._create_string_prop(thread_group, "ThreadGroup.ramp_time", str(ramp_up_time))
        self._create_bool_prop(thread_group, "ThreadGroup.scheduler", False)
        self._create_string_prop(thread_group, "ThreadGroup.duration", "")
        self._create_string_prop(thread_group, "ThreadGroup.delay", "")
        self._create_bool_prop(thread_group, "ThreadGroup.same_user_on_next_iteration", True)

        main_controller = self._create_element(thread_group, "elementProp", {
            "name": "ThreadGroup.main_controller",
            "elementType": "LoopController",
            "guiclass": "LoopControlPanel",
            "testclass": "LoopController",
            "testname": "Loop Controller",
            "enabled": "true"
        })
        self._create_bool_prop(main_controller, "LoopController.continue_forever", (loop_count == -1))
        self._create_string_prop(main_controller, "LoopController.loops", str(loop_count))

        return thread_group

    def add_http_sampler(self, parent_element: ET.Element, name: str, method: str, path: str,
                         parameters: Optional[Dict[str, Any]] = None, body: Optional[str] = None) -> ET.Element:
        self._add_comment(parent_element, f"HTTP Request Sampler: {name} - {method} {path}")
        sampler = self._create_element(parent_element, "HTTPSamplerProxy", {
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })

        self._create_string_prop(sampler, "HTTPSampler.method", method)
        self._create_string_prop(sampler, "HTTPSampler.path", path)
        self._create_bool_prop(sampler, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(sampler, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(sampler, "HTTPSampler.use_keepalive", True)
        self._create_string_prop(sampler, "HTTPSampler.connect_timeout", "")
        self._create_string_prop(sampler, "HTTPSampler.response_timeout", "")

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

        if method in ["POST", "PUT", "PATCH"] and body:
            self._create_bool_prop(sampler, "HTTPSampler.postBodyRaw", True)

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
            self._create_string_prop(http_argument, "Argument.metadata", "=")

        return sampler

    def add_csv_data_config(self, parent_element: ET.Element, filename: str, variable_names: str, delimiter: str = ",",
                            quoted_data: bool = False):
        self._add_comment(parent_element, "CSV Data Set Config: Provides data for variables from 'data.csv'.")
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

        return csv_config_element

    def add_response_assertion(self, parent_element: ET.Element, name: str, response_field: str, test_type: str,
                               pattern: str):
        self._add_comment(parent_element, f"Response Assertion: Checks if '{response_field}' {test_type} '{pattern}'.")
        assertion = self._create_element(parent_element, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": name,
            "enabled": "true"
        })
        test_strings_prop = self._create_collection_prop(assertion, "Assertion.test_strings")
        self._create_string_prop(test_strings_prop, "TestString", pattern)

        self._create_string_prop(assertion, "Assertion.custom_message", "")
        self._create_string_prop(assertion, "Assertion.test_field", response_field)
        self._create_string_prop(assertion, "Assertion.test_type", test_type)
        self._create_bool_prop(assertion, "Assertion.assume_success", False)
        self._create_string_prop(assertion, "Assertion.scope", "variable")
        self._create_bool_prop(assertion, "Assertion.override_existing_properties", False)

        return assertion

    def add_json_extractor(self, parent_element: ET.Element, name: str, json_path_expr: str, var_name: str,
                           match_no: str = "1", default_value: str = "NOT_FOUND"):
        self._add_comment(parent_element, f"JSON Extractor: Extracts '{json_path_expr}' into variable '{var_name}'.")
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
        self._create_string_prop(extractor, "JSONPostProcessor.scope", "body")

        return extractor

    def add_header_manager(self, parent_element: ET.Element, headers: Dict[str, str],
                           name: str = "HTTP Header Manager"):
        self._add_comment(parent_element, "HTTP Header Manager: Manages request headers.")
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
        return header_manager

    def add_view_results_tree_listener(self, parent_element: ET.Element, name: str = "View Results Tree"):
        self._add_comment(parent_element,
                          "Listener: View Results Tree - Displays sampler results during/after test execution.")
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "ViewResultsFullVisualizer",
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        name_elem = self._create_element(obj_prop, "name")
        name_elem.text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
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
        self._create_element(value_elem, "fieldNames").text = "true"
        self._create_element(value_elem, "responseHeaders").text = "false"
        self._create_element(value_elem, "requestHeaders").text = "false"
        self._create_element(value_elem, "responseDataOnError").text = "false"
        self._create_element(value_elem, "saveAssertionResultsFailureMessage").text = "true"
        self._create_element(value_elem, "assertionsResultsToSave").text = "0"
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        return listener

    def add_summary_report_listener(self, parent_element: ET.Element, name: str = "Summary Report"):
        self._add_comment(parent_element, "Listener: Summary Report - Provides a concise summary of test results.")
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "SummaryReport",
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        name_elem = self._create_element(obj_prop, "name")
        name_elem.text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
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
        self._create_element(value_elem, "fieldNames").text = "true"
        self._create_element(value_elem, "responseHeaders").text = "false"
        self._create_element(value_elem, "requestHeaders").text = "false"
        self._create_element(value_elem, "responseDataOnError").text = "false"
        self._create_element(value_elem, "saveAssertionResultsFailureMessage").text = "true"
        self._create_element(value_elem, "assertionsResultsToSave").text = "0"
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        return listener

    def add_aggregate_report_listener(self, parent_element: ET.Element, name: str = "Aggregate Report"):
        self._add_comment(parent_element, "Listener: Aggregate Report - Provides aggregated statistics for samples.")
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "StatVisualizer",
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        name_elem = self._create_element(obj_prop, "name")
        name_elem.text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
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
        self._create_element(value_elem, "fieldNames").text = "true"
        self._create_element(value_elem, "responseHeaders").text = "false"
        self._create_element(value_elem, "requestHeaders").text = "false"
        self._create_element(value_elem, "responseDataOnError").text = "false"
        self._create_element(value_elem, "saveAssertionResultsFailureMessage").text = "true"
        self._create_element(value_elem, "assertionsResultsToSave").text = "0"
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        return listener

    def add_constant_timer(self, parent_element: ET.Element, delay_ms: int, name: str = "Constant Timer"):
        self._add_comment(parent_element, f"Constant Timer: Introduces a {delay_ms}ms delay between requests.")
        timer = self._create_element(parent_element, "ConstantTimer", {
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(timer, "ConstantTimer.delay", str(delay_ms))
        return timer

    def generate_jmx(self,
                     app_base_url: str,
                     thread_group_users: int,
                     ramp_up_time: int,
                     loop_count: int,
                     scenario_plan: Dict[str, Any],
                     csv_data_to_include: Optional[str] = None,
                     global_constant_timer_delay: int = 0,
                     test_plan_name: str = "Web Application Performance Test",
                     thread_group_name: str = "Users",
                     http_defaults_protocol: str = "https",
                     http_defaults_domain: str = "",
                     http_defaults_port: str = "",
                     http_defaults_base_path: str = "/",
                     full_swagger_spec: Optional[Dict[str, Any]] = None  # Added for potential future use or debugging
                     ) -> (str, Optional[str]):

        # HTTP Request Defaults is added here, before the Thread Group
        http_defaults = self.add_http_request_defaults(
            self.test_plan_children_hash_tree,
            protocol=http_defaults_protocol,
            domain=http_defaults_domain,
            port=http_defaults_port,
            base_path=http_defaults_base_path
        )
        self.http_defaults_children_hash_tree = self._create_element(self.test_plan_children_hash_tree, "hashTree")

        # Step 1: Add Thread Group
        thread_group_element = self.add_thread_group(
            num_users=thread_group_users,
            ramp_up_time=ramp_up_time,
            loop_count=loop_count,
            parent_element=self.test_plan_children_hash_tree,
            name=thread_group_name
        )
        # SIBLING hashTree for the Thread Group. This will contain all samplers, assertions, etc.
        thread_group_children_hash_tree = self._create_element(self.test_plan_children_hash_tree, "hashTree")

        # Add Global Constant Timer if enabled
        if global_constant_timer_delay > 0:
            self.add_constant_timer(thread_group_children_hash_tree, global_constant_timer_delay)
            self._create_element(thread_group_children_hash_tree, "hashTree")  # SIBLING hashTree for the timer

        # Add CSV Data Set Config if there's data to parameterize
        csv_file_name = "data.csv"
        if csv_data_to_include:
            # Extract variable names from the first line of csv_data_to_include
            csv_lines = csv_data_to_include.strip().split('\n')
            if len(csv_lines) > 0:
                csv_variable_names = csv_lines[0]  # First line is headers
                self.add_csv_data_config(
                    parent_element=thread_group_children_hash_tree,
                    filename=csv_file_name,
                    variable_names=csv_variable_names,
                    delimiter=","  # Assuming comma delimiter for now
                )
                self._create_element(thread_group_children_hash_tree, "hashTree")  # SIBLING hashTree for CSVDataSet

        # --- Add requests from scenario_plan ---
        if scenario_plan is None:
            scenario_plan = {"requests": []}

        for request_data in scenario_plan['requests']:
            method = request_data['method']
            path = request_data['path']
            name = request_data['name']
            body = request_data.get('body')
            headers = request_data.get('headers', {})
            parameters_from_config = request_data.get('parameters', {})
            assertions = request_data.get('assertions', [])
            json_extractors = request_data.get('json_extractors', [])

            current_sampler = self.add_http_sampler(thread_group_children_hash_tree, name, method, path,
                                                    parameters_from_config, body)
            # Each sampler must be followed by its own hashTree for its children elements
            current_sampler_children_hash_tree = self._create_element(thread_group_children_hash_tree, "hashTree")

            # Add Header Manager as a child of the current sampler's hashTree
            if headers:
                self.add_header_manager(current_sampler_children_hash_tree, headers)
                self._create_element(current_sampler_children_hash_tree,
                                     "hashTree")  # SIBLING hashTree for HeaderManager

            # Add Assertions as children of the current sampler's hashTree
            for assertion_data in assertions:
                if assertion_data['type'] == "Response Code":
                    self.add_response_assertion(
                        current_sampler_children_hash_tree,
                        f"{name} - {assertion_data['type']} {assertion_data['value']}",
                        "Assertion.response_code",
                        "Equals",
                        assertion_data['value']
                    )
                    self._create_element(current_sampler_children_hash_tree,
                                         "hashTree")  # SIBLING hashTree for Assertion

            # Add JSON Extractors as children of the current sampler's hashTree
            for extractor_data in json_extractors:
                self.add_json_extractor(
                    current_sampler_children_hash_tree,
                    f"{name} - Extract {extractor_data['var_name']}",
                    extractor_data['json_path_expr'],
                    extractor_data['var_name']
                )
                self._create_element(current_sampler_children_hash_tree, "hashTree")  # SIBLING hashTree for Extractor

        # --- Add Listeners at the Test Plan level (children of test_plan_children_hash_tree) ---
        self.add_view_results_tree_listener(self.test_plan_children_hash_tree)
        self._create_element(self.test_plan_children_hash_tree, "hashTree")

        self.add_summary_report_listener(self.test_plan_children_hash_tree)
        self._create_element(self.test_plan_children_hash_tree, "hashTree")

        self.add_aggregate_report_listener(self.test_plan_children_hash_tree)
        self._create_element(self.test_plan_children_hash_tree, "hashTree")

        # Generate XML string
        rough_string = ET.tostring(self.test_plan_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_jmx = reparsed.toprettyxml(indent="  ")

        # Clean up the XML declaration to prevent issues
        final_jmx_content = pretty_jmx.split('?>\n', 1)[-1].strip()
        # Add a standard XML declaration at the top
        final_jmx_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + final_jmx_content

        return final_jmx_content, csv_data_to_include
