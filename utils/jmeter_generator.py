import xml.etree.ElementTree as ET
import xml.dom.minidom
import random
import string
import json
import logging
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    def __init__(self, test_plan_name="Test Plan", thread_group_name="Users"):
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name
        self.root = None
        self.hash_tree_node = None  # Represents the main hashTree under TestPlan

    def _pretty_print_xml(self, element):
        """Returns a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(element, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _create_test_plan(self):
        """Creates the basic JMeter Test Plan structure."""
        self.root = ET.Element("jmeterTestPlan", {
            "version": "1.2",
            "properties": "5.0",
            "jmeter": "5.6.2"  # Assuming a recent JMeter version for compatibility
        })

        test_plan = ET.SubElement(self.root, "TestPlan", {
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": self.test_plan_name,
            "enabled": "true"
        })

        ET.SubElement(test_plan, "stringProp", {"name": "TestPlan.comments"})
        ET.SubElement(test_plan, "boolProp", {"name": "TestPlan.functional_mode"}).text = "false"
        ET.SubElement(test_plan, "boolProp", {"name": "TestPlan.tearDown_on_shutdown"}).text = "true"
        ET.SubElement(test_plan, "boolProp", {"name": "TestPlan.serialize_threadgroups"}).text = "false"

        self.hash_tree_node = ET.SubElement(self.root, "hashTree")  # This is the main hash tree for the Test Plan
        return test_plan

    def _add_user_defined_variables(self, parent_element, variables: Dict[str, str]):
        """Adds User Defined Variables to the Test Plan or Thread Group."""
        if not variables:
            return

        udv = ET.SubElement(parent_element, "Arguments", {
            "guiclass": "ArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        ET.SubElement(udv, "collectionProp", {"name": "Arguments.arguments"})

        for name, value in variables.items():
            ET.SubElement(udv.find("collectionProp"), "elementProp", {
                "name": name,
                "elementType": "Argument",
                "testclass": "Argument",
                "guiclass": "ArgumentPanel"
            })
            ET.SubElement(udv.find(f"collectionProp/elementProp[@name='{name}']"), "stringProp",
                          {"name": "Argument.name"}).text = name
            ET.SubElement(udv.find(f"collectionProp/elementProp[@name='{name}']"), "stringProp",
                          {"name": "Argument.value"}).text = value
            ET.SubElement(udv.find(f"collectionProp/elementProp[@name='{name}']"), "stringProp",
                          {"name": "Argument.metadata"}).text = "="

        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for UDV

    def _add_http_request_defaults(self, parent_element, protocol, domain, port, base_path):
        """Adds HTTP Request Defaults."""
        http_defaults = ET.SubElement(parent_element, "ConfigTestElement", {
            "guiclass": "HttpDefaultsGui",
            "testclass": "ConfigTestElement",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        })
        ET.SubElement(http_defaults, "elementProp",
                      {"name": "HTTPsampler.Arguments", "elementType": "Arguments", "guiclass": "HTTPArgumentsPanel",
                       "testclass": "Arguments", "enabled": "true"})
        ET.SubElement(http_defaults.find("elementProp"), "collectionProp", {"name": "Arguments.arguments"})

        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.domain"}).text = domain
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.port"}).text = str(port) if port else ""
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.protocol"}).text = protocol
        ET.SubElement(http_defaults, "stringProp", {
            "name": "HTTPSampler.path"}).text = base_path if base_path != '/' else ""  # JMeter path defaults can be empty string for root
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.contentEncoding"}).text = ""
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.implementation"}).text = "HttpClient4"
        ET.SubElement(http_defaults, "boolProp", {"name": "HTTPSampler.as_browser"}).text = "false"
        ET.SubElement(http_defaults, "boolProp", {"name": "HTTPSampler.concurrentDwn"}).text = "false"
        ET.SubElement(http_defaults, "boolProp", {"name": "HTTPSampler.image_parser"}).text = "false"
        ET.SubElement(http_defaults, "boolProp", {"name": "HTTPSampler.postBodyRaw"}).text = "false"
        ET.SubElement(http_defaults, "elementProp", {"name": "HTTPsampler.response_timeout",
                                                     "elementType": "long"}).text = "30000"  # Default 30 sec timeout
        ET.SubElement(http_defaults, "elementProp", {"name": "HTTPsampler.connect_timeout",
                                                     "elementType": "long"}).text = "5000"  # Default 5 sec connect timeout

        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for HTTP Request Defaults

    def _add_http_cookie_manager(self, parent_element):
        """Adds an HTTP Cookie Manager."""
        cookie_manager = ET.SubElement(parent_element, "CookieManager", {
            "guiclass": "CookiePanel",
            "testclass": "CookieManager",
            "testname": "HTTP Cookie Manager",
            "enabled": "true"
        })
        # Properties for Cookie Manager
        ET.SubElement(cookie_manager, "boolProp", {"name": "CookieManager.clearEachIteration"}).text = "true"
        ET.SubElement(cookie_manager, "stringProp", {"name": "CookieManager.policy"}).text = "standard"
        ET.SubElement(cookie_manager, "stringProp", {"name": "CookieManager.implementation"}).text = "HttpClient4"

        # Add the hashTree sibling for the Cookie Manager
        ET.SubElement(parent_element, "hashTree")

    def _add_csv_data_set_config(self, parent_element, filename="data.csv"):
        """Adds a CSV Data Set Config element."""
        csv_config = ET.SubElement(parent_element, "CSVDataSet", {
            "guiclass": "TestBeanGUI",
            "testclass": "CSVDataSet",
            "testname": f"CSV Data Config ({filename})",
            "enabled": "true"
        })
        ET.SubElement(csv_config, "stringProp", {"name": "delimiter"}).text = ","
        ET.SubElement(csv_config, "stringProp", {"name": "fileEncoding"}).text = "UTF-8"
        ET.SubElement(csv_config, "stringProp", {"name": "filename"}).text = filename
        ET.SubElement(csv_config, "boolProp", {"name": "ignoreFirstLine"}).text = "true"
        ET.SubElement(csv_config, "boolProp", {"name": "quotedData"}).text = "false"
        ET.SubElement(csv_config, "boolProp", {"name": "recycle"}).text = "true"
        ET.SubElement(csv_config, "stringProp", {"name": "separator"}).text = ","
        ET.SubElement(csv_config, "boolProp", {"name": "stopThread"}).text = "false"
        ET.SubElement(csv_config, "stringProp", {"name": "variableNames"}).text = ""  # Will be populated by CSV header
        ET.SubElement(csv_config, "boolProp", {"name": "shareMode"}).text = "true"

        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for CSV Config

        return csv_config  # Return so variableNames can be updated later

    def _add_constant_timer(self, parent_element, delay_ms: int):
        """Adds a Constant Timer."""
        timer = ET.SubElement(parent_element, "ConstantTimer", {
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": f"Constant Timer ({delay_ms} ms)",
            "enabled": "true"
        })
        ET.SubElement(timer, "stringProp", {"name": "ConstantTimer.delay"}).text = str(delay_ms)
        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for Constant Timer

    def _add_http_request_sampler(self, parent_element, name, method, path, headers, parameters, body,
                                  is_raw_body=False):
        """Adds an HTTP Request Sampler."""
        sampler = ET.SubElement(parent_element, "HTTPSamplerProxy", {
            "guiclass": "HttpSamplerGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })

        ET.SubElement(sampler, "elementProp",
                      {"name": "HTTPsampler.Arguments", "elementType": "Arguments", "guiclass": "HTTPArgumentsPanel",
                       "testclass": "Arguments", "enabled": "true"})
        arguments_prop = sampler.find("elementProp[@name='HTTPsampler.Arguments']")
        collection_prop = ET.SubElement(arguments_prop, "collectionProp", {"name": "Arguments.arguments"})

        # Add query parameters
        for param_name, param_value in parameters.items():
            arg_elem = ET.SubElement(collection_prop, "elementProp",
                                     {"name": param_name, "elementType": "HTTPArgument"})
            ET.SubElement(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "true"
            ET.SubElement(arg_elem, "stringProp", {"name": "Argument.value"}).text = str(param_value)
            ET.SubElement(arg_elem, "stringProp", {"name": "Argument.metadata"}).text = "="
            ET.SubElement(arg_elem, "boolProp", {"name": "HTTPArgument.use_equals"}).text = "true"
            ET.SubElement(arg_elem, "stringProp", {"name": "Argument.name"}).text = param_name

        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.method"}).text = method
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.path"}).text = path
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.follow_redirects"}).text = "true"
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.auto_redirects"}).text = "false"
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.use_keepalive"}).text = "true"
        ET.SubElement(sampler, "boolProp",
                      {"name": "HTTPSampler.DO_MULTIPART_POST"}).text = "false"  # Set to true if multipart/form-data
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART"}).text = "false"
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.image_parser"}).text = "false"
        ET.SubElement(sampler, "stringProp",
                      {"name": "HTTPSampler.connect_timeout"}).text = ""  # Use default from HTTP Request Defaults
        ET.SubElement(sampler, "stringProp",
                      {"name": "HTTPSampler.response_timeout"}).text = ""  # Use default from HTTP Request Defaults

        if body:
            ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"}).text = "true"
            body_arg_elem = ET.SubElement(arguments_prop, "elementProp",
                                          {"name": "postBody", "elementType": "HTTPArgument"})
            ET.SubElement(body_arg_elem, "stringProp", {"name": "Argument.value"}).text = body
            ET.SubElement(body_arg_elem, "stringProp", {"name": "Argument.metadata"}).text = "="

        # Add HTTP Header Manager
        if headers:
            header_manager_tree = ET.SubElement(parent_element, "hashTree")  # Hash tree for Header Manager
            header_manager = ET.SubElement(header_manager_tree, "HeaderManager", {
                "guiclass": "HeaderPanel",
                "testclass": "HeaderManager",
                "testname": "HTTP Header Manager",
                "enabled": "true"
            })
            ET.SubElement(header_manager, "collectionProp", {"name": "HeaderManager.headers"})

            for header_name, header_value in headers.items():
                header_elem = ET.SubElement(header_manager.find("collectionProp"), "elementProp",
                                            {"name": header_name, "elementType": "Header"})
                ET.SubElement(header_elem, "stringProp", {"name": "Header.name"}).text = header_name
                ET.SubElement(header_elem, "stringProp", {"name": "Header.value"}).text = header_value

        # Sampler's own hash tree for children (e.g., assertions, extractors)
        return ET.SubElement(parent_element, "hashTree")

    def _add_json_extractor(self, parent_element, var_name, json_path_expr):
        """Adds a JSON Extractor."""
        extractor = ET.SubElement(parent_element, "JSONPostProcessor", {
            "guiclass": "JSONPostProcessorGui",
            "testclass": "JSONPostProcessor",
            "testname": f"JSON Extractor - {var_name}",
            "enabled": "true"
        })
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.referenceNames"}).text = var_name
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.jsonPathExprs"}).text = json_path_expr
        ET.SubElement(extractor, "stringProp",
                      {"name": "JSONPostProcessor.matchNumbers"}).text = "1"  # Extract first match
        ET.SubElement(extractor, "stringProp",
                      {"name": "JSONPostProcessor.defaultValues"}).text = f"NOT_FOUND_{var_name}"
        ET.SubElement(extractor, "boolProp", {"name": "JSONPostProcessor.computeConcatenation"}).text = "false"

        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for JSON Extractor

    def _add_response_assertion(self, parent_element, response_code="200", contains_text=""):
        """Adds a Response Assertion."""
        assertion = ET.SubElement(parent_element, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": "Response Assertion",
            "enabled": "true"
        })
        ET.SubElement(assertion, "collectionProp", {"name": "Asserion.test_strings"})
        if response_code:
            ET.SubElement(assertion.find("collectionProp"), "stringProp", {"name": response_code}).text = response_code
        if contains_text:
            ET.SubElement(assertion.find("collectionProp"), "stringProp", {"name": contains_text}).text = contains_text
            ET.SubElement(assertion, "stringProp", {"name": "Assertion.test_field"}).text = "Assertion.response_data"
            ET.SubElement(assertion, "boolProp", {"name": "Assertion.assume_success"}).text = "false"
            ET.SubElement(assertion, "intProp", {"name": "Assertion.test_type"}).text = "2"  # Contains

        ET.SubElement(assertion, "stringProp", {"name": "Assertion.custom_message"}).text = ""
        ET.SubElement(assertion, "stringProp", {"name": "Assertion.test_field"}).text = "Assertion.response_code"
        ET.SubElement(assertion, "boolProp", {"name": "Assertion.assume_success"}).text = "false"
        ET.SubElement(assertion, "intProp", {"name": "Assertion.test_type"}).text = "8"  # Equals

        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for Response Assertion

    def _add_setup_teardown_thread_group(self, parent_element, group_type: str):
        """Adds a Setup or Teardown Thread Group."""
        thread_group = ET.SubElement(parent_element, "SetupThreadGroup" if group_type == "setup" else "PostThreadGroup",
                                     {
                                         "guiclass": "SetupThreadGroupGui" if group_type == "setup" else "PostThreadGroupGui",
                                         "testclass": "SetupThreadGroup" if group_type == "setup" else "PostThreadGroup",
                                         "testname": f"{group_type.capitalize()} Thread Group",
                                         "enabled": "true"
                                     })
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}).text = "continue"
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.num_threads"}).text = "1"
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.ramp_time"}).text = "1"
        ET.SubElement(thread_group, "longProp", {"name": "ThreadGroup.start_time"}).text = "0"
        ET.SubElement(thread_group, "longProp", {"name": "ThreadGroup.end_time"}).text = "0"
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}).text = "false"
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.duration"}).text = ""
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.delay"}).text = ""
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"}).text = "true"
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.ramp_up"}).text = "1"
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.num_threads"}).text = "1"
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.once_only"}).text = "true"
        ET.SubElement(thread_group, "stringProp", {"name": "TestPlan.comments"}).text = ""

        return ET.SubElement(parent_element, "hashTree")  # Hash tree for the Thread Group itself

    def generate_jmx(self, app_base_url: str, thread_group_users: int, ramp_up_time: int, loop_count: int,
                     scenario_plan: Dict[str, Any], csv_data_to_include: Optional[str] = None,
                     global_constant_timer_delay: int = 0,
                     test_plan_name: str = "Generated Test Plan",
                     thread_group_name: str = "Users",
                     http_defaults_protocol: str = "https",
                     http_defaults_domain: str = "example.com",
                     http_defaults_port: str = "",
                     http_defaults_base_path: str = "/",
                     full_swagger_spec: Dict[str, Any] = None,
                     enable_setup_teardown_thread_groups: bool = False
                     ):
        """
        Generates a JMeter JMX script based on the provided scenario plan.
        """
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name

        self._create_test_plan()

        test_plan_hash_tree = self.hash_tree_node

        # Extract variables from full_swagger_spec definitions if available and relevant
        user_defined_vars = {}
        if full_swagger_spec and 'definitions' in full_swagger_spec:
            # Example: extracting common values or enums to UDVs
            if 'Pet' in full_swagger_spec['definitions'] and 'status' in full_swagger_spec['definitions']['Pet'][
                'properties']:
                enum_values = full_swagger_spec['definitions']['Pet']['properties']['status'].get('enum', [])
                if enum_values:
                    user_defined_vars['petStatusEnum'] = ','.join(enum_values)
            if 'User' in full_swagger_spec['definitions'] and 'username' in full_swagger_spec['definitions']['User'][
                'properties']:
                # Could add a placeholder for dynamic username if not coming from CSV
                user_defined_vars['defaultUsername'] = "jmeter_user"

        if user_defined_vars:
            self._add_user_defined_variables(test_plan_hash_tree, user_defined_vars)

        self._add_http_request_defaults(test_plan_hash_tree, http_defaults_protocol, http_defaults_domain,
                                        http_defaults_port, http_defaults_base_path)
        self._add_http_cookie_manager(test_plan_hash_tree)

        csv_config_element = None
        if csv_data_to_include:
            csv_config_element = self._add_csv_data_set_config(test_plan_hash_tree)
            # Update variableNames in the CSVDataSet element
            first_line = csv_data_to_include.split('\n')[0]
            if first_line:
                csv_config_element.find("stringProp[@name='variableNames']").text = first_line.replace(',', ', ')

            csv_file_name = "data.csv"  # Standard name
            with open(csv_file_name, "w") as f:
                f.write(csv_data_to_include)

        if enable_setup_teardown_thread_groups:
            # Setup Thread Group
            setup_thread_group_hash_tree = self._add_setup_teardown_thread_group(test_plan_hash_tree, "setup")
            # You can add specific setup requests here
            # Example: _add_http_request_sampler(setup_thread_group_hash_tree, "Setup Request", "GET", "/setup", {}, {})

        # Main Thread Group
        thread_group = ET.SubElement(test_plan_hash_tree, "ThreadGroup", {
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": self.thread_group_name,
            "enabled": "true"
        })
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}).text = "continue"
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.num_threads"}).text = str(thread_group_users)
        ET.SubElement(thread_group, "intProp", {"name": "ThreadGroup.ramp_time"}).text = str(ramp_up_time)
        ET.SubElement(thread_group, "longProp", {"name": "ThreadGroup.start_time"}).text = "0"
        ET.SubElement(thread_group, "longProp", {"name": "ThreadGroup.end_time"}).text = "0"
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}).text = "false"
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.duration"}).text = ""
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.delay"}).text = ""
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"}).text = "true"

        # Set loop count
        loop_controller_prop = ET.SubElement(thread_group, "elementProp",
                                             {"name": "ThreadGroup.main_controller", "elementType": "LoopController",
                                              "guiclass": "LoopControlPanel", "testclass": "LoopController",
                                              "enabled": "true"})
        ET.SubElement(loop_controller_prop, "boolProp",
                      {"name": "LoopController.continue_forever"}).text = "true" if loop_count == -1 else "false"
        if loop_count != -1:
            ET.SubElement(loop_controller_prop, "intProp", {"name": "LoopController.loops"}).text = str(loop_count)
        else:
            ET.SubElement(loop_controller_prop, "stringProp",
                          {"name": "LoopController.loops"}).text = "-1"  # JMeter expects -1 as string for infinite

        thread_group_hash_tree = ET.SubElement(test_plan_hash_tree, "hashTree")  # Hash tree for Thread Group elements

        # Add global Constant Timer if enabled
        if global_constant_timer_delay > 0:
            self._add_constant_timer(thread_group_hash_tree, global_constant_timer_delay)

        # Add requests from scenario plan
        for request_config in scenario_plan.get("requests", []):
            sampler_hash_tree = self._add_http_request_sampler(
                thread_group_hash_tree,
                request_config["name"],
                request_config["method"],
                request_config["path"],
                request_config.get("headers", {}),
                request_config.get("parameters", {}),
                request_config.get("body"),
                request_config.get("is_raw_body", False)
            )

            # Add JSON Extractors
            for extractor in request_config.get("json_extractors", []):
                self._add_json_extractor(sampler_hash_tree, extractor["var_name"], extractor["json_path_expr"])

            # Add Assertions
            for assertion in request_config.get("assertions", []):
                if assertion["type"] == "Response Code":
                    self._add_response_assertion(sampler_hash_tree, response_code=assertion["value"])
                elif assertion["type"] == "Response Body Contains":
                    self._add_response_assertion(sampler_hash_tree, contains_text=assertion["value"])

            # Add Think Time (per request)
            if request_config.get("think_time", 0) > 0:
                self._add_constant_timer(sampler_hash_tree, request_config["think_time"])

        if enable_setup_teardown_thread_groups:
            # Teardown Thread Group
            teardown_thread_group_hash_tree = self._add_setup_teardown_thread_group(test_plan_hash_tree, "teardown")
            # You can add specific teardown requests here
            # Example: _add_http_request_sampler(teardown_thread_group_hash_tree, "Teardown Request", "GET", "/teardown", {}, {})

        # Add standard listeners (Summary Report, View Results Tree)
        self._add_summary_report(test_plan_hash_tree)
        self._add_view_results_tree(test_plan_hash_tree)

        jmx_content = self._pretty_print_xml(self.root)

        # Return JMX content and CSV file name if generated
        return jmx_content, "data.csv" if csv_data_to_include else None

    def _add_summary_report(self, parent_element):
        """Adds a Summary Report listener."""
        summary_report = ET.SubElement(parent_element, "kg.apc.jmeter.reporters.Summarizer", {
            "guiclass": "kg.apc.jmeter.reporters.SummarizerGui",
            "testclass": "kg.apc.jmeter.reporters.Summarizer",
            "testname": "Summary Report",
            "enabled": "true"
        })
        ET.SubElement(summary_report, "boolProp", {"name": "ResultCollector.error_logging"}).text = "false"
        ET.SubElement(summary_report, "objProp", {"name": "saveConfig"}).text = ""
        ET.SubElement(summary_report.find("objProp"), "boolProp", {"name": "ResultCollector.enabled"}).text = "true"
        ET.SubElement(summary_report.find("objProp"), "stringProp", {"name": "filename"}).text = ""
        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for Summary Report

    def _add_view_results_tree(self, parent_element):
        """Adds a View Results Tree listener."""
        view_results = ET.SubElement(parent_element, "ResultCollector", {
            "guiclass": "ViewResultsFullVisualizer",
            "testclass": "ResultCollector",
            "testname": "View Results Tree",
            "enabled": "true"
        })
        ET.SubElement(view_results, "boolProp", {"name": "ResultCollector.error_logging"}).text = "false"
        ET.SubElement(view_results, "objProp", {"name": "saveConfig"}).text = ""
        ET.SubElement(view_results.find("objProp"), "boolProp", {"name": "ResultCollector.enabled"}).text = "true"
        ET.SubElement(view_results.find("objProp"), "stringProp", {"name": "filename"}).text = ""
        ET.SubElement(parent_element, "hashTree")  # Sibling hash tree for View Results Tree

