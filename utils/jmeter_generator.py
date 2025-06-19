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
        self.test_plan_hash_tree = None  # Represents the main hashTree under TestPlan

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
        self.test_plan_hash_tree = ET.SubElement(self.root, "hashTree")

        test_plan = ET.SubElement(self.test_plan_hash_tree, "TestPlan", {
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": self.test_plan_name,
            "enabled": "true"
        })
        ET.SubElement(test_plan, "stringProp", {"name": "TestPlan.comments"})
        ET.SubElement(test_plan, "boolProp", {"name": "TestPlan.functional_mode"}).text = "false"
        ET.SubElement(test_plan, "boolProp", {"name": "TestPlan.serialize_threadgroups"}).text = "false"

        # User Defined Variables - often placed directly under TestPlan's hashTree
        user_defined_variables = ET.SubElement(test_plan, "elementProp", {
            "name": "TestPlan.user_defined_variables",
            "elementType": "Arguments",
            "guiclass": "ArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        ET.SubElement(user_defined_variables, "collectionProp", {"name": "Arguments.arguments"})
        ET.SubElement(test_plan, "stringProp", {"name": "TestPlan.user_define_classpath"})

        ET.SubElement(self.test_plan_hash_tree, "hashTree")  # HashTree for the TestPlan children

    def _create_thread_group(self, parent_hash_tree, num_users: int, ramp_up_time: int, loop_count: int,
                             enable_setup_teardown: bool = False, group_type: str = "normal"):
        """
        Creates a Thread Group element.
        group_type can be "normal", "setup", or "teardown".
        """
        thread_group_guiclass = "ThreadGroupGui"
        thread_group_testclass = "ThreadGroup"

        thread_group_name = self.thread_group_name
        if group_type == "setup":
            thread_group_name = "SetUp Thread Group"
            thread_group_guiclass = "SetupThreadGroupGui"
            thread_group_testclass = "SetupThreadGroup"
        elif group_type == "teardown":
            thread_group_name = "TearDown Thread Group"
            thread_group_guiclass = "TeardownThreadGroupGui"
            thread_group_testclass = "PostThreadGroup"  # JMeter uses PostThreadGroup for tearDown

        thread_group = ET.SubElement(parent_hash_tree, "ThreadGroup", {
            "guiclass": thread_group_guiclass,
            "testclass": thread_group_testclass,
            "testname": thread_group_name,
            "enabled": "true"
        })

        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}).text = "continue"
        ET.SubElement(thread_group, "elementProp", {
            "name": "ThreadGroup.main_controller",
            "elementType": "LoopController",
            "guiclass": "LoopControlPanel",
            "testclass": "LoopController",
            "testname": "Loop Controller",
            "enabled": "true"
        })

        loop_controller = thread_group.find(".//LoopController")
        ET.SubElement(loop_controller, "boolProp",
                      {"name": "LoopController.continue_forever"}).text = "true" if loop_count == -1 else "false"
        ET.SubElement(loop_controller, "stringProp", {"name": "LoopController.loops"}).text = str(
            loop_count) if loop_count != -1 else ""

        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}).text = str(num_users)
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}).text = str(ramp_up_time)
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}).text = "false"
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.duration"}).text = ""
        ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.delay"}).text = ""
        ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"}).text = "true"

        thread_group_hash_tree = ET.SubElement(parent_hash_tree, "hashTree")
        return thread_group_hash_tree

    def _add_http_request_defaults(self, parent_hash_tree, protocol, domain, port, base_path):
        """Adds an HTTP Request Defaults element."""
        http_defaults = ET.SubElement(parent_hash_tree, "ConfigTestElement", {
            "guiclass": "HttpDefaultsGui",
            "testclass": "ConfigTestElement",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        })
        # Capture the elementProp directly
        http_sampler_arguments_element_prop = ET.SubElement(http_defaults, "elementProp", {
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        # Use the captured elementProp to add its child
        ET.SubElement(http_sampler_arguments_element_prop, "collectionProp", {"name": "Arguments.arguments"})

        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.domain"}).text = domain
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.port"}).text = str(port) if port else ""
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.protocol"}).text = protocol
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.contentEncoding"})
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.path"}).text = base_path
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.concurrentDwn"}).text = "false"
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.concurrentPool"}).text = "6"
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.connect_timeout"})
        ET.SubElement(http_defaults, "stringProp", {"name": "HTTPSampler.response_timeout"})

        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for HTTP Request Defaults

    def _add_http_request_sampler(self, name: str, method: str, path: str, parameters: Dict[str, str],
                                  body: Optional[str], headers: Dict[str, str], protocol: str, domain: str, port: str,
                                  base_path: str):
        """
        Creates an HTTP Request Sampler.
        Headers, assertions, extractors are handled separately.
        """
        sampler = ET.SubElement(ET.Element("template"), "HTTPSamplerProxy", {  # Create in a dummy root to return
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": name,
            "enabled": "true"
        })
        ET.SubElement(sampler, "elementProp", {
            "name": "HTTPsampler.Arguments",
            "elementType": "Arguments",
            "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })

        args_collection = ET.SubElement(sampler.find(".//Arguments"), "collectionProp", {"name": "Arguments.arguments"})

        for param_name, param_value in parameters.items():
            arg_element = ET.SubElement(args_collection, "elementProp",
                                        {"name": param_name, "elementType": "HTTPArgument"})
            ET.SubElement(arg_element, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "false"
            ET.SubElement(arg_element, "stringProp", {"name": "Argument.value"}).text = str(param_value)
            ET.SubElement(arg_element, "stringProp", {"name": "Argument.metadata"}).text = "="
            ET.SubElement(arg_element, "stringProp", {"name": "Argument.name"}).text = param_name

        ET.SubElement(sampler, "stringProp", {
            "name": "HTTPSampler.domain"})  # Domain will come from HTTP Request Defaults or be explicitly set
        ET.SubElement(sampler, "stringProp",
                      {"name": "HTTPSampler.port"})  # Port will come from HTTP Request Defaults or be explicitly set
        ET.SubElement(sampler, "stringProp", {
            "name": "HTTPSampler.protocol"})  # Protocol will come from HTTP Request Defaults or be explicitly set
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.contentEncoding"})
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.path"}).text = path
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.method"}).text = method
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.follow_redirects"}).text = "true"
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.auto_redirects"}).text = "false"
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.use_keepalive"}).text = "true"
        ET.SubElement(sampler, "boolProp",
                      {"name": "HTTPSampler.DO_MULTIPART_POST"}).text = "false"  # Only if multipart
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.embedded_url_re"})
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.connect_timeout"})
        ET.SubElement(sampler, "stringProp", {"name": "HTTPSampler.response_timeout"})

        if body:
            ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"}).text = "true"
            body_prop = ET.SubElement(sampler, "elementProp",
                                      {"name": "HTTPsampler.Arguments", "elementType": "Arguments"})
            body_coll = ET.SubElement(body_prop, "collectionProp", {"name": "Arguments.arguments"})
            body_arg = ET.SubElement(body_coll, "elementProp", {"name": "", "elementType": "HTTPArgument"})
            ET.SubElement(body_arg, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "false"
            ET.SubElement(body_arg, "stringProp", {"name": "Argument.value"}).text = body
            ET.SubElement(body_arg, "stringProp", {"name": "Argument.metadata"}).text = "="

        return sampler

    def _add_header_manager(self, parent_hash_tree, headers: Dict[str, str]):
        """
        Adds an HTTP Header Manager.
        Correctly structured as per JMeter's requirements.
        """
        header_manager = ET.SubElement(parent_hash_tree, "HeaderManager", {
            "guiclass": "HeaderPanel",
            "testclass": "HeaderManager",
            "testname": "HTTP Header Manager",
            "enabled": "true"
        })

        # collectionProp is a direct child of HeaderManager
        collection_prop = ET.SubElement(header_manager, "collectionProp", {"name": "HeaderManager.headers"})

        for header_name, header_value in headers.items():
            element_prop = ET.SubElement(collection_prop, "elementProp", {"name": header_name, "elementType": "Header"})
            ET.SubElement(element_prop, "stringProp", {"name": "Header.name"}).text = header_name
            ET.SubElement(element_prop, "stringProp", {"name": "Header.value"}).text = header_value

        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for the Header Manager

    def _add_json_extractor(self, parent_hash_tree, json_path_expr: str, var_name: str, sample_method: str,
                            sample_path: str):
        """Adds a JSON Extractor."""
        extractor = ET.SubElement(parent_hash_tree, "JSONPostProcessor", {
            "guiclass": "JSONPostProcessorGui",
            "testclass": "JSONPostProcessor",
            "testname": f"JSON Extractor - {var_name}",
            "enabled": "true"
        })
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.referenceNames"}).text = var_name
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.jsonPathExprs"}).text = json_path_expr
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.matchNumbers"}).text = "1"
        ET.SubElement(extractor, "stringProp",
                      {"name": "JSONPostProcessor.defaultValues"}).text = f"NOT_FOUND_{var_name}"
        ET.SubElement(extractor, "boolProp", {
            "name": "JSONPostProcessor.indirectExtr"}).text = "true"  # Ensure this is enabled for dynamic content
        ET.SubElement(extractor, "stringProp", {"name": "JSONPostProcessor.scope"}).text = "body"  # Default scope

        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for the JSON Extractor

    def _add_response_assertion(self, parent_hash_tree, response_code: str = "200"):
        """Adds a Response Assertion for HTTP status code."""
        assertion = ET.SubElement(parent_hash_tree, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": "Response Code Assertion",
            "enabled": "true"
        })
        ET.SubElement(assertion, "collectionProp", {"name": "Asserion.test_strings"})
        ET.SubElement(assertion.find(".//collectionProp"), "stringProp", {"name": "200"}).text = response_code
        ET.SubElement(assertion, "stringProp", {"name": "Assertion.custom_message"})
        ET.SubElement(assertion, "stringProp", {"name": "Assertion.test_field"}).text = "jm_rc"  # Response Code
        ET.SubElement(assertion, "boolProp", {"name": "Assertion.assume_success"}).text = "false"
        ET.SubElement(assertion, "intProp", {"name": "Assertion.test_type"}).text = "2"  # Equals

        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for the Response Assertion

    def _add_constant_timer(self, parent_hash_tree, delay_ms: int):
        """Adds a Constant Timer."""
        timer = ET.SubElement(parent_hash_tree, "ConstantTimer", {
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": f"Constant Timer - {delay_ms}ms",
            "enabled": "true"
        })
        ET.SubElement(timer, "stringProp", {"name": "ConstantTimer.delay"}).text = str(delay_ms)
        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for the Constant Timer

    def _add_csv_data_set_config(self, parent_hash_tree, filename: str, variable_names: List[str]):
        """Adds a CSV Data Set Config element."""
        csv_config = ET.SubElement(parent_hash_tree, "CSVDataSet", {
            "guiclass": "TestBeanGUI",
            "testclass": "CSVDataSet",
            "testname": f"CSV Data Set Config - {filename}",
            "enabled": "true"
        })
        ET.SubElement(csv_config, "stringProp", {"name": "filename"}).text = filename
        ET.SubElement(csv_config, "stringProp", {"name": "fileEncoding"}).text = "UTF-8"
        ET.SubElement(csv_config, "stringProp", {"name": "variableNames"}).text = ",".join(variable_names)
        ET.SubElement(csv_config, "stringProp", {"name": "delimiter"}).text = ","
        ET.SubElement(csv_config, "boolProp", {"name": "ignoreFirstLine"}).text = "true"
        ET.SubElement(csv_config, "boolProp", {"name": "quotedData"}).text = "false"
        ET.SubElement(csv_config, "boolProp", {"name": "recycle"}).text = "true"
        ET.SubElement(csv_config, "boolProp", {"name": "stopThread"}).text = "false"
        ET.SubElement(csv_config, "stringProp", {"name": "shareMode"}).text = "shareModeAll"  # shareMode.get=""

        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for the CSV Data Set Config

    def _add_listeners(self, parent_hash_tree):
        """Adds common listeners: Summary Report and View Results Tree."""
        # Summary Report
        summary_report = ET.SubElement(parent_hash_tree, "ResultCollector", {
            "guiclass": "SummariserGui",  # Corrected guiclass
            "testclass": "ResultCollector",
            "testname": "Summary Report",
            "enabled": "true"
        })
        ET.SubElement(summary_report, "boolProp", {"name": "ResultCollector.error_logging"}).text = "false"
        obj_prop_summary = ET.SubElement(summary_report, "objProp", {"name": "saveConfig"})
        ET.SubElement(obj_prop_summary, "boolProp", {"name": "ResultCollector.enabled"}).text = "true"
        ET.SubElement(obj_prop_summary, "stringProp", {"name": "filename"}).text = ""
        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for Summary Report

        # View Results Tree
        view_results = ET.SubElement(parent_hash_tree, "ResultCollector", {
            "guiclass": "ViewResultsFullVisualizer",
            "testclass": "ResultCollector",
            "testname": "View Results Tree",
            "enabled": "true"
        })
        ET.SubElement(view_results, "boolProp", {"name": "ResultCollector.error_logging"}).text = "false"
        obj_prop_view = ET.SubElement(view_results, "objProp", {"name": "saveConfig"})
        ET.SubElement(obj_prop_view, "boolProp", {"name": "ResultCollector.enabled"}).text = "true"
        ET.SubElement(obj_prop_view, "stringProp", {"name": "filename"}).text = ""
        ET.SubElement(parent_hash_tree, "hashTree")  # HashTree for View Results Tree

    def generate_jmx(self, app_base_url: str, thread_group_users: int, ramp_up_time: int, loop_count: int,
                     scenario_plan: Dict[str, Any], csv_data_to_include: Optional[str] = None,
                     global_constant_timer_delay: int = 0, test_plan_name: str = "Test Plan",
                     thread_group_name: str = "Users", http_defaults_protocol: str = "https",
                     http_defaults_domain: str = "example.com", http_defaults_port: str = "",
                     http_defaults_base_path: str = "/", full_swagger_spec: Dict[str, Any] = None,
                     enable_setup_teardown_thread_groups: bool = False):
        """
        Generates a JMeter JMX script based on the provided scenario plan.
        """
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name
        self._create_test_plan()

        # Add HTTP Request Defaults at the Test Plan level
        self._add_http_request_defaults(self.test_plan_hash_tree.find("hashTree"), http_defaults_protocol,
                                        http_defaults_domain, http_defaults_port, http_defaults_base_path)

        # Main Thread Group (Users)
        main_thread_group_hash_tree = self._create_thread_group(
            self.test_plan_hash_tree.find("hashTree"),  # Append to the main Test Plan hash tree
            thread_group_users, ramp_up_time, loop_count,
            group_type="normal"
        )

        csv_variable_names = set()  # To collect unique variable names for CSV config
        for req_config in scenario_plan['requests']:
            for param_field in req_config.get('parameters_and_body_fields', []):
                if param_field.get('source') == 'from_csv' and \
                        'table_name' in param_field and 'column_name' in param_field:
                    csv_variable_names.add(f"csv_{param_field['table_name']}_{param_field['column_name']}")

        # Add CSV Data Set Config if there are any CSV mappings
        if csv_data_to_include and csv_variable_names:
            self._add_csv_data_set_config(main_thread_group_hash_tree, "data.csv", sorted(list(csv_variable_names)))

        # Add global Constant Timer if enabled
        if global_constant_timer_delay > 0:
            self._add_constant_timer(main_thread_group_hash_tree, global_constant_timer_delay)

        # Add scenario requests
        for request_config in scenario_plan['requests']:
            sampler_name = request_config['name']
            method = request_config['method']
            path = request_config['path']
            params = request_config.get('parameters', {})
            headers = request_config.get('headers', {})
            body = request_config.get('body')
            assertions = request_config.get('assertions', [])
            json_extractors = request_config.get('json_extractors', [])
            think_time = request_config.get('think_time', 0)  # Local think time for the request

            # Create HTTP Sampler
            http_sampler = self._add_http_request_sampler(sampler_name, method, path, params, body, headers,
                                                          http_defaults_protocol, http_defaults_domain,
                                                          http_defaults_port, http_defaults_base_path)
            main_thread_group_hash_tree.append(http_sampler)

            # Create a hashTree specifically for this sampler's children
            sampler_children_hash_tree = ET.SubElement(main_thread_group_hash_tree, "hashTree")

            # Add Header Manager (if headers exist for this specific request)
            if headers:
                self._add_header_manager(sampler_children_hash_tree, headers)

            # Add JSON Extractors
            for extractor in json_extractors:
                # Pass method and path for better naming/context in JSON Extractor
                self._add_json_extractor(sampler_children_hash_tree, extractor['json_path_expr'], extractor['var_name'],
                                         method, path)

            # Add Assertions
            for assertion in assertions:
                if assertion['type'] == 'Response Code':
                    self._add_response_assertion(sampler_children_hash_tree, assertion['value'])
                # Extend with other assertion types as needed

            # Add a Constant Timer specifically for this request, if defined in the scenario plan
            if think_time > 0:
                self._add_constant_timer(sampler_children_hash_tree, think_time)

        # Add Listeners to the main thread group
        self._add_listeners(main_thread_group_hash_tree)

        # Setup and TearDown Thread Groups (if enabled)
        if enable_setup_teardown_thread_groups:
            setup_thread_group_hash_tree = self._create_thread_group(
                self.test_plan_hash_tree.find("hashTree"),
                1, 1, 1,  # Typically 1 user, 1 sec ramp-up, 1 loop for setup/teardown
                group_type="setup"
            )
            # Add setup elements here (e.g., login, data setup)
            # For now, just add a placeholder
            placeholder_setup = ET.SubElement(setup_thread_group_hash_tree, "GenericController", {
                "guiclass": "LogicControllerGui", "testclass": "GenericController",
                "testname": "Setup Actions Placeholder", "enabled": "true"
            })
            ET.SubElement(setup_thread_group_hash_tree, "hashTree")  # HashTree for setup placeholder

            teardown_thread_group_hash_tree = self._create_thread_group(
                self.test_plan_hash_tree.find("hashTree"),
                1, 1, 1,
                group_type="teardown"
            )
            # Add teardown elements here (e.g., logout, data cleanup)
            # For now, just add a placeholder
            placeholder_teardown = ET.SubElement(teardown_thread_group_hash_tree, "GenericController", {
                "guiclass": "LogicControllerGui", "testclass": "GenericController",
                "testname": "Teardown Actions Placeholder", "enabled": "true"
            })
            ET.SubElement(teardown_thread_group_hash_tree, "hashTree")  # HashTree for teardown placeholder

        pretty_xml = self._pretty_print_xml(self.root)
        return pretty_xml, ET.ElementTree(self.root)

