import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any, Optional
import logging
import json  # Import json module for parsing LLM parameter strings

# Assuming SwaggerEndpoint dataclass is available from utils.swagger_parser
from utils.swagger_parser import SwaggerEndpoint

# from utils.database_connector import DatabaseConnector, DatabaseConfig # For potential future direct DB usage in JMX

logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    """
    Generates a JMeter .jmx script based on Swagger endpoints,
    database mappings, and user-defined test scenario details.
    """

    def __init__(self, test_plan_name: str = "Web Application Performance Test",
                 thread_group_name: str = "Users"):
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name

        # Align jmeter version with the user's working JMX for better compatibility
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0", jmeter="5.6.3")

        # This is the outermost hashTree element, a sibling to the TestPlan itself
        self.root_hash_tree_container = ET.SubElement(self.test_plan_root, "hashTree")

        # TestPlan element: Child of the root_hash_tree_container
        self.test_plan = self._create_element(self.root_hash_tree_container, "TestPlan", attrib={
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": self.test_plan_name,  # User specified name
            "enabled": "true"
        })
        self._create_string_prop(self.test_plan, "TestPlan.comments",
                                 "A test plan to evaluate the performance of a web application under load.")
        self._create_bool_prop(self.test_plan, "TestPlan.functional_mode", False)
        self._create_bool_prop(self.test_plan, "TestPlan.tearDown_on_shutdown", True)
        self._create_bool_prop(self.test_plan, "TestPlan.serialize_threadgroups", False)

        # User Defined Variables for TestPlan (Exact element as specified)
        user_defined_variables = self._create_element(self.test_plan, "elementProp", {
            "name": "TestPlan.user_defined_variables",
            "elementType": "Arguments",
            "guiclass": "ArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(user_defined_variables, "Arguments.arguments")  # Empty collection for arguments
        self._create_string_prop(self.test_plan, "TestPlan.user_define_classpath", "")

        # TestPlan's SIBLING hashTree - this will contain HTTP Defaults, Thread Groups, Listeners
        self.test_plan_children_hash_tree = ET.SubElement(self.root_hash_tree_container, "hashTree")

        # Default HTTP Request Defaults (add as child of test_plan_children_hash_tree)
        # These will be set based on the `app_base_url` from the UI
        self.http_defaults = self.add_http_request_defaults(self.test_plan_children_hash_tree)

        # SIBLING hashTree for HTTP Request Defaults
        self.http_defaults_children_hash_tree = ET.SubElement(self.test_plan_children_hash_tree, "hashTree")

        # CSV Data Set Config placeholder, added if needed later
        self.csv_config = None

    def _create_element(self, parent, tag, attrib=None):
        if attrib is None:
            attrib = {}
        return ET.SubElement(parent, tag, attrib)

    def _create_element_with_text(self, parent, tag, text, attrib=None):
        """Helper to create an element with text content."""
        element = self._create_element(parent, tag, attrib)
        element.text = str(text)
        return element

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

    def add_http_request_defaults(self, parent, protocol="https", domain="", port="", connect_timeout="",
                                  response_timeout=""):
        """
        Adds HTTP Request Defaults to the test plan.
        Domain and protocol are expected to be set based on the UI's app_base_url.
        """
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
        self._create_collection_prop(arguments_prop, "Arguments.arguments")  # Empty collection for arguments

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

        self._create_bool_prop(config, "HTTPSampler.send_chunked_post_body", False)
        self._create_bool_prop(config, "HTTPSampler.follow_redirects", True)
        self._create_bool_prop(config, "HTTPSampler.auto_redirects", False)
        self._create_bool_prop(config, "HTTPSampler.use_keepalive", True)
        self._create_bool_prop(config, "HTTPSampler.DO_MULTIPART_POST", False)
        self._create_bool_prop(config, "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART", True)
        self._create_bool_prop(config, "HTTPSampler.concurrentDwn", False)

        return config

    def add_thread_group(self, num_users, ramp_up_time, loop_count, parent_element, name="Main Users", is_setup=False,
                         is_teardown=False):
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

        # Setup/Teardown specific properties
        if is_setup:
            self._create_bool_prop(thread_group, "ThreadGroup.is_setUp_ThreadGroup", True)
            self._create_bool_prop(thread_group, "ThreadGroup.is_tearDown_ThreadGroup", False)
        elif is_teardown:
            self._create_bool_prop(thread_group, "ThreadGroup.is_setUp_ThreadGroup", False)
            self._create_bool_prop(thread_group, "ThreadGroup.is_tearDown_ThreadGroup", True)
        else:  # Regular thread group
            self._create_bool_prop(thread_group, "ThreadGroup.is_setUp_ThreadGroup", False)
            self._create_bool_prop(thread_group, "ThreadGroup.is_tearDown_ThreadGroup", False)

        main_controller = self._create_element(thread_group, "elementProp", {
            "name": "ThreadGroup.main_controller",
            "elementType": "LoopController",
            "guiclass": "LoopControlPanel",
            "testclass": "LoopController",
            "testname": "Loop Controller",
            "enabled": "true"
        })
        self._create_bool_prop(main_controller, "LoopController.continue_forever", loop_count == -1)
        self._create_string_prop(main_controller, "LoopController.loops", str(loop_count))

        return thread_group

    def add_http_sampler(self, parent_element, name, method, path, parameters=None, body=None) -> ET.Element:
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

        if method in ["POST", "PUT", "PATCH"] and body:
            self._create_bool_prop(sampler, "HTTPSampler.postBodyRaw", True)

            http_argument = self._create_element(collection_prop, "elementProp", {
                "name": "",
                "elementType": "HTTPArgument"
            })
            self._create_bool_prop(http_argument, "HTTPArgument.always_encode", False)
            self._create_string_prop(http_argument, "Argument.value", body)
            self._create_string_prop(http_argument, "Argument.metadata", "=")
        elif parameters:  # For GET/DELETE or form-urlencoded POST/PUT with parameters
            for param_name, param_value in parameters.items():
                arg_prop = self._create_element(collection_prop, "elementProp", {
                    "name": param_name,
                    "elementType": "HTTPArgument"
                })
                self._create_bool_prop(arg_prop, "HTTPArgument.always_encode", False)
                self._create_string_prop(arg_prop, "Argument.value", str(param_value))
                self._create_string_prop(arg_prop, "Argument.metadata", "=")
                self._create_string_prop(arg_prop, "Argument.name", param_name)

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

        return csv_config_element

    def add_response_assertion(self, parent_element, name, response_field, test_type, pattern):
        assertion = self._create_element(parent_element, "ResponseAssertion", {
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": name,
            "enabled": "true"
        })
        test_strings_prop = self._create_collection_prop(assertion, "Asserion.test_strings")
        self._create_string_prop(test_strings_prop, "TestString", pattern)

        self._create_string_prop(assertion, "Assertion.custom_message", "")
        self._create_string_prop(assertion, "Assertion.test_field",
                                 response_field)  # 1=response_data, 2=response_code, 4=response_headers, 5=request_headers, 6=url, 8=response_message
        self._create_string_prop(assertion, "Assertion.test_type",
                                 test_type)  # 1=contains, 2=matches, 4=equals, 8=not_contains, 16=not_matches, 32=not_equals
        self._create_bool_prop(assertion, "Assertion.assume_success", False)
        self._create_string_prop(assertion, "Assertion.scope", "variable")
        self._create_bool_prop(assertion, "Assertion.override_existing_properties", False)

        return assertion

    def add_json_assertion(self, parent_element, name, json_path_expr, expected_value, is_regex=False,
                           expect_null=False):
        """
        Adds a JSON Assertion (JSON JMESPath Assertion) to check values in JSON responses.
        """
        assertion = self._create_element(parent_element, "JSONPathAssertion", {
            "guiclass": "JSONPathAssertionGui",
            "testclass": "JSONPathAssertion",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(assertion, "JSONPathAssertion.jsonPath", json_path_expr)
        self._create_string_prop(assertion, "JSONPathAssertion.expectedValue", str(expected_value))
        self._create_bool_prop(assertion, "JSONPathAssertion.jsonValidation", True)
        self._create_bool_prop(assertion, "JSONPathAssertion.expectNull", expect_null)
        self._create_bool_prop(assertion, "JSONPathAssertion.regex", is_regex)
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
        return header_manager

    def add_constant_timer(self, parent_element, name, delay_milliseconds):
        """Adds a Constant Timer element."""
        timer = self._create_element(parent_element, "ConstantTimer", {
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(timer, "ConstantTimer.delay", str(delay_milliseconds))
        return timer

    def add_view_results_tree_listener(self, parent_element, name="View Results Tree"):
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "ViewResultsFullVisualizer",  # Corrected guiclass name
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
        self._create_element_with_text(value_elem, "assertionsResultsToSave", "0")
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        return listener

    def add_summary_report_listener(self, parent_element, name="Summary Report"):
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "SummaryReport",  # Updated guiclass name
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener,
                                        "objProp")  # FIX: Correctly initialize obj_prop as a child of listener
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
        self._create_element_with_text(value_elem, "assertionsResultsToSave", "0")
        self._create_element(value_elem, "bytes").text = "true"
        self._create_element(value_elem, "sentBytes").text = "true"
        self._create_element(value_elem, "url").text = "true"
        self._create_element(value_elem, "threadCounts").text = "true"
        self._create_element(value_elem, "idleTime").text = "true"
        self._create_element(value_elem, "connectTime").text = "true"

        return listener

    def add_aggregate_report_listener(self, parent_element, name="Aggregate Report"):
        listener = self._create_element(parent_element, "ResultCollector", {
            "guiclass": "StatVisualizer",  # Correct guiclass for Aggregate Report
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

    def _add_component_with_sibling_hash_tree(self, parent_container: ET.Element, component_element: ET.Element):
        """
        Helper to add a component element to a parent container, immediately followed by its
        empty hashTree sibling. This ensures correct JMeter XML structure for elements
        that are parents to other test elements.
        """
        # The component element is already created (e.g., by add_csv_data_config)
        # We just need to ensure its sibling hashTree is added to the same parent_container.
        ET.SubElement(parent_container, "hashTree")

    def generate_jmx(self,
                     app_base_url: str,
                     thread_group_users: int,
                     ramp_up_time: int,
                     loop_count: int,
                     scenario_plan: Dict[str, Any],  # This is now the definitive scenario
                     csv_data_to_include: Optional[str] = None,  # CSV content to be included in data.csv
                     global_constant_timer_delay: int = 0,
                     test_plan_name: str = "Web Application Performance Test",
                     thread_group_name: str = "Users",
                     http_defaults_protocol: str = "https",
                     http_defaults_domain: str = "",
                     http_defaults_port: str = "",
                     http_defaults_base_path: str = "/",
                     full_swagger_spec: Dict[str, Any] = None,  # Passed for internal reference
                     enable_setup_teardown_thread_groups: bool = False  # New flag
                     ) -> (str, Optional[str]):
        """
        Generates the JMeter JMX XML string based on the provided scenario plan.

        Args:
            app_base_url: The base URL of the application (e.g., https://example.com/v2).
            thread_group_users: Number of concurrent users.
            ramp_up_time: Time in seconds for all users to start.
            loop_count: Number of iterations per user (-1 for infinite).
            scenario_plan: A structured dictionary describing the test scenario,
                           derived directly from the UI inputs.
            csv_data_to_include: Optional string content for the data.csv file.
            global_constant_timer_delay: Delay for the global constant timer in milliseconds.
            test_plan_name: Name of the overall JMeter Test Plan.
            thread_group_name: Name of the main Thread Group.
            http_defaults_protocol: Protocol for HTTP Request Defaults.
            http_defaults_domain: Domain for HTTP Request Defaults.
            http_defaults_port: Port for HTTP Request Defaults.
            http_defaults_base_path: Base path for HTTP Request Defaults (from Swagger).
            full_swagger_spec: The complete, resolved Swagger specification dictionary.
            enable_setup_teardown_thread_groups: If True, adds Setup and Teardown Thread Groups.

        Returns:
            A tuple containing the JMeter JMX XML string and the CSV data string.
        """

        # Assign csv_data_to_include to a local variable 'csv_data'
        csv_data = csv_data_to_include

        # Update Test Plan name
        self._create_string_prop(self.test_plan, "testname", test_plan_name)

        # Update HTTP Request Defaults
        self._create_string_prop(self.http_defaults, "HTTPSampler.protocol", http_defaults_protocol)
        self._create_string_prop(self.http_defaults, "HTTPSampler.domain", http_defaults_domain)
        self._create_string_prop(self.http_defaults, "HTTPSampler.port", str(http_defaults_port))
        # HTTP Request Defaults path can also be set, ensuring base_path is always used.
        # This will simplify individual sampler paths if they only contain endpoint specific parts.
        self._create_string_prop(self.http_defaults, "HTTPSampler.path", http_defaults_base_path)

        # Main Test Plan hashTree content (this is a conceptual list, elements are added directly)

        # Step 1: Add Setup Thread Group (if enabled)
        if enable_setup_teardown_thread_groups:
            setup_tg = self.add_thread_group(num_users=1, ramp_up_time=0, loop_count=1,
                                             parent_element=self.test_plan_children_hash_tree,
                                             name="1-Setup Thread Group", is_setup=True)
            self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, setup_tg)

            # Example setup request (e.g., clearing cache, or specific login if not per user)
            # You would add specific setup samplers here if needed
            # login_sampler = self.add_http_sampler(setup_tg_hash_tree, "Setup_Login", "POST", "/v2/auth/admin_login")
            # self._create_element(setup_tg_hash_tree, "hashTree") # This would be inside setup_tg_hash_tree if it were a direct child

        # Step 2: Add CSV Data Set Config
        if csv_data:
            csv_lines = csv_data.strip().split('\n')
            if csv_lines and len(csv_lines) > 0:
                variable_names = csv_lines[0]
                csv_config_element = self.add_csv_data_config(
                    parent_element=self.test_plan_children_hash_tree,
                    # Correctly adds CSVDataSet to test_plan_children_hash_tree
                    filename="data.csv",
                    variable_names=variable_names
                )
                self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, csv_config_element)

        # Step 3: Add Main Thread Group
        main_thread_group_element = self.add_thread_group(
            num_users=thread_group_users,
            ramp_up_time=ramp_up_time,
            loop_count=loop_count,
            parent_element=self.test_plan_children_hash_tree,  # Main TG is a child of TestPlan's children hashTree
            name=self.thread_group_name
        )
        # SIBLING hashTree for the Main Thread Group. This will contain all samplers, assertions, etc.
        main_thread_group_children_hash_tree = ET.SubElement(self.test_plan_children_hash_tree, "hashTree")

        # Add global Constant Timer if enabled (as a child of the main thread group hash tree)
        if global_constant_timer_delay > 0:
            timer = self.add_constant_timer(
                main_thread_group_children_hash_tree,
                name="Global Constant Timer",
                delay_milliseconds=global_constant_timer_delay
            )
            self._add_component_with_sibling_hash_tree(main_thread_group_children_hash_tree, timer)

        # Add HTTP Request Samplers and associated elements based on scenario_plan
        for req_config in scenario_plan["requests"]:
            request_name = req_config["name"]
            method = req_config["method"]
            request_path = req_config["path"]
            parameters = req_config.get("parameters", {})
            body = req_config.get("body")
            headers = req_config.get("headers", {})
            assertions = req_config.get("assertions", [])
            json_extractors = req_config.get("json_extractors", [])
            think_time = req_config.get("think_time", 0)  # in seconds

            # Add HTTP Sampler to the main thread group's hash tree
            current_sampler = self.add_http_sampler(
                parent_element=main_thread_group_children_hash_tree,
                name=request_name,
                method=method,
                path=request_path,
                parameters=parameters,
                body=body
            )
            # SIBLING hashTree for the current sampler, where its children (headers, assertions, etc.) will go
            sampler_children_hash_tree = ET.SubElement(main_thread_group_children_hash_tree, "hashTree")

            # Add Header Manager if headers are present
            if headers:
                header_manager = self.add_header_manager(sampler_children_hash_tree, headers,
                                                         name=f"{request_name} Headers")
                self._add_component_with_sibling_hash_tree(sampler_children_hash_tree,
                                                           header_manager)  # Use helper here

            # Add JSON Extractors
            for extractor_config in json_extractors:
                extractor = self.add_json_extractor(
                    sampler_children_hash_tree,
                    name=f"{request_name} - Extract {extractor_config['var_name']}",
                    json_path_expr=extractor_config['json_path_expr'],
                    var_name=extractor_config['var_name']
                )
                self._add_component_with_sibling_hash_tree(sampler_children_hash_tree, extractor)  # Use helper here

            # Add Assertions
            for assertion_config in assertions:
                assertion_type = assertion_config["type"]
                assertion_value = assertion_config["value"]

                if assertion_type == "Response Code":
                    assertion_element = self.add_response_assertion(
                        sampler_children_hash_tree,
                        name=f"{request_name} - Check Status {assertion_value}",
                        response_field="2",  # 2 means Response Code
                        test_type="4",  # 4 means Equals
                        pattern=assertion_value
                    )
                elif assertion_type == "Response Body Contains":
                    assertion_element = self.add_response_assertion(
                        sampler_children_hash_tree,
                        name=f"{request_name} - Check Body Contains '{assertion_value}'",
                        response_field="1",  # 1 means Response Data
                        test_type="1",  # 1 means Contains
                        pattern=assertion_value
                    )
                elif assertion_type == "JSON Field Value":  # New JSON assertion
                    json_path = assertion_config.get("json_path_expr", "")
                    expected = assertion_config.get("expected_value", "")
                    is_regex = assertion_config.get("is_regex", False)
                    expect_null = assertion_config.get("expect_null", False)
                    assertion_element = self.add_json_assertion(
                        sampler_children_hash_tree,
                        name=f"{request_name} - JSON Field '{json_path}' Value",
                        json_path_expr=json_path,
                        expected_value=expected,
                        is_regex=is_regex,
                        expect_null=expect_null
                    )
                self._add_component_with_sibling_hash_tree(sampler_children_hash_tree,
                                                           assertion_element)  # Use helper here

            # Add Think Time (Constant Timer)
            if think_time > 0:
                think_time_timer = self.add_constant_timer(
                    sampler_children_hash_tree,
                    name=f"{request_name} Think Time",
                    delay_milliseconds=think_time * 1000  # Convert seconds to milliseconds
                )
                self._add_component_with_sibling_hash_tree(sampler_children_hash_tree,
                                                           think_time_timer)  # Use helper here

        # Step 4: Add Teardown Thread Group (if enabled)
        if enable_setup_teardown_thread_groups:
            teardown_tg = self.add_thread_group(num_users=1, ramp_up_time=0, loop_count=1,
                                                parent_element=self.test_plan_children_hash_tree,
                                                name="3-Teardown Thread Group", is_teardown=True)
            self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, teardown_tg)

            # Example teardown request (e.g., delete created user, logout)
            # You would add specific teardown samplers here if needed, often using variables extracted earlier.
            # logout_sampler = self.add_http_sampler(teardown_tg_hash_tree, "Teardown_Logout", "GET", "/v2/user/logout")
            # self._create_element(teardown_tg_hash_tree, "hashTree")

        # --- Add Listeners (placed at the end of the TestPlan's children hash tree) ---
        view_results_tree = self.add_view_results_tree_listener(self.test_plan_children_hash_tree)
        self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, view_results_tree)

        summary_report = self.add_summary_report_listener(self.test_plan_children_hash_tree)
        self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, summary_report)

        aggregate_report = self.add_aggregate_report_listener(self.test_plan_children_hash_tree)
        self._add_component_with_sibling_hash_tree(self.test_plan_children_hash_tree, aggregate_report)

        # Generate XML string
        rough_string = ET.tostring(self.test_plan_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_jmx = reparsed.toprettyxml(indent="  ")

        # Remove XML declaration and DOCTYPE for compatibility
        final_jmx_content = pretty_jmx.split('?>\n', 1)[-1].strip()

        return final_jmx_content, csv_data
