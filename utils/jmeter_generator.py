import xml.etree.ElementTree as ET
from xml.dom import minidom
from urllib.parse import urlparse
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import random
import string
import json

logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    def __init__(self, test_plan_name: str = "Web Application Performance Test",
                 thread_group_name: str = "Users"):
        # Initialize the root of the JMeter Test Plan XML structure
        self.test_plan_root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0", jmeter="5.6.3")

        # This is the outermost hashTree element, a sibling to the TestPlan itself
        self.root_hash_tree_container = self._create_element(self.test_plan_root, "hashTree")

        # TestPlan element: Child of the root_hash_tree_container
        self.test_plan = self._create_element(self.root_hash_tree_container, "TestPlan", attrib={
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": test_plan_name,  # Use provided test plan name
            "enabled": "true"
        })
        self._create_string_prop(self.test_plan, "TestPlan.comments",
                                 "A test plan to evaluate the performance of a web application under load.")
        self._create_bool_prop(self.test_plan, "TestPlan.functional_mode", False)
        self._create_bool_prop(self.test_plan, "TestPlan.tearDown_on_shutdown", True)
        self._create_bool_prop(self.test_plan, "TestPlan.serialize_threadgroups", False)

        # User Defined Variables for TestPlan
        user_defined_variables = self._create_element(self.test_plan, "elementProp", attrib={
            "name": "TestPlan.user_defined_variables",
            "elementType": "Arguments",
            "guiclass": "ArgumentsPanel",
            "testclass": "Arguments",
            "testname": "User Defined Variables",
            "enabled": "true"
        })
        self._create_collection_prop(user_defined_variables, "Arguments.arguments")
        self._create_string_prop(self.test_plan, "TestPlan.user_define_classpath", "")

        # CORRECTED: This hashTree should be a SIBLING to TestPlan,
        # both being children of root_hash_tree_container.
        self.test_plan_global_hashtree = self._create_element(self.root_hash_tree_container, "hashTree")

        self.csv_config = None  # Placeholder for CSV config
        self.thread_group_name = thread_group_name  # Store thread group name

    def _create_element(self, parent: ET.Element, tag: str, attrib: Optional[Dict[str, str]] = None) -> ET.Element:
        """Helper to create an XML SubElement."""
        if attrib is None:
            attrib = {}
        return ET.SubElement(parent, tag, attrib)

    def _create_collection_prop(self, parent: ET.Element, name: str) -> ET.Element:
        """Helper to create a <collectionProp> element."""
        prop = self._create_element(parent, "collectionProp", {"name": name})
        return prop

    def _create_string_prop(self, parent: ET.Element, name: str, value: Any) -> ET.Element:
        """Helper to create a <stringProp> element."""
        prop = self._create_element(parent, "stringProp", {"name": name})
        prop.text = str(value)
        return prop

    def _create_bool_prop(self, parent: ET.Element, name: str, value: bool) -> ET.Element:
        """Helper to create a <boolProp> element."""
        prop = self._create_element(parent, "boolProp", {"name": name})
        prop.text = "true" if value else "false"
        return prop

    def _create_int_prop(self, parent: ET.Element, name: str, value: int) -> ET.Element:
        """Helper to create an <intProp> element."""
        prop = self._create_element(parent, "intProp", {"name": name})
        prop.text = str(value)
        return prop

    def add_http_request_defaults(self, parent: ET.Element, protocol: str = "https", domain: str = "example.com",
                                  port: str = "", connect_timeout: str = "", response_timeout: str = "") -> ET.Element:
        """Adds an HTTP Request Defaults Config Element."""
        config = self._create_element(parent, "ConfigTestElement", attrib={
            "guiclass": "HttpDefaultsGui",
            "testclass": "HttpDefaults",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        })

        arguments_prop = self._create_element(config, "elementProp", attrib={
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
        self._create_string_prop(config, "HTTPSampler.contentEncoding", "UTF-8")
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

        return config

    def add_thread_group(self, num_users: int, ramp_up_time: int, loop_count: int,
                         parent_element: ET.Element) -> ET.Element:
        """Adds a Thread Group element. This method returns ONLY the thread group element.
           Its associated hashTree must be added as a SIBLING in the calling code."""
        thread_group = self._create_element(parent_element, "ThreadGroup", attrib={
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": self.thread_group_name,  # Use provided test plan name
            "enabled": "true"
        })
        self._create_string_prop(thread_group, "ThreadGroup.on_sample_error", "continue")

        self._create_string_prop(thread_group, "ThreadGroup.num_threads", str(num_users))
        self._create_string_prop(thread_group, "ThreadGroup.ramp_time", str(ramp_up_time))
        self._create_bool_prop(thread_group, "ThreadGroup.scheduler", False)
        self._create_string_prop(thread_group, "ThreadGroup.duration", "")
        self._create_string_prop(thread_group, "ThreadGroup.delay", "")
        self._create_bool_prop(thread_group, "ThreadGroup.same_user_on_next_iteration", True)

        main_controller = self._create_element(thread_group, "elementProp", attrib={
            "name": "ThreadGroup.main_controller",
            "elementType": "LoopController",
            "guiclass": "LoopControlPanel",
            "testclass": "LoopController",
            "testname": "Loop Controller",
            "enabled": "true"
        })
        if loop_count == -1:
            self._create_bool_prop(main_controller, "LoopController.continue_forever", True)
            self._create_string_prop(main_controller, "LoopController.loops", "-1")
        else:
            self._create_bool_prop(main_controller, "LoopController.continue_forever", False)
            self._create_string_prop(main_controller, "LoopController.loops", str(loop_count))

        return thread_group

    def add_http_sampler(self, parent_element: ET.Element, name: str, method: str, path: str,
                         parameters: Optional[Dict[str, str]] = None, body: Optional[str] = None) -> ET.Element:
        """Adds an HTTP Sampler element. This method returns ONLY the sampler element.
           Its associated hashTree must be added as a SIBLING in the calling code."""
        sampler = self._create_element(parent_element, "HTTPSamplerProxy", attrib={
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

        # Arguments for parameters (URL parameters or form data)
        arguments = self._create_element(sampler, "elementProp", attrib={
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
                arg_prop = self._create_element(collection_prop, "elementProp", attrib={
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

            body_data_arguments = self._create_element(sampler, "elementProp", attrib={
                "name": "HTTPsampler.Arguments",
                "elementType": "Arguments",
                "guiclass": "HTTPArgumentsPanel",
                "testclass": "Arguments",
                "enabled": "true"
            })
            body_data_collection_prop = self._create_collection_prop(body_data_arguments, "Arguments.arguments")

            http_argument = self._create_element(body_data_collection_prop, "elementProp", attrib={
                "name": "",  # This must be empty for raw body
                "elementType": "HTTPArgument"
            })
            self._create_bool_prop(http_argument, "HTTPArgument.always_encode", False)
            self._create_string_prop(http_argument, "Argument.value", body)
            self._create_string_prop(http_argument, "Argument.metadata", "")  # Metadata should be empty for raw body

        return sampler

    def add_csv_data_config(self, parent_element: ET.Element, filename: str, variable_names: str, delimiter: str = ",",
                            quoted_data: bool = False) -> ET.Element:
        """Adds a CSV Data Set Config element."""
        csv_config_element = self._create_element(parent_element, "CSVDataSet", attrib={
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

    def add_constant_timer(self, parent_element: ET.Element, delay_ms: int, name: str = "Constant Timer") -> ET.Element:
        """Adds a Constant Timer element."""
        timer = self._create_element(parent_element, "ConstantTimer", attrib={
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": name,
            "enabled": "true"
        })
        self._create_string_prop(timer, "ConstantTimer.delay", str(delay_ms))

        return timer

    def add_response_assertion(self, parent_element: ET.Element, name: str, response_field: str, test_type: str,
                               pattern: str) -> ET.Element:
        """Adds a Response Assertion element."""
        assertion = self._create_element(parent_element, "ResponseAssertion", attrib={
            "guiclass": "AssertionGui",
            "testclass": "ResponseAssertion",
            "testname": name,
            "enabled": "true"
        })

        # CORRECTED: Use "Asserion.test_strings" as per user's specific feedback for JMeter compatibility.
        test_strings_prop = self._create_collection_prop(assertion, "Asserion.test_strings")
        # As per JMeter's default behavior, the name attribute often matches the pattern here.
        str_prop = self._create_element(test_strings_prop, "stringProp", attrib={"name": pattern})
        str_prop.text = pattern

        self._create_string_prop(assertion, "Assertion.custom_message", "")
        self._create_string_prop(assertion, "Assertion.test_field", response_field)
        self._create_int_prop(assertion, "Assertion.test_type", int(test_type))  # Changed to intProp
        self._create_bool_prop(assertion, "Assertion.assume_success", False)
        # Removed Assertion.scope (defaults to 'Main sample and sub-samples' when omitted)
        # Removed Assertion.pattern_mode (defaults to 'Substring'/'Equals' when omitted)

        self._create_bool_prop(assertion, "Assertion.override_existing_properties", False)

        return assertion

    def add_json_extractor(self, parent_element: ET.Element, name: str, json_path_expr: str, var_name: str,
                           match_no: str = "1",
                           default_value: str = "NOT_FOUND") -> ET.Element:
        """Adds a JSON Extractor element."""
        extractor = self._create_element(parent_element, "JSONPostProcessor", attrib={
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
                           name: str = "HTTP Header Manager") -> ET.Element:
        """Adds an HTTP Header Manager element."""
        header_manager = self._create_element(parent_element, "HeaderManager", attrib={
            "guiclass": "HeaderPanel",
            "testclass": "HeaderManager",
            "testname": name,
            "enabled": "true"
        })
        header_collection_prop = self._create_collection_prop(header_manager, "HeaderManager.headers")
        for header_name, header_value in headers.items():
            header_element = self._create_element(header_collection_prop, "elementProp", attrib={
                "name": "",
                "elementType": "Header"
            })
            self._create_string_prop(header_element, "Header.name", header_name)
            self._create_string_prop(header_element, "Header.value", header_value)

        return header_element

    def add_cookie_manager(self, parent_element: ET.Element, name: str = "HTTP Cookie Manager") -> ET.Element:
        """Adds an HTTP Cookie Manager element."""
        cookie_manager = self._create_element(parent_element, "CookieManager", attrib={
            "guiclass": "CookiePanel",
            "testclass": "CookieManager",
            "testname": name,
            "enabled": "true"
        })
        self._create_collection_prop(cookie_manager, "CookieManager.cookies")
        self._create_bool_prop(cookie_manager, "CookieManager.clearEachIteration", True)

        return cookie_manager

    def add_view_results_tree_listener(self, parent_element: ET.Element, name: str = "View Results Tree") -> ET.Element:
        """Adds a View Results Tree Listener."""
        listener = self._create_element(parent_element, "ResultCollector", attrib={
            "guiclass": "ViewResultsFullVisualizer",
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        self._create_element(obj_prop, "name").text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", attrib={"class": "SampleSaveConfiguration"})
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

    def add_summary_report_listener(self, parent_element: ET.Element, name: str = "Summary Report") -> ET.Element:
        """Adds a Summary Report Listener."""
        listener = self._create_element(parent_element, "ResultCollector", attrib={
            "guiclass": "SummaryReport",
            "testclass": "ResultCollector",
            "testname": name,
            "enabled": "true"
        })
        self._create_bool_prop(listener, "ResultCollector.error_logging", False)
        obj_prop = self._create_element(listener, "objProp")
        self._create_element(obj_prop, "name").text = "saveConfig"

        value_elem = self._create_element(obj_prop, "value", attrib={"class": "SampleSaveConfiguration"})
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

    def generate_jmx(self, app_base_url: str, thread_group_users: int, ramp_up_time: int, loop_count: int,
                     scenario_plan: Dict[str, List[Dict[str, Any]]],
                     test_plan_name: str,  # Added parameter
                     thread_group_name: str,  # Added parameter
                     csv_data: Optional[str] = None,
                     global_constant_timer_delay: int = 0,
                     database_connector: Any = None,
                     db_tables_schema: Optional[Dict[str, List[Dict[str, str]]]] = None) -> (str, Optional[str]):
        """
        Generates a JMeter JMX script based on the provided parameters and scenario plan.
        """

        # Pass custom names to __init__
        self.__init__(test_plan_name=test_plan_name, thread_group_name=thread_group_name)

        # 1. Add HTTP Request Defaults
        http_defaults = self.add_http_request_defaults(
            parent=self.test_plan_global_hashtree,
            protocol=urlparse(app_base_url).scheme,
            domain=urlparse(app_base_url).hostname,
            port=str(urlparse(app_base_url).port) if urlparse(app_base_url).port else "",
            connect_timeout="5000",
            response_timeout="10000"
        )
        self._create_element(self.test_plan_global_hashtree, "hashTree")  # Sibling hashTree for HTTP Defaults

        # 2. Add Thread Group
        thread_group_element = self.add_thread_group(
            num_users=thread_group_users,
            ramp_up_time=ramp_up_time,
            loop_count=loop_count,
            parent_element=self.test_plan_global_hashtree
        )
        # Create the SIBLING hashTree for the ThreadGroup.
        # This is the hashTree where elements like ConstantTimer, HTTPSamplerProxy, etc., for THIS thread group will be placed.
        thread_group_elements_parent_hashtree = self._create_element(self.test_plan_global_hashtree, "hashTree")

        generated_csv_content = None

        # 3. CSV Data Set Config (Conditional logic)
        current_csv_headers = []
        if database_connector and db_tables_schema:
            unique_mapped_columns = set()
            for request_data in scenario_plan['requests']:
                for param_name, param_value in request_data.get('parameters', {}).items():
                    if isinstance(param_value, str) and param_value.startswith("${") and param_value.endswith("}"):
                        db_col_ref = param_value[2:-1].replace('_', '.')
                        unique_mapped_columns.add(db_col_ref)

            extracted_csv_data = {}
            for col_ref in unique_mapped_columns:
                try:
                    table_name, column_name = col_ref.split('.')
                    if table_name in database_connector.get_tables():
                        df = database_connector.preview_data(table_name, limit=None)
                        if column_name in df.columns:
                            extracted_csv_data[col_ref] = df[column_name].tolist()
                            if col_ref not in current_csv_headers:
                                current_csv_headers.append(col_ref)
                except Exception as e:
                    logger.warning(f"Warning: Could not fetch data for {col_ref} from DB for CSV: {e}")

            if current_csv_headers and extracted_csv_data:
                csv_config_element = self.add_csv_data_config(
                    parent_element=thread_group_elements_parent_hashtree,  # Use the correct parent hashTree
                    filename="data.csv",
                    variable_names=",".join([col.replace('.', '_') for col in current_csv_headers])
                )
                self._create_element(thread_group_elements_parent_hashtree,
                                     "hashTree")  # Sibling hashTree for CSV Config

                generated_csv_content = ",".join([col.replace('.', '_') for col in current_csv_headers]) + "\n"
                max_rows = 0
                if extracted_csv_data:
                    max_rows = max(len(v) for v in extracted_csv_data.values())

                for i in range(max_rows):
                    row_values = []
                    for header in current_csv_headers:
                        values = extracted_csv_data.get(header, [])
                        row_values.append(str(values[i]) if i < len(values) else "")
                    generated_csv_content += ",".join(row_values) + "\n"

        # 4. Add Global Constant Timer if enabled
        if global_constant_timer_delay > 0:
            self.add_constant_timer(thread_group_elements_parent_hashtree,
                                    global_constant_timer_delay)  # Use the correct parent hashTree
            self._create_element(thread_group_elements_parent_hashtree,
                                 "hashTree")  # Sibling hashTree for Constant Timer

        # 5. Add HTTP Cookie Manager (placed globally under TestPlan's hashTree)
        self.add_cookie_manager(self.test_plan_global_hashtree)
        self._create_element(self.test_plan_global_hashtree, "hashTree")  # Sibling hashTree for Cookie Manager

        # 6. Add HTTP Header Manager (Global, placed under TestPlan's hashTree)
        header_manager_global = self.add_header_manager(
            self.test_plan_global_hashtree,
            {
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            name="HTTP Header Manager (Global)"
        )
        self._create_element(self.test_plan_global_hashtree, "hashTree")  # Sibling hashTree for Global Header Manager

        # 7. Add each request from the scenario plan
        for request_data in scenario_plan.get("requests", []):
            name = request_data.get('name', 'HTTP Request')
            method = request_data.get('method', 'GET')
            path = request_data.get('path', '/')
            parameters = request_data.get('parameters', {})
            body = request_data.get('body')
            headers = request_data.get('headers', {})  # Get headers from scenario plan
            assertions = request_data.get('assertions', [])
            json_extractors = request_data.get('json_extractors', [])

            # Add Sampler itself
            current_sampler = self.add_http_sampler(
                thread_group_elements_parent_hashtree, name, method, path, parameters, body
                # Use the correct parent hashTree
            )
            # Create the SIBLING hashTree for the HTTPSamplerProxy.
            # This is the hashTree where elements like HeaderManager, ResponseAssertion, etc., for THIS sampler will be placed.
            sampler_elements_parent_hashtree = self._create_element(thread_group_elements_parent_hashtree, "hashTree")

            # Add Headers specific to this request (if any)
            if headers:
                self.add_header_manager(sampler_elements_parent_hashtree, headers, name=f"{name} - Header Manager")
                self._create_element(sampler_elements_parent_hashtree,
                                     "hashTree")  # Sibling hashTree for Header Manager

            # Add Assertions for this request
            for assertion in assertions:
                if assertion.get("type") == "Response Code" and assertion.get("value"):
                    self.add_response_assertion(sampler_elements_parent_hashtree, f"{name} - Response Code Assertion",
                                                "Assertion.response_code", "2", assertion["value"])
                    self._create_element(sampler_elements_parent_hashtree, "hashTree")  # Sibling hashTree for Assertion
                elif assertion.get("type") == "Response Body Contains" and assertion.get("value"):
                    self.add_response_assertion(sampler_elements_parent_hashtree,
                                                f"{name} - Response Body Contains Assertion",
                                                "Assertion.response_data", "2", assertion["value"])
                    self._create_element(sampler_elements_parent_hashtree, "hashTree")  # Sibling hashTree for Assertion

            # Add JSON Extractors for this request
            for extractor in json_extractors:
                if extractor.get("json_path_expr") and extractor.get("var_name"):
                    self.add_json_extractor(sampler_elements_parent_hashtree,
                                            f"{name} - Extract {extractor['var_name']}",
                                            extractor['json_path_expr'], extractor['var_name'])
                    self._create_element(sampler_elements_parent_hashtree, "hashTree")  # Sibling hashTree for Extractor

        # 8. Add Listeners to TestPlan's global hashTree
        self.add_view_results_tree_listener(self.test_plan_global_hashtree)
        self._create_element(self.test_plan_global_hashtree, "hashTree")  # Sibling hashTree for View Results Tree

        self.add_summary_report_listener(self.test_plan_global_hashtree)
        self._create_element(self.test_plan_global_hashtree, "hashTree")  # Sibling hashTree for Summary Report

        # Generate XML string
        rough_string = ET.tostring(self.test_plan_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_jmx = reparsed.toprettyxml(indent="  ")

        # JMeter JMX files often don't have the `DOCTYPE` declaration.
        final_jmx_content = pretty_jmx.split('?>\n', 1)[-1].strip()

        return final_jmx_content, generated_csv_content
