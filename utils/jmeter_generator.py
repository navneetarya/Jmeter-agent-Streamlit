import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Optional, Tuple
import re
import random
import string
import json
import logging
from urllib.parse import urlparse, quote_plus
import requests  # Import requests for API calls
import pandas as pd  # Import pandas for type hinting in _generate_jmx_from_llm_design

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    """
    Generates JMeter JMX scripts based on provided API endpoint configurations.
    """

    def __init__(self, test_plan_name: str = "Test Plan", thread_group_name: str = "Users"):
        """
        Initializes the JMeterScriptGenerator.
        Args:
            test_plan_name (str): The name for the JMeter Test Plan.
            thread_group_name (str): The name for the main Thread Group.
        """
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name

    @staticmethod
    def _create_element(parent: ET.Element, tag: str, attrib: Dict[str, str] = None, text: str = None) -> ET.Element:
        """Helper to create and append an XML element."""
        element = ET.SubElement(parent, tag, attrib=attrib if attrib is not None else {})
        if text is not None:
            element.text = text
        return element

    @staticmethod
    def _prettify_xml(elem: ET.Element) -> str:
        """Returns a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _add_http_request_defaults(self, test_plan_hash_tree: ET.Element, protocol: str, domain: str, port: str,
                                   base_path: str):
        """Adds HTTP Request Defaults to the Test Plan."""
        config_defaults_attrib = {
            "elementType": "HttpDefaults",
            "guiclass": "HttpDefaultsGui",
            "testclass": "HttpDefaults",
            "testname": "HTTP Request Defaults",
            "enabled": "true"
        }
        config_defaults = self._create_element(test_plan_hash_tree, "ConfigTestElement", config_defaults_attrib)

        # Add a hashTree for the HTTP Request Defaults
        self._create_element(test_plan_hash_tree, "hashTree")  # This hashTree follows the ConfigTestElement

        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.protocol"}, protocol)
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.domain"}, domain)
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.port"}, port)
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.path"}, base_path)
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.contentEncoding"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.proxyHost"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.proxyPort"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.proxyUser"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.proxyPass"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.connectTimeout"}, "")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.responseTimeout"}, "")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.follow_redirects"}, "true")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.auto_redirects"}, "false")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.use_keepalive"}, "true")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.DO_MULTIPART_POST"}, "false")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART"}, "false")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.image_parser"}, "false")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.concurrentDwn"}, "false")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.concurrentPool"}, "6")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.md5sum"}, "false")
        self._create_element(config_defaults, "intProp", {"name": "HTTPSampler.embedded_url_re"},
                             "0")  # Corrected type and value
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.implementation"}, "HttpClient4")

        # Add the HTTPsampler.Arguments element with its collectionProp to HTTP Request Defaults
        arguments_element = self._create_element(config_defaults, "elementProp",
                                                 {"name": "HTTPsampler.Arguments", "elementType": "Arguments"})
        self._create_element(arguments_element, "collectionProp", {"name": "Arguments.arguments"})

    def _add_http_request_sampler(self, parent_hash_tree: ET.Element, request_config: Dict[str, Any]):
        """
        Adds an HTTP Request Sampler to a Thread Group's hashTree,
        along with its associated Header Manager, Assertions, and Extractors
        within the sampler's own hashTree.
        """
        sampler_attrib = {
            "elementType": "HTTPSamplerProxy",
            "guiclass": "HttpTestSampleGui",
            "testclass": "HTTPSamplerProxy",
            "testname": request_config.get("name", "HTTP Request"),
            "enabled": "true"
        }
        sampler = self._create_element(parent_hash_tree, "HTTPSamplerProxy", sampler_attrib)

        # Create the hashTree for the sampler, and add components to it
        sampler_hash_tree = self._create_element(parent_hash_tree, "hashTree")

        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"},
                             "true" if request_config.get("body") else "false")

        # Always add the HTTPsampler.Arguments element with its collectionProp
        # This ensures the structure is consistent even if no params/body are present.
        arguments_element = self._create_element(sampler, "elementProp",
                                                 {"name": "HTTPsampler.Arguments", "elementType": "Arguments",
                                                  "guiclass": "HTTPArgumentsPanel", "testclass": "Arguments",
                                                  "testname": "User Defined Variables", "enabled": "true"})
        arguments_collection_prop = self._create_element(arguments_element, "collectionProp",
                                                         {"name": "Arguments.arguments"})

        # HTTP Arguments for body (if postBodyRaw is true) - now add to the *already existing* arguments_collection_prop
        if request_config.get("body"):
            arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                            {"name": "", "elementType": "HTTPArgument"})
            self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, "false")
            self._create_element(arg_elem, "stringProp", {"name": "Argument.value"},
                                 json.dumps(request_config["body"]) if isinstance(request_config["body"],
                                                                                  (dict, list)) else str(
                                     request_config["body"]))
            self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")
            self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.use_multipart"}, "false")

        # Query Parameters - now add to the *already existing* arguments_collection_prop
        if request_config.get("parameters"):
            for param_name, param_value in request_config["parameters"].items():
                arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                                {"name": param_name, "elementType": "HTTPArgument"})
                self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, "true")
                self._create_element(arg_elem, "stringProp", {"name": "Argument.value"}, str(param_value))
                self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")
                self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.use_multipart"}, "false")

        # Other sampler properties
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.domain"}, "")  # Use defaults
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.port"}, "")  # Use defaults
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.protocol"}, "")  # Use defaults
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.contentEncoding"}, "")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.path"}, request_config.get("path", "/"))
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.method"},
                             request_config.get("method", "GET").upper())
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.follow_redirects"}, "true")
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.auto_redirects"}, "false")
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.use_keepalive"}, "true")
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.DO_MULTIPART_POST"}, "false")
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.BROWSER_COMPATIBLE_MULTIPART"}, "false")
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.image_parser"}, "false")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.concurrentDwn"}, "false")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.concurrentPool"}, "6")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.connectTimeout"}, "")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.responseTimeout"}, "")
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.implementation"}, "HttpClient4")

        # HTTP Header Manager - now inside sampler_hash_tree
        if request_config.get("headers"):
            header_manager_attrib = {
                "elementType": "HeaderManager",
                "guiclass": "HeaderPanel",
                "testclass": "HeaderManager",
                "testname": "HTTP Header Manager",
                "enabled": "true"
            }
            header_manager = self._create_element(sampler_hash_tree, "HeaderManager", header_manager_attrib)
            collection_prop = self._create_element(header_manager, "collectionProp", {"name": "HeaderManager.headers"})
            for name, value in request_config["headers"].items():
                header_elem = self._create_element(collection_prop, "elementProp",
                                                   {"name": "", "elementType": "Header"})
                self._create_element(header_elem, "stringProp", {"name": "Header.name"}, name)
                self._create_element(header_elem, "stringProp", {"name": "Header.value"}, value)
            self._create_element(sampler_hash_tree, "hashTree")  # ADDED: hashTree for HeaderManager

        # JSON Extractor - now inside sampler_hash_tree
        if request_config.get("json_extractors"):
            for i, extractor in enumerate(request_config["json_extractors"]):
                extractor_attrib = {
                    "elementType": "JSONPostProcessor",
                    "guiclass": "JSONPostProcessorGui",
                    "testclass": "JSONPostProcessor",
                    "testname": f"JSON Extractor - {extractor.get('var_name', f'Var{i + 1}')}",
                    "enabled": "true"
                }
                json_extractor = self._create_element(sampler_hash_tree, "JSONPostProcessor", extractor_attrib)
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.referenceNames"},
                                     extractor["var_name"])
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.jsonPathExprs"},
                                     extractor["json_path_expr"])
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.defaultValues"},
                                     "NOT_FOUND")
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.matchNumbers"}, "1")
                self._create_element(json_extractor, "boolProp", {"name": "JSONPostProcessor.is=ContainedText"},
                                     "false")
                self._create_element(sampler_hash_tree, "hashTree")  # ADDED: hashTree for JSONPostProcessor

        # Assertions - now inside sampler_hash_tree
        if request_config.get("assertions"):
            for assertion_config in request_config["assertions"]:
                if assertion_config["type"] == "Response Code":
                    response_assertion_attrib = {
                        "elementType": "ResponseAssertion",
                        "guiclass": "AssertionGui",
                        "testclass": "ResponseAssertion",
                        "testname": "Response Code Assertion",
                        "enabled": "true"
                    }
                    response_assertion = self._create_element(sampler_hash_tree, "ResponseAssertion",
                                                              response_assertion_attrib)
                    string_prop_field = self._create_element(response_assertion, "collectionProp",
                                                             {"name": "Asserion.test_strings"})
                    self._create_element(string_prop_field, "stringProp", {"name": "200"},
                                         assertion_config["value"])  # Assuming value is the code
                    self._create_element(response_assertion, "stringProp", {"name": "Assertion.test_field"},
                                         "Assertion.response_code")
                    self._create_element(response_assertion, "boolProp", {"name": "Assertion.assume_success"}, "false")
                    self._create_element(response_assertion, "intProp", {"name": "Assertion.test_type"}, "16")  # Equals
                    self._create_element(sampler_hash_tree, "hashTree")  # ADDED: hashTree for ResponseAssertion
                elif assertion_config["type"] == "Response Body Contains":
                    response_assertion_attrib = {
                        "elementType": "ResponseAssertion",
                        "guiclass": "AssertionGui",
                        "testclass": "ResponseAssertion",
                        "testname": "Response Body Assertion",
                        "enabled": "true"
                    }
                    response_assertion = self._create_element(sampler_hash_tree, "ResponseAssertion",
                                                              response_assertion_attrib)
                    string_prop_field = self._create_element(response_assertion, "collectionProp",
                                                             {"name": "Asserion.test_strings"})
                    self._create_element(string_prop_field, "stringProp", {},
                                         assertion_config["value"])  # Value is the pattern
                    self._create_element(response_assertion, "stringProp", {"name": "Assertion.test_field"},
                                         "Assertion.response_data")  # Response Body
                    self._create_element(response_assertion, "boolProp", {"name": "Assertion.assume_success"}, "false")
                    self._create_element(response_assertion, "intProp", {"name": "Assertion.test_type"},
                                         "2")  # Contains
                    self._create_element(sampler_hash_tree, "hashTree")  # ADDED: hashTree for ResponseAssertion

    def _add_csv_data_set_config(self, parent_hash_tree: ET.Element, csv_config: Dict[str, Any]):
        """Adds a CSV Data Set Config to the Test Plan."""
        config_element_attrib = {
            "elementType": "CSVDataSet",
            "guiclass": "TestBeanGUI",
            "testclass": "CSVDataSet",
            "testname": f"CSV Data Set Config - {csv_config['filename']}",
            "enabled": "true"
        }
        csv_data_set = self._create_element(parent_hash_tree, "CSVDataSet", config_element_attrib)
        # Add a hashTree for the CSV Data Set Config
        self._create_element(parent_hash_tree, "hashTree")  # This hashTree follows the CSVDataSet

        self._create_element(csv_data_set, "stringProp", {"name": "filename"}, csv_config["filename"])
        self._create_element(csv_data_set, "stringProp", {"name": "fileEncoding"}, "UTF-8")
        self._create_element(csv_data_set, "stringProp", {"name": "variableNames"},
                             ",".join(csv_config["variable_names"]))
        self._create_element(csv_data_set, "boolProp", {"name": "ignoreFirstLine"},
                             "true")  # Assuming headers are present
        self._create_element(csv_data_set, "boolProp", {"name": "quotedData"}, "false")
        self._create_element(csv_data_set, "boolProp", {"name": "recycle"}, "true")
        self._create_element(csv_data_set, "boolProp", {"name": "stopThread"}, "false")
        self._create_element(csv_data_set, "stringProp", {"name": "shareMode"}, "shareMode.all")
        self._create_element(csv_data_set, "stringProp", {"name": "delimiter"}, ",")

    def _add_constant_timer(self, parent_hash_tree: ET.Element, delay_ms: int):
        """Adds a Constant Timer to a Thread Group."""
        timer_attrib = {
            "elementType": "ConstantTimer",
            "guiclass": "ConstantTimerGui",
            "testclass": "ConstantTimer",
            "testname": f"Constant Timer - {delay_ms}ms",
            "enabled": "true"
        }
        timer = self._create_element(parent_hash_tree, "ConstantTimer", timer_attrib)
        # Add a hashTree for the Constant Timer
        self._create_element(parent_hash_tree, "hashTree")  # This hashTree follows the ConstantTimer

        self._create_element(timer, "stringProp", {"name": "ConstantTimer.delay"}, str(delay_ms))

    def _add_result_collector(self, parent_hash_tree: ET.Element, guiclass: str, testname: str, enabled: bool,
                              filename: str = "", save_config_params: Dict[str, Any] = None):
        """Adds a ResultCollector listener to the Test Plan with configurable save options."""
        listener_attrib = {
            "elementType": "ResultCollector",
            "guiclass": guiclass,
            "testclass": "ResultCollector",
            "testname": testname,
            "enabled": str(enabled).lower()  # JMeter expects "true" or "false" string
        }
        listener = self._create_element(parent_hash_tree, "ResultCollector", listener_attrib)
        self._create_element(parent_hash_tree, "hashTree")  # Each component needs a hashTree

        self._create_element(listener, "boolProp", {"name": "ResultCollector.error_logging"},
                             "false")  # Always false as per user examples

        obj_prop = self._create_element(listener, "objProp")
        self._create_element(obj_prop, "name", text="saveConfig")
        value_elem = self._create_element(obj_prop, "value", attrib={"class": "SampleSaveConfiguration"})

        # Define all possible SampleSaveConfiguration properties and their default values
        # Default to False for booleans, 0 for int if not explicitly provided
        all_save_config_properties = {
            "time": False, "latency": False, "timestamp": False, "success": False, "label": False,
            "code": False, "message": False, "threadName": False, "dataType": False, "encoding": False,
            "assertions": False, "subresults": False, "responseData": False, "samplerData": False,
            "xml": False, "fieldNames": False, "responseHeaders": False, "requestHeaders": False,
            "responseDataOnError": False, "saveAssertionResultsFailureMessage": False,
            "assertionsResultsToSave": 0,  # Integer
            "bytes": False, "sentBytes": False, "url": False, "threadCounts": False, "idleTime": False,
            "connectTime": False
        }

        # Update with provided custom parameters, overwriting defaults
        if save_config_params:
            for key, val in save_config_params.items():
                if key in all_save_config_properties:  # Only update if key is valid
                    all_save_config_properties[key] = val

        # Add properties to the value_elem
        for key, val in all_save_config_properties.items():
            if isinstance(val, bool):
                # JMeter expects boolean values as direct child tags like <time>true</time>
                self._create_element(value_elem, key, text=str(val).lower())
            elif isinstance(val, int):
                # JMeter expects integer values as direct child tags like <assertionsResultsToSave>0</assertionsResultsToSave>
                self._create_element(value_elem, key, text=str(val))

        self._create_element(listener, "stringProp", {"name": "filename"}, filename)

    def generate_jmx(
            self,
            app_base_url: str,
            thread_group_users: int,
            ramp_up_time: int,
            loop_count: int,
            scenario_plan: Dict[str, Any],  # This is the internal representation after LLM/manual processing
            csv_configs: Optional[List[Dict[str, Any]]] = None,
            global_constant_timer_delay: int = 0,
            test_plan_name: str = "Generated Test Plan",
            thread_group_name: str = "Users",
            http_defaults_protocol: str = "https",
            http_defaults_domain: str = "example.com",
            http_defaults_port: str = "",
            http_defaults_base_path: str = "/",
            full_swagger_spec: Optional[Dict[str, Any]] = None,
            enable_setup_teardown_thread_groups: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Generates the full JMeter JMX XML string.

        Args:
            app_base_url (str): The base URL of the application under test (for reference, defaults are used for HTTP defaults).
            thread_group_users (int): Number of concurrent users.
            ramp_up_time (int): Ramp-up period in seconds.
            loop_count (int): Number of loops (-1 for infinite).
            scenario_plan (Dict[str, Any]): A dictionary containing the 'requests' list, where each request is a dict.
            csv_configs (Optional[List[Dict[str, Any]]]): List of dictionaries for CSV Data Set Config.
                                                         Each dict should have 'filename' and 'variable_names'.
            global_constant_timer_delay (int): Delay in milliseconds for a global constant timer.
            test_plan_name (str): Name for the JMeter Test Plan.
            thread_group_name (str): Name for the main Thread Group.
            http_defaults_protocol (str): Protocol for HTTP Request Defaults (e.g., "https").
            http_defaults_domain (str): Domain for HTTP Request Defaults (e.g., "api.example.com").
            http_defaults_port (str): Port for HTTP Request Defaults (e.g., "443" or empty).
            http_defaults_base_path (str): Base path for HTTP Request Defaults (e.g., "/api/v1").
            full_swagger_spec (Optional[Dict[str, Any]]): The full Swagger/OpenAPI spec dictionary.
            enable_setup_teardown_thread_groups (bool): Whether to include Setup/Teardown Thread Groups.

        Returns:
            Tuple[str, Optional[str]]: The JMX XML string and the CSV data string if generated, else None.
        """
        jmx_template_root = ET.Element("jmeterTestPlan", {
            "version": "1.2",
            "properties": "5.0",
            "jmeter": "5.6.2"  # Updated JMeter version
        })

        # This is the top-level hashTree immediately under jmeterTestPlan
        root_hash_tree = self._create_element(jmx_template_root, "hashTree")

        test_plan_attrib = {
            "elementType": "TestPlan",
            "guiclass": "TestPlanGui",
            "testclass": "TestPlan",
            "testname": test_plan_name,
            "enabled": "true"
        }
        # TestPlan is a direct child of root_hash_tree
        test_plan = self._create_element(root_hash_tree, "TestPlan", test_plan_attrib)
        self._create_element(test_plan, "stringProp", {"name": "TestPlan.comments"}, "")
        self._create_element(test_plan, "boolProp", {"name": "TestPlan.functional_mode"}, "false")
        self._create_element(test_plan, "boolProp", {"name": "TestPlan.tearDown_on_shutdown"}, "true")
        self._create_element(test_plan, "boolProp", {"name": "TestPlan.serialize_threadgroups"}, "false")

        # This is the hashTree that contains all the elements belonging to the Test Plan.
        # It must be a sibling to the TestPlan element, under the same parent (root_hash_tree).
        test_plan_elements_hash_tree = self._create_element(root_hash_tree, "hashTree")

        # All subsequent calls to add elements must use 'test_plan_elements_hash_tree' as their parent_hash_tree.

        # HTTP Request Defaults (global)
        self._add_http_request_defaults(
            test_plan_elements_hash_tree,  # Corrected parent
            http_defaults_protocol,
            http_defaults_domain,
            http_defaults_port,
            http_defaults_base_path
        )

        # Setup Thread Group
        if enable_setup_teardown_thread_groups:
            setup_thread_group_attrib = {
                "elementType": "SetupThreadGroup",
                "guiclass": "SetupThreadGroupGui",
                "testclass": "SetupThreadGroup",
                "testname": "setUp Thread Group",
                "enabled": "true"
            }
            setup_thread_group = self._create_element(test_plan_elements_hash_tree, "SetupThreadGroup",
                                                      setup_thread_group_attrib)  # Corrected parent
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}, "1")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}, "1")
            self._create_element(setup_thread_group, "boolProp", {"name": "ThreadGroup.main_controller"}, "true")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.scheduler"}, "false")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.duration"}, "")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.delay"}, "")
            self._create_element(setup_thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"},
                                 "true")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.start_time"}, "")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.end_time"}, "")
            self._create_element(setup_thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}, "continue")
            self._create_element(setup_thread_group, "intProp", {"name": "ThreadGroup.action_on_err"}, "0")

            self._create_element(test_plan_elements_hash_tree, "hashTree")  # This hashTree follows the SetupThreadGroup

        # Main Thread Group
        thread_group_attrib = {
            "elementType": "ThreadGroup",
            "guiclass": "ThreadGroupGui",
            "testclass": "ThreadGroup",
            "testname": thread_group_name,
            "enabled": "true"
        }
        thread_group = self._create_element(test_plan_elements_hash_tree, "ThreadGroup",
                                            thread_group_attrib)  # Corrected parent
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}, "continue")
        self._create_element(thread_group, "elementProp",
                             {"name": "ThreadGroup.main_controller", "elementType": "LoopController",
                              "guiclass": "LoopControlPanel", "testclass": "LoopController",
                              "testname": "Loop Controller", "enabled": "true"})
        loop_controller = thread_group.find("elementProp")
        self._create_element(loop_controller, "boolProp", {"name": "LoopController.continue_forever"},
                             "true" if loop_count == -1 else "false")
        self._create_element(loop_controller, "stringProp", {"name": "LoopController.loops"},
                             str(loop_count) if loop_count != -1 else "")
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}, str(thread_group_users))
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}, str(ramp_up_time))
        self._create_element(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}, "false")
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.duration"}, "")
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.delay"}, "")
        self._create_element(thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"}, "true")
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.start_time"}, "")
        self._create_element(thread_group, "stringProp", {"name": "ThreadGroup.end_time"}, "")

        thread_group_hash_tree = self._create_element(test_plan_elements_hash_tree,
                                                      "hashTree")  # This hashTree follows the ThreadGroup

        # Add CSV Data Set Configs
        if csv_configs:
            for csv_config in csv_configs:
                self._add_csv_data_set_config(thread_group_hash_tree, csv_config)

        # Add Global Constant Timer
        if global_constant_timer_delay > 0:
            self._add_constant_timer(thread_group_hash_tree, global_constant_timer_delay)

        # Add HTTP Request Samplers
        for request_config in scenario_plan.get("requests", []):
            self._add_http_request_sampler(thread_group_hash_tree, request_config)

        # Teardown Thread Group
        if enable_setup_teardown_thread_groups:
            teardown_thread_group_attrib = {
                "elementType": "PostThreadGroup",
                "guiclass": "PostThreadGroupGui",
                "testclass": "PostThreadGroup",
                "testname": "tearDown Thread Group",
                "enabled": "true"
            }
            teardown_thread_group = self._create_element(test_plan_elements_hash_tree, "PostThreadGroup",
                                                         teardown_thread_group_attrib)  # Corrected parent
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}, "1")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}, "1")
            self._create_element(teardown_thread_group, "boolProp", {"name": "ThreadGroup.main_controller"}, "true")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.scheduler"}, "false")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.duration"}, "")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.delay"}, "")
            self._create_element(teardown_thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"},
                                 "true")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.start_time"}, "")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.end_time"}, "")
            self._create_element(teardown_thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"},
                                 "continue")
            self._create_element(teardown_thread_group, "intProp", {"name": "ThreadGroup.action_on_err"}, "0")

            self._create_element(test_plan_elements_hash_tree, "hashTree")  # This hashTree follows the PostThreadGroup

        # Add Result Collectors (Listeners) as specified by the user

        # 1. View Results in Table
        self._add_result_collector(
            test_plan_elements_hash_tree,  # Corrected parent
            guiclass="TableVisualizer",
            testname="View Results in Table",
            enabled=True,
            save_config_params={
                "time": True, "latency": True, "timestamp": True, "success": True, "label": True,
               "assertions": True, "subresults": True, "responseData": False, "samplerData": False,
                "xml": False, "fieldNames": True, "responseHeaders": False, "requestHeaders": False,
                "responseDataOnError": False, "saveAssertionResultsFailureMessage": True,
                "assertionsResultsToSave": 0, "bytes": True, "sentBytes": True, "url": True,
                "threadCounts": True, "idleTime": True, "connectTime": True
            }
        )

        # 2. View Results Tree
        self._add_result_collector(
            test_plan_elements_hash_tree,  # Corrected parent
            guiclass="ViewResultsFullVisualizer",
            testname="View Results Tree",
            enabled=True,
            save_config_params={
                "time": True, "latency": True, "timestamp": True, "success": True, "label": True,
                "code": True, "message": True, "threadName": True, "dataType": True, "encoding": False,
                "assertions": True, "subresults": True, "responseData": False, "samplerData": False,
                "xml": False, "fieldNames": False, "responseHeaders": False, "requestHeaders": False,
                "responseDataOnError": False, "saveAssertionResultsFailureMessage": False,
                "assertionsResultsToSave": 0, "bytes": True, "threadCounts": True
                # idleTime and connectTime default to False
            }
        )

        # 3. Aggregate Report
        self._add_result_collector(
            test_plan_elements_hash_tree,  # Corrected parent
            guiclass="StatVisualizer",
            testname="Aggregate Report",
            enabled=True,
            save_config_params={
                "time": True, "latency": True, "timestamp": True, "success": True, "label": True,
                "code": True, "message": True, "threadName": True, "dataType": True, "encoding": False,
                "assertions": True, "subresults": True, "responseData": False, "samplerData": False,
                "xml": False, "fieldNames": True, "responseHeaders": False, "requestHeaders": False,
                "responseDataOnError": False, "saveAssertionResultsFailureMessage": True,
                "assertionsResultsToSave": 0, "bytes": True, "sentBytes": True, "url": True,
                "threadCounts": True, "idleTime": True, "connectTime": True
            }
        )

        pretty_xml = self._prettify_xml(jmx_template_root)
        return pretty_xml, None  # Returning None for CSV data as it's handled separately

    @classmethod
    def generate_jmx_from_llm(  # Renamed to fit the user's existing function name
            cls,
            llm_structured_scenario: Dict[str, Any],
            swagger_endpoints: List[Any],  # Use Any for now to avoid circular import with SwaggerEndpoint
            db_tables_schema: Dict[str, List[Dict[str, Any]]],
            db_sampled_data: Dict[str, pd.DataFrame],
            test_plan_name: str,
            thread_group_name: str,
            num_users: int,
            ramp_up_time: int,
            loop_count: int,
            global_constant_timer_delay: int,
            enable_auth_flow: bool,
            auth_login_endpoint_path: str,
            auth_login_method: str,
            auth_login_body_template: str,
            auth_token_json_path: str,
            auth_header_name: str,
            auth_header_prefix: str,
            full_swagger_spec: Dict[str, Any],
            enable_setup_teardown_thread_groups: bool,
            current_swagger_url: str  # Added for base URL derivation
    ) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[str]]:
        """
        Processes the LLM's structured scenario design and generates the JMeter JMX file,
        CSV data, and mapping metadata.

        Args:
            llm_structured_scenario (Dict[str, Any]): The structured JSON output from the LLM.
            swagger_endpoints (List[Any]): List of SwaggerEndpoint objects.
            db_tables_schema (Dict[str, List[Dict[str, Any]]]): Schema of connected database tables.
            db_sampled_data (Dict[str, pd.DataFrame]): Sampled data from database tables.
            test_plan_name (str): Name for the JMeter Test Plan.
            thread_group_name (str): Name for the main Thread Group.
            num_users (int): Number of concurrent users.
            ramp_up_time (int): Ramp-up period in seconds.
            loop_count (int): Number of loops (-1 for infinite).
            global_constant_timer_delay (int): Delay in milliseconds for a global constant timer.
            enable_auth_flow (bool): Flag indicating if authentication flow is enabled.
            auth_login_endpoint_path (str): Path of the login API endpoint.
            auth_login_method (str): HTTP method of the login API.
            auth_login_body_template (str): Template for the login request body.
            auth_token_json_path (str): JSONPath for extracting auth token.
            auth_header_name (str): Name of the Authorization header.
            auth_header_prefix (str): Prefix for the Authorization header (e.g., "Bearer ").
            full_swagger_spec (Dict[str, Any]): The full Swagger/OpenAPI spec dictionary.
            enable_setup_teardown_thread_groups (bool): Whether to include Setup/Teardown Thread Groups.
            current_swagger_url (str): The URL of the currently loaded Swagger spec, used for defaults.

        Returns:
            Tuple[Optional[str], Optional[Dict[str, str]], Optional[str]]:
            A tuple containing:
            - The generated JMX XML string.
            - A dictionary mapping CSV filenames to their content.
            - The JSON string of the mapping metadata.
        """
        from utils.data_mapper import DataMapper  # Import here to avoid circular dependency
        from utils.swagger_parser import SwaggerEndpoint  # Import here for type hinting

        jmx_content = None
        downloadable_csv_contents = {}
        mapping_metadata_download = None
        scenario_requests_configs = []
        extracted_variables_map = {}

        # If AI-structured scenario exists, use it as the base for scenario requests
        if llm_structured_scenario and 'test_plan' in llm_structured_scenario:
            llm_response_test_plan = llm_structured_scenario['test_plan']
            http_samplers_list = llm_response_test_plan.get('http_samplers', [])

            # Generate mappings based on current swagger and DB data
            # This is crucial for CSV data generation
            mappings = DataMapper.suggest_mappings(
                swagger_endpoints,
                db_tables_schema,
                db_sampled_data
            )
            mapping_metadata_download = json.dumps(mappings, indent=2)

            # Extract base URL components from Swagger spec or fallback
            protocol = "https"
            domain = "example.com"
            port = ""
            base_path_for_http_defaults = "/"

            if full_swagger_spec:
                parsed_url_from_swagger_data = urlparse(full_swagger_spec.get('host', current_swagger_url))
                protocol = full_swagger_spec.get('schemes', ['https'])[0]
                domain = parsed_url_from_swagger_data.hostname
                port = parsed_url_from_swagger_data.port if parsed_url_from_swagger_data.port else ""
                base_path_for_http_defaults = full_swagger_spec.get('basePath', '/')
                if not base_path_for_http_defaults.startswith('/'):
                    base_path_for_http_defaults = '/' + base_path_for_http_defaults
                if base_path_for_http_defaults != '/' and base_path_for_http_defaults.endswith('/'):
                    base_path_for_http_defaults = base_path_for_http_defaults.rstrip('/')
            else:
                parsed_url = urlparse(current_swagger_url)
                protocol = parsed_url.scheme
                domain = parsed_url.hostname
                port = parsed_url.port if parsed_url.port else ""
                # base_path_for_http_defaults remains "/" if no basePath in spec

            for http_sampler in http_samplers_list:
                method_str = http_sampler.get('method')
                path_from_llm = http_sampler.get('path')
                request_name = http_sampler.get('name', f"{method_str}_{path_from_llm.replace('/', '_').strip('_')}")

                # Match LLM's path to an actual SwaggerEndpoint object to get its original template path
                resolved_endpoint_obj, _ = JMeterScriptGenerator._match_swagger_path_with_generated_path(
                    swagger_endpoints, method_str, path_from_llm
                )

                if not resolved_endpoint_obj:
                    # Log a warning, but for robustness, create a basic config if endpoint not perfectly matched
                    print(
                        f"Warning: AI suggested endpoint {method_str} {path_from_llm} not perfectly matched in Swagger spec. Using LLM's path directly.")
                    # Fallback to LLM's path if no exact match
                    jmeter_formatted_path = path_from_llm
                    # Dummy resolved_endpoint_data if not found, to avoid breaking
                    resolved_endpoint_data = {"method": method_str, "path": path_from_llm}
                else:
                    resolved_endpoint_data = resolved_endpoint_obj.to_dict()
                    # Use the template path from resolved_endpoint_obj for JMeter generation
                    jmeter_formatted_path = resolved_endpoint_obj.path

                request_config = {
                    "endpoint_key": f"{method_str} {resolved_endpoint_obj.path if resolved_endpoint_obj else path_from_llm}",
                    "name": request_name,
                    "method": method_str,
                    "path": "",  # Will be set below
                    "parameters": {},
                    "headers": {},  # Initialize headers here
                    "body": None,
                    "assertions": [],  # Initialize assertions here
                    "json_extractors": [],
                    "think_time": 0
                }

                # Path parameters: LLM provides values; substitute into the template path
                for pp in http_sampler.get('path_params', []):
                    param_name = pp['name']
                    param_value = pp['value']
                    jmeter_formatted_path = jmeter_formatted_path.replace(f"{{{param_name}}}", param_value)

                # Set the final path for the sampler, handling base_path
                final_request_path_for_jmeter = jmeter_formatted_path
                if base_path_for_http_defaults != '/' and final_request_path_for_jmeter.startswith(
                        base_path_for_http_defaults):
                    final_request_path_for_jmeter = final_request_path_for_jmeter[len(base_path_for_http_defaults):]
                    if not final_request_path_for_jmeter.startswith('/'):
                        final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                    if final_request_path_for_jmeter == "//":
                        final_request_path_for_jmeter = "/"
                if final_request_path_for_jmeter and not final_request_path_for_jmeter.startswith('/'):
                    final_request_path_for_jmeter = '/' + final_request_path_for_jmeter
                request_config["path"] = final_request_path_for_jmeter

                # Query parameters
                for qp in http_sampler.get('query_params', []):
                    request_config['parameters'][qp['name']] = qp['value']

                # Headers
                for header in http_sampler.get('headers', []):
                    request_config['headers'][header['name']] = header['value']

                # Authentication Header injection (if enabled and not login request)
                if enable_auth_flow and request_config[
                    "endpoint_key"] != f"{auth_login_method} {auth_login_endpoint_path}":
                    auth_header_name_lower = auth_header_name.lower()
                    if not any(h_name.lower() == auth_header_name_lower for h_name in request_config['headers'].keys()):
                        if "auth_token" in extracted_variables_map:
                            request_config["headers"][
                                auth_header_name] = f"{auth_header_prefix}{extracted_variables_map['auth_token']}"
                        else:
                            request_config["headers"][auth_header_name] = f"{auth_header_prefix}<<AUTH_TOKEN_MISSING>>"
                            print(
                                f"Warning: Auth flow enabled for {request_config['endpoint_key']}, but auth token not found/extracted. Using placeholder.")

                # Body
                if 'body' in http_sampler and http_sampler['body'] is not None:
                    if isinstance(http_sampler['body'], str):
                        try:
                            request_config["body"] = json.loads(http_sampler['body'])
                        except json.JSONDecodeError:
                            request_config["body"] = http_sampler['body']
                    else:
                        request_config["body"] = http_sampler['body']

                    if 'content_type' in http_sampler and http_sampler['content_type']:
                        request_config["headers"]["Content-Type"] = http_sampler['content_type']
                    elif 'Content-Type' not in request_config['headers']:
                        request_config["headers"]["Content-Type"] = "application/json"  # Default for JSON body

                # Assertions
                for assertion_data in http_sampler.get('assertions', []):
                    if assertion_data['type'] == 'response_code':
                        request_config['assertions'].append(
                            {"type": "Response Code", "value": assertion_data['pattern']})
                    elif assertion_data[
                        'type'] == "text_response":  # Changed from "text_response" to "Response Body Contains"
                        request_config['assertions'].append(
                            {"type": "Response Body Contains", "value": assertion_data['pattern']})
                    else:
                        request_config['assertions'].append(assertion_data)

                # Extractions
                for extractor_data in http_sampler.get('extractions', []):
                    if isinstance(extractor_data, dict):
                        json_path_expr = extractor_data.get('json_path')
                        var_name = extractor_data.get('var_name')
                        if json_path_expr and var_name:
                            request_config['json_extractors'].append({
                                "json_path_expr": json_path_expr,
                                "var_name": var_name
                            })
                            extracted_variables_map[var_name.lower()] = f"${{{var_name}}}"
                        else:
                            print(f"Warning: Malformed extraction entry from LLM and skipped: {extractor_data}")
                    else:
                        print(f"Warning: Non-dictionary extraction entry from LLM found and skipped: {extractor_data}")

                scenario_requests_configs.append(request_config)

            # --- CSV Data Generation ---
            csv_configs_for_generator = []
            csv_data_for_jmeter = {}
            csv_headers = set()

            # Iterate through the LLM's structured scenario to identify CSV needs
            # The LLM's `csv_data_set_config` is the primary source for CSV variable names
            llm_csv_config_from_response = llm_response_test_plan.get('csv_data_set_config', {})
            if isinstance(llm_csv_config_from_response, list) and llm_csv_config_from_response:
                llm_csv_config_from_response = llm_csv_config_from_response[0]  # Assume one main CSV config

            if llm_csv_config_from_response and llm_csv_config_from_response.get('variable_names'):
                # LLM's variable names are like "clientId", "newName"
                # We need to map them back to "csv_TableName_ColumnName" if they came from DB
                for var_name in llm_csv_config_from_response['variable_names']:
                    # Try to find a mapping that produced this variable name
                    found_mapping = False
                    for endpoint_key, params_map in mappings.items():
                        for param_path, mapping_info in params_map.items():
                            # The 'value' in mapping_info for CSV is already in the format "${csv_TableName_ColumnName}"
                            # We need to extract TableName_ColumnName from it to match var_name
                            if mapping_info['source'] == "DB Sample (CSV)" and \
                                    mapping_info['value'].replace('$', '').replace('{', '').replace('}', '').startswith(
                                        'csv_') and \
                                    mapping_info['value'].replace('$', '').replace('{', '').replace('}', '').split(
                                        'csv_', 1)[1] == var_name:

                                table_name_col_name = \
                                mapping_info['value'].replace('$', '').replace('{', '').replace('}', '').split('csv_',
                                                                                                               1)[1]
                                parts = table_name_col_name.split('_', 1)
                                if len(parts) == 2:
                                    table_name, column_name = parts
                                else:
                                    logger.warning(f"Could not parse table and column name from {table_name_col_name}")
                                    continue  # Skip if format is unexpected

                                if table_name in db_sampled_data and column_name in db_sampled_data[table_name].columns:
                                    jmeter_var_name_for_csv_header = var_name  # Use the LLM's provided variable name for the CSV header

                                    if jmeter_var_name_for_csv_header not in csv_data_for_jmeter:
                                        csv_data_for_jmeter[jmeter_var_name_for_csv_header] = \
                                        db_sampled_data[table_name][column_name].tolist()
                                        csv_headers.add(jmeter_var_name_for_csv_header)
                                    else:
                                        # Ensure the column has enough data if it's already added but from a different source
                                        if len(csv_data_for_jmeter[jmeter_var_name_for_csv_header]) < len(
                                                db_sampled_data[table_name][column_name]):
                                            csv_data_for_jmeter[jmeter_var_name_for_csv_header] = \
                                            db_sampled_data[table_name][column_name].tolist()
                                    found_mapping = True
                                    break
                        if found_mapping:
                            break
                    if not found_mapping:
                        logger.warning(
                            f"CSV variable '{var_name}' from LLM's design not found in DB Sample mappings. Will try to generate dummy if no data for it.")
                        # If a CSV variable is mentioned by LLM but no DB mapping, and it's used in a request,
                        # ensure it's still part of CSV headers. We can't populate its data from DB,
                        # but it might be manually provided or filled as empty.
                        csv_headers.add(var_name)  # Add the LLM's suggested var_name directly as a header

            if csv_headers and csv_data_for_jmeter:
                csv_headers_list = sorted(list(csv_headers))
                generated_csv_string = ",".join(csv_headers_list) + "\n"  # Use LLM's var names as headers

                max_rows = 0
                if csv_data_for_jmeter:
                    max_rows = max(len(v) for v in csv_data_for_jmeter.values())

                for i in range(max_rows):
                    row_values = []
                    for header_key in csv_headers_list:
                        values = csv_data_for_jmeter.get(header_key, [])
                        row_values.append(str(values[i]) if i < len(values) else "")
                    generated_csv_string += ",".join(row_values) + "\n"

                csv_configs_for_generator.append({
                    'filename': llm_csv_config_from_response.get('filename', "data.csv"),
                    'variable_names': csv_headers_list,  # Use the actual headers from LLM and DB mapping
                })
                downloadable_csv_contents[
                    llm_csv_config_from_response.get('filename', "data.csv")] = generated_csv_string
            elif llm_csv_config_from_response and llm_csv_config_from_response.get('variable_names'):
                # LLM indicated CSV variables, but no data from DB was mapped.
                # Still generate an empty CSV with headers.
                csv_headers_list_from_llm = llm_csv_config_from_response['variable_names']
                generated_csv_string = ",".join(csv_headers_list_from_llm) + "\n"  # Use LLM's var names as headers
                csv_configs_for_generator.append({
                    'filename': llm_csv_config_from_response.get('filename', "data.csv"),
                    'variable_names': csv_headers_list_from_llm,
                })
                downloadable_csv_contents[
                    llm_csv_config_from_response.get('filename', "data.csv")] = generated_csv_string

            # --- JMeter JMX Generation ---
            generator = cls(test_plan_name=test_plan_name, thread_group_name=thread_group_name)
            jmx_content, _ = generator.generate_jmx(
                app_base_url=current_swagger_url,
                thread_group_users=num_users,
                ramp_up_time=ramp_up_time,
                loop_count=loop_count,
                scenario_plan={"requests": scenario_requests_configs},  # Pass the constructed scenario configs
                csv_configs=csv_configs_for_generator,  # Pass the generated CSV configs
                global_constant_timer_delay=global_constant_timer_delay,
                test_plan_name=test_plan_name,
                thread_group_name=thread_group_name,
                http_defaults_protocol=protocol,
                http_defaults_domain=domain,
                http_defaults_port=port,
                http_defaults_base_path=base_path_for_http_defaults,
                full_swagger_spec=full_swagger_spec,
                enable_setup_teardown_thread_groups=enable_setup_teardown_thread_groups
            )

        return jmx_content, downloadable_csv_contents, mapping_metadata_download

    @staticmethod
    def _match_swagger_path_with_generated_path(swagger_endpoints: List[Any], method: str, generated_path: str) -> \
    Tuple[Optional[Any], Dict[str, str]]:
        """
        Attempts to match a generated path from LLM to an actual Swagger endpoint,
        extracting path parameters if the generated path contains concrete values.
        Returns the matched SwaggerEndpoint and a dict of extracted path params.

        Note: This is duplicated from app.py to avoid circular dependency for SwaggerEndpoint type.
        Consider refactoring common utility functions.
        """
        from utils.swagger_parser import SwaggerEndpoint  # Import locally to avoid circular dependency
        extracted_params = {}

        for ep in swagger_endpoints:
            if not isinstance(ep, SwaggerEndpoint):  # Ensure it's a SwaggerEndpoint object
                continue
            if ep.method.upper() != method.upper():
                continue

            swagger_path_regex_pattern = ep.path
            path_param_names = []

            for param in ep.parameters:
                if param.get('in') == 'path':
                    param_name = param['name']
                    swagger_path_regex_pattern = swagger_path_regex_pattern.replace(f"{{{param_name}}}",
                                                                                    f"(?P<{param_name}>[^/]+)")
                    path_param_names.append(param_name)

            swagger_path_regex_pattern = "^" + swagger_path_regex_pattern + "$"

            match = re.match(swagger_path_regex_pattern, generated_path)

            if match:
                for p_name in path_param_names:
                    if p_name in match.groupdict():
                        extracted_params[p_name] = match.group(p_name)
                return ep, extracted_params

        return None, {}
