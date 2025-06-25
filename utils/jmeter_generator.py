import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Tuple
import re
import json
import logging
import pandas as pd
import yaml
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
    """
    Translates a structured, pre-designed test plan JSON into a JMeter JMX file and associated CSVs.
    """

    def __init__(self, test_plan_name: str = "Test Plan", thread_group_name: str = "Users"):
        self.test_plan_name = test_plan_name
        self.thread_group_name = thread_group_name

    @staticmethod
    def _create_element(parent: ET.Element, tag: str, attrib: Dict[str, str] = None, text: str = None) -> ET.Element:
        element = ET.SubElement(parent, tag, attrib=attrib if attrib is not None else {})
        if text is not None:
            element.text = text
        return element

    @staticmethod
    def _prettify_xml(elem: ET.Element) -> str:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _add_http_request_defaults(self, parent_hash_tree: ET.Element, protocol: str, domain: str, port: str):
        config_defaults_attrib = {
            "elementType": "HttpDefaults", "guiclass": "HttpDefaultsGui", "testclass": "HttpDefaults",
            "testname": "HTTP Request Defaults", "enabled": "true"
        }
        config_defaults = self._create_element(parent_hash_tree, "ConfigTestElement", config_defaults_attrib)
        self._create_element(parent_hash_tree, "hashTree")

        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.protocol"}, protocol)
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.domain"}, domain)
        if port:
            self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.port"}, str(port))

        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.contentEncoding"}, "UTF-8")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.follow_redirects"}, "true")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.auto_redirects"}, "false")
        self._create_element(config_defaults, "boolProp", {"name": "HTTPSampler.use_keepalive"}, "true")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.implementation"}, "HttpClient4")

        arguments_element = self._create_element(config_defaults, "elementProp", {
            "name": "HTTPsampler.Arguments", "elementType": "Arguments", "guiclass": "HTTPArgumentsPanel",
            "testclass": "Arguments", "testname": "User Defined Variables", "enabled": "true"
        })
        self._create_element(arguments_element, "collectionProp", {"name": "Arguments.arguments"})

    def _add_http_request_sampler(self, parent_hash_tree: ET.Element, request_config: Dict[str, Any]):
        unique_name = request_config.get("name", f"{request_config.get('method')} {request_config.get('path')}")
        sampler_attrib = {"elementType": "HTTPSamplerProxy", "guiclass": "HttpTestSampleGui",
                          "testclass": "HTTPSamplerProxy", "testname": unique_name, "enabled": "true"}
        sampler = self._create_element(parent_hash_tree, "HTTPSamplerProxy", sampler_attrib)
        sampler_hash_tree = self._create_element(parent_hash_tree, "hashTree")

        # --- START: CORRECTED BODY/PARAMETER LOGIC ---
        has_body = request_config.get("body") is not None
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"}, "true" if has_body else "false")

        arguments_element = self._create_element(sampler, "elementProp",
                                                 {"name": "HTTPsampler.Arguments", "elementType": "Arguments",
                                                  "guiclass": "HTTPArgumentsPanel", "testclass": "Arguments",
                                                  "enabled": "true"})
        arguments_collection_prop = self._create_element(arguments_element, "collectionProp",
                                                         {"name": "Arguments.arguments"})
        # 1. Add Body data if it exists
        if has_body:
            arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                            {"name": "", "elementType": "HTTPArgument"})
            self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, "false")
            body_str = json.dumps(request_config["body"], indent=2)  # Pretty print body
            self._create_element(arg_elem, "stringProp", {"name": "Argument.value"}, body_str)
            self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")

        # 2. Add Query parameters (which are now correctly separated)
        if request_config.get("parameters") and isinstance(request_config["parameters"], list):
            for param_obj in request_config["parameters"]:
                if isinstance(param_obj, dict) and "name" in param_obj and "value" in param_obj:
                    arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                                    {"name": param_obj["name"], "elementType": "HTTPArgument"})
                    self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, "true")
                    self._create_element(arg_elem, "stringProp", {"name": "Argument.value"}, str(param_obj["value"]))
                    self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")
                    self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.use_equals"}, "true")
        # --- END: CORRECTED BODY/PARAMETER LOGIC ---

        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.path"}, request_config.get("path", "/"))
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.method"},
                             request_config.get("method", "GET").upper())
        if request_config.get("headers"):
            header_manager = self._create_element(sampler_hash_tree, "HeaderManager",
                                                  {"elementType": "HeaderManager", "guiclass": "HeaderPanel",
                                                   "testclass": "HeaderManager", "testname": "HTTP Header Manager",
                                                   "enabled": "true"})
            collection_prop = self._create_element(header_manager, "collectionProp", {"name": "HeaderManager.headers"})
            for name, value in request_config["headers"].items():
                header_elem = self._create_element(collection_prop, "elementProp",
                                                   {"name": "", "elementType": "Header"})
                self._create_element(header_elem, "stringProp", {"name": "Header.name"}, name)
                self._create_element(header_elem, "stringProp", {"name": "Header.value"}, value)
            self._create_element(sampler_hash_tree, "hashTree")
        if request_config.get("extractions"):
            for extractor in request_config["extractions"]:
                json_extractor = self._create_element(sampler_hash_tree, "JSONPostProcessor",
                                                      {"elementType": "JSONPostProcessor",
                                                       "guiclass": "JSONPostProcessorGui",
                                                       "testclass": "JSONPostProcessor",
                                                       "testname": f"Extract {extractor.get('variable_name')}",
                                                       "enabled": "true"})
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.referenceNames"},
                                     extractor.get("variable_name", ""))
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.jsonPathExprs"},
                                     extractor.get("json_path", ""))
                self._create_element(json_extractor, "stringProp", {"name": "JSONPostProcessor.defaultValues"},
                                     "NOT_FOUND")
                self._create_element(sampler_hash_tree, "hashTree")
        if request_config.get("assertions"):
            for assertion_config in request_config["assertions"]:
                is_json_path_assert = assertion_config.get("json_path_assertion", False)
                if is_json_path_assert:
                    # JSON Path Assertion
                    assertion = self._create_element(sampler_hash_tree, "JSONPathAssertion",
                                                     {"elementType": "JSONPathAssertion",
                                                      "guiclass": "JSONPathAssertionGui",
                                                      "testclass": "JSONPathAssertion",
                                                      "testname": f"Assert JSON Path Exists: {assertion_config.get('pattern')}",
                                                      "enabled": "true"})
                    self._create_element(assertion, "stringProp", {"name": "JSON_PATH"},
                                         str(assertion_config.get("pattern", "")))
                    self._create_element(assertion, "boolProp", {"name": "EXPECT_NULL"}, "false")
                    self._create_element(assertion, "boolProp", {"name": "VALIDATE_JSON"}, "true")
                else:
                    # Response Code/Text Assertion
                    assertion_test_field = "Assertion.response_code" if assertion_config.get(
                        "type") == "response_code" else "Assertion.response_data"
                    assertion_test_type = "16" if assertion_config.get("type") == "response_code" else "2"
                    assertion = self._create_element(sampler_hash_tree, "ResponseAssertion",
                                                     {"elementType": "ResponseAssertion", "guiclass": "AssertionGui",
                                                      "testclass": "ResponseAssertion",
                                                      "testname": f"Assert {assertion_config.get('type')}",
                                                      "enabled": "true"})
                    string_prop_field = self._create_element(assertion, "collectionProp",
                                                             {"name": "Asserion.test_strings"})
                    self._create_element(string_prop_field, "stringProp", {"name": "pattern"},
                                         str(assertion_config.get("pattern", "")))
                    self._create_element(assertion, "stringProp", {"name": "Assertion.test_field"},
                                         assertion_test_field)
                    self._create_element(assertion, "boolProp", {"name": "Assertion.assume_success"}, "false")
                    self._create_element(assertion, "intProp", {"name": "Assertion.test_type"}, assertion_test_type)
                self._create_element(sampler_hash_tree, "hashTree")

    def _add_csv_data_set_config(self, parent_hash_tree: ET.Element, filename: str, variable_names: List[str]):
        csv_data_set = self._create_element(parent_hash_tree, "CSVDataSet",
                                            {"elementType": "CSVDataSet", "guiclass": "TestBeanGUI",
                                             "testclass": "CSVDataSet", "testname": f"CSV Data Set - {filename}",
                                             "enabled": "true"})
        self._create_element(parent_hash_tree, "hashTree")
        self._create_element(csv_data_set, "stringProp", {"name": "filename"}, filename)
        self._create_element(csv_data_set, "stringProp", {"name": "fileEncoding"}, "UTF-8")
        # --- START: CSV HEADER FIX ---
        # Clean up variable names to remove any whitespace or newlines
        cleaned_variable_names = [name.strip() for name in variable_names]
        self._create_element(csv_data_set, "stringProp", {"name": "variableNames"}, ",".join(cleaned_variable_names))
        # --- END: CSV HEADER FIX ---
        self._create_element(csv_data_set, "boolProp", {"name": "ignoreFirstLine"}, "false")
        self._create_element(csv_data_set, "boolProp", {"name": "quotedData"}, "true")
        self._create_element(csv_data_set, "boolProp", {"name": "recycle"}, "true")
        self._create_element(csv_data_set, "boolProp", {"name": "stopThread"}, "false")
        self._create_element(csv_data_set, "stringProp", {"name": "shareMode"}, "shareMode.all")
        self._create_element(csv_data_set, "stringProp", {"name": "delimiter"}, ",")

    def _add_strict_view_results_tree_listener(self, parent_hash_tree: ET.Element):
        listener = self._create_element(parent_hash_tree, "ResultCollector",
                                        {"guiclass": "ViewResultsFullVisualizer", "testclass": "ResultCollector",
                                         "testname": "View Results Tree", "enabled": "true"})
        self._create_element(parent_hash_tree, "hashTree")
        self._create_element(listener, "boolProp", {"name": "ResultCollector.error_logging"}, "false")
        obj_prop = self._create_element(listener, "objProp")
        self._create_element(obj_prop, "name", text="saveConfig")
        value_elem = self._create_element(obj_prop, "value", attrib={"class": "SampleSaveConfiguration"})
        self._create_element(value_elem, "time", text="true");
        self._create_element(value_elem, "latency", text="true");
        self._create_element(value_elem, "timestamp", text="true");
        self._create_element(value_elem, "success", text="true");
        self._create_element(value_elem, "label", text="true");
        self._create_element(value_elem, "code", text="true");
        self._create_element(value_elem, "message", text="true");
        self._create_element(value_elem, "threadName", text="true");
        self._create_element(value_elem, "dataType", text="true");
        self._create_element(value_elem, "encoding", text="false");
        self._create_element(value_elem, "assertions", text="true");
        self._create_element(value_elem, "subresults", text="true");
        self._create_element(value_elem, "responseData", text="false");
        self._create_element(value_elem, "samplerData", text="false");
        self._create_element(value_elem, "xml", text="false");
        self._create_element(value_elem, "fieldNames", text="false");
        self._create_element(value_elem, "responseHeaders", text="false");
        self._create_element(value_elem, "requestHeaders", text="false");
        self._create_element(value_elem, "responseDataOnError", text="false");
        self._create_element(value_elem, "saveAssertionResultsFailureMessage", text="false");
        self._create_element(value_elem, "assertionsResultsToSave", text="0");
        self._create_element(value_elem, "bytes", text="true");
        self._create_element(value_elem, "threadCounts", text="true");
        self._create_element(value_elem, "sentBytes", text="false");
        self._create_element(value_elem, "url", text="false");
        self._create_element(value_elem, "idleTime", text="false");
        self._create_element(value_elem, "connectTime", text="false")
        self._create_element(listener, "stringProp", {"name": "filename"}, "")

    @classmethod
    def generate_jmx_and_artifacts(
            cls,
            designed_test_cases: List[Dict[str, Any]],
            db_sampled_data: Dict[str, pd.DataFrame],
            test_plan_name: str,
            thread_group_name: str,
            num_users: int,
            ramp_up_time: int,
            loop_count: int,
            full_swagger_spec: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str], str, str]:

        table_to_columns = {}

        def scan_for_csv_vars(obj):
            if isinstance(obj, dict):
                for v in obj.values(): scan_for_csv_vars(v)
            elif isinstance(obj, list):
                for item in obj: scan_for_csv_vars(item)
            elif isinstance(obj, str):
                matches = re.findall(r"\${csv_([^_]+)_([^}]+)}", obj)
                for match in matches:
                    table, column = match
                    if table not in table_to_columns: table_to_columns[table] = set()
                    table_to_columns[table].add(column)

        scan_for_csv_vars(designed_test_cases)

        csv_files = {}
        for table, columns in table_to_columns.items():
            if table in db_sampled_data:
                df_table = db_sampled_data[table]
                columns_to_include = [col for col in columns if col in df_table.columns]
                if columns_to_include:
                    csv_filename = f"{table}_data.csv"
                    jmeter_df = pd.DataFrame()
                    for col in columns_to_include:
                        jmeter_variable_name = f"csv_{table}_{col}"
                        jmeter_df[jmeter_variable_name] = df_table[col]
                    csv_files[csv_filename] = jmeter_df.to_csv(index=False)

        protocol, domain, port = "https", "example.com", ""
        if isinstance(full_swagger_spec, dict):
            host = full_swagger_spec.get('host')
            schemes = full_swagger_spec.get('schemes', ['https'])
            protocol = schemes[0] if schemes else 'https'
            if host:
                if '://' not in host:
                    host = f"{protocol}://{host}"
                parsed_url = urlparse(host)
                domain = parsed_url.hostname or domain
                port_val = parsed_url.port
                port = str(port_val) if port_val else ""

        generator = cls(test_plan_name=test_plan_name, thread_group_name=thread_group_name)
        root = ET.Element("jmeterTestPlan", {"version": "1.2", "properties": "5.0", "jmeter": "5.6.2"})
        root_hash_tree = cls._create_element(root, "hashTree")

        test_plan = cls._create_element(root_hash_tree, "TestPlan",
                                        {"elementType": "TestPlan", "guiclass": "TestPlanGui", "testclass": "TestPlan",
                                         "testname": test_plan_name, "enabled": "true"})
        test_plan_hash_tree = cls._create_element(root_hash_tree, "hashTree")

        thread_group = cls._create_element(test_plan_hash_tree, "ThreadGroup",
                                           {"elementType": "ThreadGroup", "guiclass": "ThreadGroupGui",
                                            "testclass": "ThreadGroup", "testname": thread_group_name,
                                            "enabled": "true"})
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}, "continue")
        loop_controller = cls._create_element(thread_group, "elementProp",
                                              {"name": "ThreadGroup.main_controller", "elementType": "LoopController",
                                               "guiclass": "LoopControlPanel", "testclass": "LoopController",
                                               "enabled": "true"})
        cls._create_element(loop_controller, "boolProp", {"name": "LoopController.continue_forever"},
                            "true" if loop_count == -1 else "false")
        cls._create_element(loop_controller, "stringProp", {"name": "LoopController.loops"},
                            str(loop_count) if loop_count != -1 else "")
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}, str(num_users))
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}, str(ramp_up_time))
        cls._create_element(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}, "false")

        thread_group_hash_tree = cls._create_element(test_plan_hash_tree, "hashTree")

        generator._add_http_request_defaults(thread_group_hash_tree, protocol, domain, port)

        for filename, content in csv_files.items():
            # --- START: CSV HEADER FIX ---
            header = content.split('\n', 1)[0].strip()
            variable_names = header.split(',')
            # --- END: CSV HEADER FIX ---
            generator._add_csv_data_set_config(thread_group_hash_tree, filename, variable_names)

        for case in designed_test_cases:
            generator._add_http_request_sampler(thread_group_hash_tree, case)

        generator._add_strict_view_results_tree_listener(test_plan_hash_tree)

        jmx_content = cls._prettify_xml(root)

        json_content = json.dumps({"designed_test_cases": designed_test_cases}, indent=2)
        yaml_content = yaml.dump({"designed_test_cases": designed_test_cases}, allow_unicode=True, sort_keys=False)

        return jmx_content, csv_files, json_content, yaml_content