import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Any, Tuple
import re
import json
import logging
import pandas as pd
import yaml
from urllib.parse import urlparse, unquote

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JMeterScriptGenerator:
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

    def _add_test_plan_udv(self, test_plan_element: ET.Element):
        udv_prop = self._create_element(test_plan_element, "elementProp",
                                        {"name": "TestPlan.user_defined_variables", "elementType": "Arguments"})
        udv_prop.set("guiclass", "ArgumentsPanel")
        udv_prop.set("testclass", "Arguments")
        udv_prop.set("testname", "User Defined Variables")
        udv_prop.set("enabled", "true")
        self._create_element(udv_prop, "collectionProp", {"name": "Arguments.arguments"})

    def _add_http_request_defaults(self, parent_hash_tree: ET.Element):
        config_defaults_attrib = {"elementType": "HttpDefaults", "guiclass": "HttpDefaultsGui",
                                  "testclass": "HttpDefaults", "testname": "HTTP Request Defaults", "enabled": "true"}
        config_defaults = self._create_element(parent_hash_tree, "ConfigTestElement", config_defaults_attrib)
        arguments_element = self._create_element(config_defaults, "elementProp",
                                                 {"name": "HTTPsampler.Arguments", "elementType": "Arguments",
                                                  "guiclass": "HTTPArgumentsPanel", "testclass": "Arguments",
                                                  "testname": "User Defined Variables", "enabled": "true"})
        self._create_element(arguments_element, "collectionProp", {"name": "Arguments.arguments"})
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.protocol"}, "https")
        self._create_element(config_defaults, "stringProp", {"name": "HTTPSampler.domain"}, "${HOST}")
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_user_defined_variables(self, parent_hash_tree: ET.Element, target_host: str):
        udv_attrib = {"elementType": "Arguments", "guiclass": "ArgumentsPanel", "testclass": "Arguments",
                      "testname": "User Defined Variables", "enabled": "true"}
        udv = self._create_element(parent_hash_tree, "Arguments", udv_attrib)
        collection_prop = self._create_element(udv, "collectionProp", {"name": "Arguments.arguments"})
        host_prop = self._create_element(collection_prop, "elementProp", {"name": "HOST", "elementType": "Argument"})
        self._create_element(host_prop, "stringProp", {"name": "Argument.name"}, "HOST")
        self._create_element(host_prop, "stringProp", {"name": "Argument.value"}, target_host)
        self._create_element(host_prop, "stringProp", {"name": "Argument.metadata"}, "=")
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_timer(self, parent_hash_tree: ET.Element, delay: int):
        timer_attrib = {"elementType": "ConstantTimer", "guiclass": "ConstantTimerGui", "testclass": "ConstantTimer",
                        "testname": "Think Time", "enabled": "true"}
        timer = self._create_element(parent_hash_tree, "ConstantTimer", timer_attrib)
        self._create_element(timer, "stringProp", {"name": "ConstantTimer.delay"}, str(delay))
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_transaction_controller(self, parent_hash_tree: ET.Element, name: str) -> ET.Element:
        tc_attrib = {"elementType": "TransactionController", "guiclass": "TransactionControllerGui",
                     "testclass": "TransactionController", "testname": name, "enabled": "true"}
        tc = self._create_element(parent_hash_tree, "TransactionController", tc_attrib)
        self._create_element(tc, "boolProp", {"name": "TransactionController.parent"}, "true")
        tc_hash_tree = self._create_element(parent_hash_tree, "hashTree")
        return tc_hash_tree

    def _add_header_manager(self, parent_hash_tree: ET.Element, headers: Dict[str, str]):
        header_manager_attrib = {"elementType": "HeaderManager", "guiclass": "HeaderPanel",
                                 "testclass": "HeaderManager", "testname": "HTTP Header Manager", "enabled": "true"}
        header_manager = self._create_element(parent_hash_tree, "HeaderManager", header_manager_attrib)
        collection_prop = self._create_element(header_manager, "collectionProp", {"name": "HeaderManager.headers"})
        for name, value in headers.items():
            header_elem = self._create_element(collection_prop, "elementProp", {"name": "", "elementType": "Header"})
            self._create_element(header_elem, "stringProp", {"name": "Header.name"}, name)
            self._create_element(header_elem, "stringProp", {"name": "Header.value"}, value)
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_json_extractor(self, parent_hash_tree: ET.Element, ref_name: str, json_path: str):
        extractor_attrib = {"elementType": "JSONPostProcessor", "guiclass": "JSONPostProcessorGui",
                            "testclass": "JSONPostProcessor", "testname": f"Extract {ref_name}", "enabled": "true"}
        extractor = self._create_element(parent_hash_tree, "JSONPostProcessor", extractor_attrib)
        self._create_element(extractor, "stringProp", {"name": "JSONPostProcessor.referenceNames"}, ref_name)
        self._create_element(extractor, "stringProp", {"name": "JSONPostProcessor.jsonPathExprs"}, json_path)
        self._create_element(extractor, "stringProp", {"name": "JSONPostProcessor.match_numbers"}, "1")
        self._create_element(extractor, "stringProp", {"name": "JSONPostProcessor.defaultValues"},
                             f"{ref_name}_NOT_FOUND")
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_response_assertion(self, parent_hash_tree: ET.Element, codes: List[str]):
        assertion_attrib = {"elementType": "ResponseAssertion", "guiclass": "AssertionGui",
                            "testclass": "ResponseAssertion", "testname": f"Assert Success Status - {', '.join(codes)}",
                            "enabled": "true"}
        assertion = self._create_element(parent_hash_tree, "ResponseAssertion", assertion_attrib)
        collection_prop = self._create_element(assertion, "collectionProp", {"name": "Asserion.test_strings"})
        for code in codes:
            self._create_element(collection_prop, "stringProp", {"name": str(hash(code))}, code)
        self._create_element(assertion, "stringProp", {"name": "Assertion.test_field"}, "Assertion.response_code")
        self._create_element(assertion, "boolProp", {"name": "Assertion.assume_success"}, "false")
        self._create_element(assertion, "intProp", {"name": "Assertion.test_type"}, "16")  # 16 = "Matches" operator
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_http_request_sampler(self, parent_hash_tree: ET.Element, request_config: Dict[str, Any]):
        sampler_attrib = {"elementType": "HTTPSamplerProxy", "guiclass": "HttpTestSampleGui",
                          "testclass": "HTTPSamplerProxy", "testname": request_config.get("name"), "enabled": "true"}
        sampler = self._create_element(parent_hash_tree, "HTTPSamplerProxy", sampler_attrib)
        arguments_element = self._create_element(sampler, "elementProp",
                                                 {"name": "HTTPsampler.Arguments", "elementType": "Arguments",
                                                  "guiclass": "HTTPArgumentsPanel", "testclass": "Arguments",
                                                  "enabled": "true"})
        arguments_collection_prop = self._create_element(arguments_element, "collectionProp",
                                                         {"name": "Arguments.arguments"})

        has_body = request_config.get("body") is not None
        self._create_element(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"}, "true" if has_body else "false")

        if has_body:
            arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                            {"name": "", "elementType": "HTTPArgument"})
            self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, "false")
            body_str = json.dumps(request_config["body"], indent=2) if isinstance(request_config["body"],
                                                                                  (dict, list)) else str(
                request_config["body"])
            self._create_element(arg_elem, "stringProp", {"name": "Argument.value"}, body_str)
            self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")
        else:
            parsed_url = urlparse(request_config.get("path", "/"))
            query_params = parsed_url.query.split('&') if parsed_url.query else []
            for param in query_params:
                if '=' in param:
                    name, value = param.split('=', 1)
                    arg_elem = self._create_element(arguments_collection_prop, "elementProp",
                                                    {"name": unquote(name), "elementType": "HTTPArgument"})
                    self._create_element(arg_elem, "boolProp", {"name": "HTTPArgument.always_encode"}, False)
                    self._create_element(arg_elem, "stringProp", {"name": "Argument.value"}, unquote(value))
                    self._create_element(arg_elem, "stringProp", {"name": "Argument.metadata"}, "=")
                    self._create_element(arg_elem, "stringProp", {"name": "Argument.name"}, unquote(name))

        path_only = request_config.get("path", "/").split('?')[0]
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.path"}, path_only)
        self._create_element(sampler, "stringProp", {"name": "HTTPSampler.method"},
                             request_config.get("method", "GET").upper())
        sampler_hash_tree = self._create_element(parent_hash_tree, "hashTree")

        if request_config.get("headers"): self._add_header_manager(sampler_hash_tree, request_config["headers"])
        if request_config.get("extractions"):
            for extraction in request_config["extractions"]:
                if extraction.get("type") == "JSONExtractor": self._add_json_extractor(sampler_hash_tree,
                                                                                       extraction.get("refname"),
                                                                                       extraction.get("jsonpath"))
        if request_config.get("assertions"):
            for assertion in request_config["assertions"]:
                if assertion.get("type") == "ResponseAssertion" and assertion.get("codes"):
                    self._add_response_assertion(sampler_hash_tree, assertion["codes"])

    def _add_csv_data_set_config(self, parent_hash_tree: ET.Element, filename: str, variable_names: List[str]):
        csv_attrib = {"elementType": "CSVDataSet", "guiclass": "TestBeanGUI", "testclass": "CSVDataSet",
                      "testname": f"CSV Data Set Config - {filename}", "enabled": "true"}
        csv_config = self._create_element(parent_hash_tree, "CSVDataSet", csv_attrib)
        self._create_element(csv_config, "stringProp", {"name": "filename"}, f"data/{filename}")
        self._create_element(csv_config, "stringProp", {"name": "variableNames"}, ",".join(variable_names))
        self._create_element(csv_config, "boolProp", {"name": "ignoreFirstLine"}, "true")
        self._create_element(csv_config, "stringProp", {"name": "delimiter"}, ",")
        self._create_element(csv_config, "boolProp", {"name": "quotedData"}, "true")
        self._create_element(csv_config, "boolProp", {"name": "recycle"}, "true")
        self._create_element(csv_config, "boolProp", {"name": "stopThread"}, "false")
        self._create_element(csv_config, "stringProp", {"name": "shareMode"}, "shareMode.all")
        parent_hash_tree.append(ET.Element("hashTree"))

    def _add_strict_view_results_tree_listener(self, parent_hash_tree: ET.Element):
        listener_attrib = {"guiclass": "ViewResultsFullVisualizer", "testclass": "ResultCollector",
                           "testname": "View Results Tree", "enabled": "true"}
        listener = self._create_element(parent_hash_tree, "ResultCollector", listener_attrib)
        self._create_element(listener, "boolProp", {"name": "ResultCollector.error_logging"}, "false")
        obj_prop = self._create_element(listener, "objProp")
        self._create_element(obj_prop, "name", text="saveConfig")
        value = self._create_element(obj_prop, "value", {"class": "SampleSaveConfiguration"})
        self._create_element(value, "time", text="true");
        self._create_element(value, "latency", text="true")
        self._create_element(value, "timestamp", text="true");
        self._create_element(value, "success", text="true")
        self._create_element(value, "label", text="true");
        self._create_element(value, "code", text="true")
        self._create_element(value, "message", text="true");
        self._create_element(value, "threadName", text="true")
        self._create_element(value, "dataType", text="true");
        self._create_element(value, "encoding", text="false")
        self._create_element(value, "assertions", text="true");
        self._create_element(value, "subresults", text="true")
        self._create_element(value, "responseData", text="false");
        self._create_element(value, "samplerData", text="false")
        self._create_element(value, "xml", text="false");
        self._create_element(value, "fieldNames", text="true")
        self._create_element(value, "responseHeaders", text="false");
        self._create_element(value, "requestHeaders", text="false")
        self._create_element(value, "responseDataOnError", text="false");
        self._create_element(value, "saveAssertionResultsFailureMessage", text="true")
        self._create_element(value, "assertionsResultsToSave", text="0");
        self._create_element(value, "bytes", text="true")
        self._create_element(value, "sentBytes", text="true");
        self._create_element(value, "url", text="true")
        self._create_element(value, "threadCounts", text="true");
        self._create_element(value, "idleTime", text="true")
        self._create_element(value, "connectTime", text="true")
        parent_hash_tree.append(ET.Element("hashTree"))

    @classmethod
    def generate_jmx_and_artifacts(cls, designed_scenarios: Dict[str, List[Dict]],
                                   db_sampled_data: Dict[str, pd.DataFrame], test_plan_name: str,
                                   thread_group_name: str, target_host: str, num_users: int, ramp_up_time: int,
                                   loop_count: int, think_time: int) -> Tuple[str, Dict[str, str], str, str]:
        table_to_columns = {}

        def scan_for_csv_vars(obj):
            if isinstance(obj, dict):
                for v in obj.values(): scan_for_csv_vars(v)
            elif isinstance(obj, list):
                for item in obj: scan_for_csv_vars(item)
            elif isinstance(obj, str):
                matches = re.findall(r"\${(csv_[^_]+_[^}]+)}", obj)
                for jmeter_var in matches:
                    parts = jmeter_var.split('_');
                    if len(parts) >= 3: table, column = parts[1], '_'.join(parts[2:])
                    if table not in table_to_columns: table_to_columns[table] = set()
                    table_to_columns[table].add(column)

        scan_for_csv_vars(designed_scenarios)
        csv_files = {}
        if db_sampled_data:
            for table, columns in table_to_columns.items():
                if table in db_sampled_data:
                    columns_to_include = sorted(list(columns))
                    if columns_to_include:
                        csv_filename = f"{table}_data.csv";
                        jmeter_df = pd.DataFrame()
                        for col in columns_to_include:
                            jmeter_variable_name = f"csv_{table}_{col}"
                            jmeter_df[jmeter_variable_name] = db_sampled_data[table][col]
                        csv_files[csv_filename] = jmeter_df.to_csv(index=False)
        generator = cls(test_plan_name=test_plan_name, thread_group_name=thread_group_name)
        root = ET.Element("jmeterTestPlan", {"version": "1.2", "properties": "5.0", "jmeter": "5.6.3"})
        root_hash_tree = cls._create_element(root, "hashTree")
        test_plan = cls._create_element(root_hash_tree, "TestPlan",
                                        {"elementType": "TestPlan", "guiclass": "TestPlanGui", "testclass": "TestPlan",
                                         "testname": test_plan_name, "enabled": "true"})
        generator._add_test_plan_udv(test_plan)
        test_plan_hash_tree = cls._create_element(root_hash_tree, "hashTree")
        generator._add_user_defined_variables(test_plan_hash_tree, target_host)
        thread_group = cls._create_element(test_plan_hash_tree, "ThreadGroup",
                                           {"elementType": "ThreadGroup", "guiclass": "ThreadGroupGui",
                                            "testclass": "ThreadGroup", "testname": thread_group_name,
                                            "enabled": "true"})
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}, "continue")
        loop_controller = cls._create_element(thread_group, "elementProp",
                                              {"name": "ThreadGroup.main_controller", "elementType": "LoopController",
                                               "guiclass": "LoopControlPanel", "testclass": "LoopController",
                                               "enabled": "true"})
        cls._create_element(loop_controller, "stringProp", {"name": "LoopController.loops"}, str(loop_count))
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}, str(num_users))
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}, str(ramp_up_time))
        cls._create_element(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}, "false")
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.duration"}, "")
        cls._create_element(thread_group, "stringProp", {"name": "ThreadGroup.delay"}, "")
        cls._create_element(thread_group, "boolProp", {"name": "ThreadGroup.same_user_on_next_iteration"}, "true")
        thread_group_hash_tree = cls._create_element(test_plan_hash_tree, "hashTree")
        generator._add_http_request_defaults(thread_group_hash_tree)
        for filename, content in csv_files.items():
            variable_names = content.splitlines()[0].split(',')
            generator._add_csv_data_set_config(thread_group_hash_tree, filename, variable_names)
        for scenario_name, steps in designed_scenarios.items():
            if not steps: continue
            tc_hash_tree = generator._add_transaction_controller(thread_group_hash_tree, scenario_name)
            for i, step in enumerate(steps):
                generator._add_http_request_sampler(tc_hash_tree, step)
                if think_time > 0 and i < len(steps) - 1:
                    generator._add_timer(tc_hash_tree, think_time)
        generator._add_strict_view_results_tree_listener(test_plan_hash_tree)
        jmx_content = cls._prettify_xml(root)
        json_content = json.dumps({"designed_scenarios": designed_scenarios}, indent=2)
        yaml_content = yaml.dump({"designed_scenarios": designed_scenarios}, allow_unicode=True, sort_keys=False)
        return jmx_content, csv_files, json_content, yaml_content