from typing import Dict, List, Any
from utils.swagger_parser import SwaggerEndpoint


class DataMapper:
    @staticmethod
    def suggest_mappings(endpoints: List[SwaggerEndpoint], tables_schema: Dict[str, List[Dict[str, str]]]) -> Dict[
        str, Dict[str, str]]:
        mappings = {}

        for endpoint in endpoints:
            endpoint_key = f"{endpoint.method} {endpoint.path}"
            mappings[endpoint_key] = {}

            for param in endpoint.parameters:
                param_name = param.get('name', '')
                param_type = param.get('type', param.get('schema', {}).get('type', ''))

                best_match = DataMapper._find_best_match(param_name, param_type, tables_schema)
                if best_match:
                    mappings[endpoint_key][param_name] = best_match

        return mappings

    @staticmethod
    def _find_best_match(param_name: str, param_type: str, tables_schema: Dict[str, List[Dict[str, str]]]) -> str:
        param_name_lower = param_name.lower()

        for table_name, columns in tables_schema.items():
            for column in columns:
                column_name = column['name'].lower()

                if param_name_lower == column_name:
                    return f"{table_name}.{column['name']}"

                if param_name_lower in column_name or column_name in param_name_lower:
                    return f"{table_name}.{column['name']}"

        return ""