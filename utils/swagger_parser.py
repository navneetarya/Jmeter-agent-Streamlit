import requests
import json
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SwaggerEndpoint:
    method: str
    path: str
    operation_id: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    body_schema: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)
    # *** NEW: Field to store success codes ***
    success_codes: List[str] = field(default_factory=list)
    security_schemes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "method": self.method,
            "path": self.path,
            "operation_id": self.operation_id,
            "summary": self.summary,
            "description": self.description,
            "parameters": self.parameters,
            "body_schema": self.body_schema,
            "responses": self.responses,
            "success_codes": self.success_codes,  # Include in the dictionary output
            "security_schemes": self.security_schemes,
        }


class SwaggerParser:
    def __init__(self, spec_input: Union[str, Dict]):
        self.spec_input = spec_input
        self.swagger_data: Dict[str, Any] = {}
        self.definitions: Dict[str, Any] = {}
        self.components: Dict[str, Any] = {}

    def load_swagger_spec(self) -> bool:
        # This method is correct and unchanged
        try:
            if isinstance(self.spec_input, str):
                response = requests.get(self.spec_input)
                response.raise_for_status()
                self.swagger_data = response.json()
            elif isinstance(self.spec_input, dict):
                self.swagger_data = self.spec_input
            else:
                return False
            if 'swagger' in self.swagger_data and self.swagger_data['swagger'].startswith('2.0'):
                self.definitions = self.swagger_data.get('definitions', {})
            elif 'openapi' in self.swagger_data and self.swagger_data['openapi'].startswith('3.'):
                self.components = self.swagger_data.get('components', {})
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading or parsing Swagger spec: {e}")
            return False

    def _resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        # This method is correct and unchanged
        if not ref_path.startswith('#/'): return {}
        parts = ref_path[2:].split('/')
        current_node = self.swagger_data
        try:
            for part in parts: current_node = current_node[part]
            return current_node
        except (KeyError, TypeError):
            logger.warning(f"Could not resolve $ref: {ref_path}")
            return {}

    def parse(self) -> List[SwaggerEndpoint]:
        if not self.swagger_data and not self.load_swagger_spec():
            return []

        endpoints: List[SwaggerEndpoint] = []
        paths = self.swagger_data.get('paths', {})

        for path, path_item in paths.items():
            common_parameters = [self._resolve_ref(p['$ref']) if '$ref' in p else p for p in
                                 path_item.get('parameters', [])]
            for method, operation in path_item.items():
                if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']: continue

                # *** NEW: Logic to extract success codes ***
                success_codes = []
                operation_responses = operation.get('responses', {})
                for code, resp_details in operation_responses.items():
                    # Success codes are typically in the 2xx range
                    if code.startswith('2'):
                        success_codes.append(code)
                # If no 2xx codes, default to 200 as a fallback
                if not success_codes:
                    success_codes.append("200")

                body_schema = None
                if 'requestBody' in operation:
                    request_body = self._resolve_ref(operation['requestBody']['$ref']) if '$ref' in operation[
                        'requestBody'] else operation['requestBody']
                    content = request_body.get('content', {})
                    media_type = content.get('application/json') or content.get('application/json-patch+json') or next(
                        iter(content.values()), None)
                    if media_type and 'schema' in media_type:
                        schema_node = media_type['schema']
                        body_schema = self._resolve_ref(schema_node['$ref']) if '$ref' in schema_node else schema_node

                operation_parameters = [self._resolve_ref(p['$ref']) if '$ref' in p else p for p in
                                        operation.get('parameters', [])]
                all_params = {p['name']: p for p in common_parameters}
                for p in operation_parameters: all_params[p['name']] = p
                for p in all_params.values():
                    if 'schema' in p and 'type' in p['schema']: p['type'] = p['schema']['type']

                endpoints.append(
                    SwaggerEndpoint(
                        method=method.upper(),
                        path=path,
                        operation_id=operation.get('operationId'),
                        summary=operation.get('summary'),
                        parameters=list(all_params.values()),
                        body_schema=body_schema,
                        responses=operation.get('responses', {}),
                        success_codes=sorted(success_codes),  # Store the extracted codes
                        security_schemes=operation.get('security', [])
                    )
                )
        return endpoints