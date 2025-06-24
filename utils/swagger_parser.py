import requests
import json
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass, field  # <--- THIS IS THE MISSING LINE

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
    security_schemes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "method": self.method, "path": self.path, "operation_id": self.operation_id,
            "summary": self.summary, "description": self.description, "parameters": self.parameters,
            "body_schema": self.body_schema, "responses": self.responses, "security_schemes": self.security_schemes,
        }


class SwaggerParser:
    def __init__(self, spec_input: Union[str, Dict]):
        self.spec_input = spec_input
        self.swagger_data: Dict[str, Any] = {}
        self.definitions: Dict[str, Any] = {}
        self.components_schemas: Dict[str, Any] = {}

    def load_swagger_spec(self) -> bool:
        try:
            if isinstance(self.spec_input, str):  # Handle URL
                response = requests.get(self.spec_input)
                response.raise_for_status()
                self.swagger_data = response.json()
            elif isinstance(self.spec_input, dict):  # Handle pre-loaded dictionary
                self.swagger_data = self.spec_input
            else:
                logger.error("Invalid input for SwaggerParser. Must be a URL string or a dictionary.")
                return False

            if 'swagger' in self.swagger_data and self.swagger_data['swagger'].startswith('2.0'):
                self.definitions = self.swagger_data.get('definitions', {})
            elif 'openapi' in self.swagger_data and self.swagger_data['openapi'].startswith('3.'):
                self.components_schemas = self.swagger_data.get('components', {}).get('schemas', {})
            else:
                logger.warning("Unsupported Swagger/OpenAPI version or malformed spec.")
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading or parsing Swagger spec: {e}")
            return False

    def _resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        parts = ref_path.replace('#/', '').split('/')
        if ref_path.startswith('#/definitions/'):
            current_def = self.definitions
            for part in parts[1:]: current_def = current_def.get(part, {})
            return current_def
        elif ref_path.startswith('#/components/schemas/'):
            current_def = self.components_schemas
            for part in parts[2:]: current_def = current_def.get(part, {})
            return current_def
        return {}

    def parse(self) -> List[SwaggerEndpoint]:
        if not self.swagger_data and not self.load_swagger_spec():
            return []

        endpoints: List[SwaggerEndpoint] = []
        paths = self.swagger_data.get('paths', {})

        for path, path_details in paths.items():
            for method, details in path_details.items():
                if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']: continue

                body_schema_resolved = None
                if 'requestBody' in details and 'content' in details.get('requestBody', {}):
                    content = details['requestBody']['content']
                    if 'application/json' in content and 'schema' in content['application/json']:
                        schema_raw = content['application/json']['schema']
                        body_schema_resolved = self._resolve_ref(
                            schema_raw.get('$ref', '')) if '$ref' in schema_raw else schema_raw

                endpoints.append(
                    SwaggerEndpoint(
                        method=method.upper(), path=path, operation_id=details.get('operationId'),
                        summary=details.get('summary'), parameters=details.get('parameters', []),
                        responses=details.get('responses', {}), security_schemes=details.get('security', []),
                        body_schema=body_schema_resolved
                    )
                )
        return endpoints
