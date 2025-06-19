import requests
import json
from urllib.parse import urlparse
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SwaggerEndpoint:
    method: str
    path: str
    operation_id: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None  # Added description field
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    body_schema: Optional[Dict[str, Any]] = None
    request_examples: List[Dict[str, Any]] = field(default_factory=list)  # Added for request examples
    responses: Dict[str, Any] = field(default_factory=dict)
    security_schemes: List[Dict[str, Any]] = field(default_factory=list)  # Added for security requirements
    error_handling: Dict[str, Any] = field(default_factory=dict)  # Added for error handling details

    # Add any other relevant metadata as needed

    def to_dict(self):
        """Converts the SwaggerEndpoint object to a dictionary, suitable for JSON serialization."""
        return {
            "method": self.method,
            "path": self.path,
            "operation_id": self.operation_id,
            "summary": self.summary,
            "description": self.description,  # Include description in dict
            "parameters": self.parameters,
            "body_schema": self.body_schema,
            "request_examples": self.request_examples,  # Include request examples
            "responses": self.responses,
            "security_schemes": self.security_schemes,  # Include security schemes
            "error_handling": self.error_handling  # Include error handling
        }


class SwaggerParser:
    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_data: Dict[str, Any] = {}
        self.definitions: Dict[str, Any] = {}  # For Swagger 2.0
        self.components_schemas: Dict[str, Any] = {}  # For OpenAPI 3.x

    def load_swagger_spec(self) -> bool:
        """Loads the Swagger/OpenAPI specification from the provided URL."""
        try:
            response = requests.get(self.swagger_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            self.swagger_data = response.json()

            # Determine OpenAPI version and set schema references accordingly
            if 'swagger' in self.swagger_data and self.swagger_data['swagger'].startswith('2.0'):
                logger.info("Swagger 2.0 specification detected.")
                self.definitions = self.swagger_data.get('definitions', {})
            elif 'openapi' in self.swagger_data and self.swagger_data['openapi'].startswith(('3.0', '3.1')):
                logger.info("OpenAPI 3.x specification detected.")
                self.components_schemas = self.swagger_data.get('components', {}).get('schemas', {})
            else:
                logger.warning("Unsupported Swagger/OpenAPI version or malformed spec.")
                return False

            logger.info("Swagger spec loaded successfully.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading Swagger spec from {self.swagger_url}: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Swagger spec: {e}")
            return False

    def _resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        """Resolves a $ref path to the actual schema definition."""
        # Clean up ref_path for dictionary lookup
        parts = ref_path.replace('#/', '').split('/')

        current_def = {}
        if ref_path.startswith('#/definitions/') and self.definitions:  # Swagger 2.0
            current_def = self.definitions
            for part in parts[1:]:  # Skip 'definitions'
                current_def = current_def.get(part, {})
        elif ref_path.startswith('#/components/schemas/') and self.components_schemas:  # OpenAPI 3.x
            current_def = self.components_schemas
            for part in parts[2:]:  # Skip 'components', 'schemas'
                current_def = current_def.get(part, {})

        if not current_def:
            logger.warning(f"Could not resolve reference: {ref_path}")
        return current_def

    def parse(self) -> List[SwaggerEndpoint]:
        """Parses the loaded Swagger/OpenAPI specification to extract API endpoint details."""
        if not self.swagger_data:
            logger.error("Swagger data not loaded. Call load_swagger_spec() first.")
            return []

        endpoints: List[SwaggerEndpoint] = []
        paths = self.swagger_data.get('paths', {})

        for path, path_details in paths.items():
            for method, details in path_details.items():
                if method.lower() not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                    continue  # Skip non-HTTP method elements like 'parameters' or 'summary' at path level

                operation_id = details.get('operationId')
                summary = details.get('summary')
                description = details.get('description')  # Extract description

                parameters_processed: List[Dict[str, Any]] = []
                # Handle parameters defined at the path level and operation level
                combined_parameters = path_details.get('parameters', []) + details.get('parameters', [])

                for param in combined_parameters:
                    # Resolve $ref for parameter schemas if present
                    if '$ref' in param:
                        resolved_param = self._resolve_ref(param['$ref'])
                        # Merge resolved properties into the parameter if necessary
                        param = {**param, **resolved_param}

                    param_name = param.get('name')
                    param_in = param.get('in')  # query, header, path, body, formData
                    param_type = param.get('type') or param.get('schema', {}).get(
                        'type')  # type for Swagger 2.0, schema.type for OpenAPI 3.x
                    param_required = param.get('required', False)
                    param_description = param.get('description')

                    # For OpenAPI 3.x, schema details are under 'schema' key
                    param_schema = param.get('schema', {})
                    if '$ref' in param_schema:
                        param_schema = self._resolve_ref(param_schema['$ref'])
                        param_type = param_schema.get('type')  # Update type from resolved schema

                    parameters_processed.append({
                        'name': param_name,
                        'in': param_in,
                        'type': param_type,
                        'required': param_required,
                        'description': param_description,
                        'schema': param_schema  # Include full schema for detailed understanding
                    })

                body_schema_resolved: Optional[Dict[str, Any]] = None
                request_examples: List[Dict[str, Any]] = []

                # Handle requestBody for OpenAPI 3.x
                request_body = details.get('requestBody')
                if request_body:
                    content = request_body.get('content', {})
                    for media_type, media_type_details in content.items():
                        schema_raw = media_type_details.get('schema', {})
                        if '$ref' in schema_raw:
                            body_schema_resolved = self._resolve_ref(schema_raw['$ref'])
                        else:
                            body_schema_resolved = schema_raw

                        # Extract examples from requestBody.content
                        if 'examples' in media_type_details:
                            for example_name, example_obj in media_type_details['examples'].items():
                                request_examples.append({
                                    "name": example_name,
                                    "summary": example_obj.get("summary"),
                                    "description": example_obj.get("description"),
                                    "value": example_obj.get("value")  # The actual example payload
                                })
                        elif 'example' in media_type_details:  # Single example
                            request_examples.append({
                                "name": "default",
                                "value": media_type_details.get("example")
                            })

                # Handle consumes/body for Swagger 2.0
                elif 'consumes' in details:
                    for content_type in details['consumes']:
                        if content_type == 'application/json':
                            # Assuming body parameter is defined with 'in: body'
                            for param in details.get('parameters', []):
                                if param.get('in') == 'body' and 'schema' in param:
                                    schema_raw = param['schema']
                                    if '$ref' in schema_raw:
                                        body_schema_resolved = self._resolve_ref(schema_raw['$ref'])
                                    else:
                                        body_schema_resolved = schema_raw
                                    break  # Found the body schema, exit loop

                responses_data = {}
                for status_code, response_obj in details.get('responses', {}).items():
                    # Resolve $ref for response schemas if present
                    if 'schema' in response_obj and '$ref' in response_obj['schema']:  # Swagger 2.0
                        response_obj['schema'] = self._resolve_ref(response_obj['schema']['$ref'])
                    elif 'content' in response_obj:  # OpenAPI 3.x
                        for media_type, media_type_details in response_obj['content'].items():
                            if 'schema' in media_type_details:
                                if '$ref' in media_type_details['schema']:
                                    media_type_details['schema'] = self._resolve_ref(
                                        media_type_details['schema']['$ref'])
                            # Extract response examples if available
                            if 'examples' in media_type_details:
                                response_obj['examples'] = []
                                for example_name, example_obj in media_type_details['examples'].items():
                                    response_obj['examples'].append({
                                        "name": example_name,
                                        "summary": example_obj.get("summary"),
                                        "description": example_obj.get("description"),
                                        "value": example_obj.get("value")
                                    })
                            elif 'example' in media_type_details:
                                response_obj['examples'] = [{
                                    "name": "default",
                                    "value": media_type_details.get("example")
                                }]

                    responses_data[status_code] = {
                        "description": response_obj.get("description"),
                        "schema": response_obj.get("schema") or response_obj.get("content", {}).get("application/json",
                                                                                                    {}).get("schema"),
                        # Try to get schema from content for OpenAPI 3.x
                        "headers": response_obj.get("headers"),  # Include response headers
                        "examples": response_obj.get("examples")  # Include response examples
                    }

                # Extract Security Schemes
                security_schemes_list = []
                security_requirements = details.get('security', [])
                for req_block in security_requirements:
                    for scheme_name, scopes in req_block.items():
                        if self.swagger_data.get('securityDefinitions') and scheme_name in self.swagger_data[
                            'securityDefinitions']:  # Swagger 2.0
                            scheme_def = self.swagger_data['securityDefinitions'][scheme_name]
                            security_schemes_list.append({
                                "name": scheme_name,
                                "type": scheme_def.get("type"),
                                "in": scheme_def.get("in"),
                                "header_name": scheme_def.get("name"),
                                "description": scheme_def.get("description"),
                                "scopes": scopes
                            })
                        elif self.swagger_data.get('components', {}).get('securitySchemes') and scheme_name in \
                                self.swagger_data['components']['securitySchemes']:  # OpenAPI 3.x
                            scheme_def = self.swagger_data['components']['securitySchemes'][scheme_name]
                            security_schemes_list.append({
                                "name": scheme_name,
                                "type": scheme_def.get("type"),
                                "scheme": scheme_def.get("scheme"),  # e.g., 'bearer'
                                "bearerFormat": scheme_def.get("bearerFormat"),
                                "in": scheme_def.get("in"),  # For API Key
                                "header_name": scheme_def.get("name"),  # For API Key
                                "description": scheme_def.get("description"),
                                "scopes": scopes
                            })

                # Extract Error Handling details (simple extraction, can be expanded)
                error_handling_details = {}
                for status_code, response_detail in responses_data.items():
                    if status_code.startswith(('4', '5')):  # Check for 4xx or 5xx status codes
                        error_handling_details[status_code] = response_detail

                endpoints.append(
                    SwaggerEndpoint(
                        method=method.upper(),
                        path=path,
                        operation_id=operation_id,
                        summary=summary,
                        description=description,  # Assign description
                        parameters=parameters_processed,
                        body_schema=body_schema_resolved,
                        request_examples=request_examples,  # Assign request examples
                        responses=responses_data,
                        security_schemes=security_schemes_list,  # Assign security schemes
                        error_handling=error_handling_details  # Assign error handling
                    )
                )
        logger.info(f"Extracted {len(endpoints)} API endpoints with detailed information.")
        return endpoints

