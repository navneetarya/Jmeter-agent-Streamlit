import json
import dataclasses
from typing import List, Dict, Any, Optional
import requests  # Import requests for fetching URLs
import logging  # Import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SwaggerEndpoint:
    """
    Represents a single API endpoint from a Swagger/OpenAPI definition.
    """
    path: str
    method: str
    summary: Optional[str] = None
    operation_id: Optional[str] = None
    parameters: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    responses: Dict[str, Any] = dataclasses.field(default_factory=dict)
    produces: List[str] = dataclasses.field(default_factory=list)
    consumes: List[str] = dataclasses.field(default_factory=list)  # Added consumes
    request_body: Optional[Dict[str, Any]] = None
    parsed_response_schemas: Dict[str, Any] = dataclasses.field(default_factory=dict)  # Added parsed_response_schemas


class SwaggerParser:
    """
    Parses a Swagger/OpenAPI JSON specification from a URL.
    """

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_spec = None  # Original fetched spec
        self.swagger_data = {}  # Deep-resolved spec
        # Automatically fetch and resolve the spec when the parser is instantiated
        self.fetch_swagger_spec()

    def fetch_swagger_spec(self) -> Dict[str, Any]:
        """
        Fetches the Swagger specification from the URL and performs deep-resolution
        of all references within the spec.
        """
        try:
            logger.info(f"Attempting to fetch Swagger spec from: {self.swagger_url}")
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            self.swagger_spec = response.json()
            # Create a deep copy of the fetched data to work with for resolution
            self.swagger_data = json.loads(json.dumps(self.swagger_spec))

            self._deep_resolve_all_refs_in_spec()  # Call the new deep resolution method

            logger.info("Swagger specification fetched and resolved successfully.")
            return self.swagger_spec  # Return the original structure for consistency, self.swagger_data holds the resolved version
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Swagger spec from {self.swagger_url}: {e}")
            raise Exception(f"Failed to fetch Swagger spec: {e}")  # Re-raise as generic Exception for Streamlit
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Swagger JSON from {self.swagger_url}: {e}")
            raise Exception(f"Failed to parse Swagger JSON: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Swagger fetch/parse: {e}")
            raise Exception(f"An unexpected error occurred: {e}")

    def resolve_swagger_ref(self, schema_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolves a $ref in Swagger/OpenAPI schema.
        Handles nested references within properties and array items.
        """
        if not isinstance(schema_obj, dict):
            return schema_obj

        if '$ref' in schema_obj:
            ref_path = schema_obj['$ref'].split('/')

            current_level = self.swagger_data  # Use self.swagger_data for resolving definitions
            try:
                if len(ref_path) > 1:  # Skip '#'
                    for part in ref_path[1:]:
                        if part in current_level:
                            current_level = current_level[part]
                        else:
                            raise KeyError(f"Path part '{part}' not found in ref.")
                resolved_schema = current_level
            except KeyError:
                logger.warning(f"Could not resolve $ref: {schema_obj['$ref']}. Returning empty object.")
                return {}  # Return empty dict if ref not found

            return self._deep_resolve_schema(
                resolved_schema)  # Recursively resolve any further $ref within the resolved schema

        # Recursively process properties for object schemas
        if 'properties' in schema_obj and isinstance(schema_obj['properties'], dict):
            for prop_name, prop_details in schema_obj['properties'].items():
                schema_obj['properties'][prop_name] = self.resolve_swagger_ref(prop_details)

        # Recursively process items for array schemas
        if 'items' in schema_obj and isinstance(schema_obj['items'], dict):
            schema_obj['items'] = self.resolve_swagger_ref(schema_obj['items'])

        return schema_obj

    def _deep_resolve_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper for recursive resolution. Ensures all sub-schemas are resolved."""
        if not isinstance(schema, dict):
            return schema

        if '$ref' in schema:
            return self.resolve_swagger_ref(schema)

        if 'properties' in schema:
            for key, value in schema['properties'].items():
                schema['properties'][key] = self._deep_resolve_schema(value)

        if 'items' in schema:
            schema['items'] = self._deep_resolve_schema(schema['items'])

        return schema

    def _deep_resolve_all_refs_in_spec(self):
        """
        Recursively resolves all $ref occurrences throughout the entire
        swagger_data dictionary (paths and definitions/components).
        Mutates self.swagger_data in place.
        """
        # First pass: Resolve all definitions/components schemas
        if 'definitions' in self.swagger_data:  # Swagger 2.0
            for def_name in list(self.swagger_data['definitions'].keys()):
                self.swagger_data['definitions'][def_name] = self.resolve_swagger_ref(
                    self.swagger_data['definitions'][def_name])
        elif 'components' in self.swagger_data and 'schemas' in self.swagger_data['components']:  # OpenAPI 3.0
            for schema_name in list(self.swagger_data['components']['schemas'].keys()):
                self.swagger_data['components']['schemas'][schema_name] = self.resolve_swagger_ref(
                    self.swagger_data['components']['schemas'][schema_name])

        # Second pass: Resolve references within paths
        if 'paths' in self.swagger_data:
            for path, path_data in self.swagger_data['paths'].items():
                for method, method_data in path_data.items():
                    if method.lower() in ["get", "post", "put", "delete", "patch"]:
                        # Parameters (path, query, header, body)
                        if 'parameters' in method_data:
                            resolved_params = []
                            for param in method_data['parameters']:
                                # Create a deep copy of the parameter to avoid issues, then resolve its schema
                                param_copy = json.loads(json.dumps(param))
                                if 'schema' in param_copy:
                                    param_copy['schema'] = self.resolve_swagger_ref(param_copy['schema'])
                                resolved_params.append(param_copy)
                            method_data['parameters'] = resolved_params  # Reassign the list with resolved params

                        # Request Body (OpenAPI 3.0 specific)
                        if 'requestBody' in method_data and 'content' in method_data['requestBody']:
                            for media_type, content_obj in method_data['requestBody']['content'].items():
                                if 'schema' in content_obj:
                                    content_obj['schema'] = self.resolve_swagger_ref(content_obj['schema'])

                        # Responses
                        if 'responses' in method_data:
                            for status_code, response_obj in method_data['responses'].items():
                                if 'schema' in response_obj:  # Swagger 2.0
                                    response_obj['schema'] = self.resolve_swagger_ref(response_obj['schema'])
                                if 'content' in response_obj:  # OpenAPI 3.0
                                    for media_type, content_obj in response_obj['content'].items():
                                        if 'schema' in content_obj:
                                            content_obj['schema'] = self.resolve_swagger_ref(content_obj['schema'])

    def get_full_swagger_spec(self) -> Dict[str, Any]:
        """Returns the fully resolved and complete Swagger specification."""
        return self.swagger_data

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        """
        Extract API endpoints from Swagger spec.
        Ensures the spec is fetched and resolved before extraction.
        """
        # self.swagger_data should now be populated by __init__ calling fetch_swagger_spec()

        endpoints = []
        paths = self.swagger_data.get('paths', {})

        for path, path_data in paths.items():
            for method, method_data in path_data.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    # These parameters and responses should already be resolved by _deep_resolve_all_refs_in_spec
                    parameters = method_data.get('parameters', [])
                    responses = method_data.get('responses', {})
                    summary = method_data.get('summary')
                    operation_id = method_data.get('operationId')
                    produces = method_data.get('produces', [])
                    consumes = method_data.get('consumes', [])

                    request_body_schema = None
                    if 'requestBody' in method_data:  # OpenAPI 3.0
                        content = method_data['requestBody'].get('content', {})
                        for media_type, media_type_schema in content.items():
                            if 'schema' in media_type_schema:
                                request_body_schema = media_type_schema['schema']
                                break
                    elif method.upper() in ['POST', 'PUT', 'PATCH']:  # Swagger 2.0 body parameter
                        for param in parameters:
                            if param.get('in') == 'body' and 'schema' in param:
                                request_body_schema = param['schema']
                                break

                    parsed_response_schemas = {}
                    for status_code, response_data in responses.items():
                        if 'schema' in response_data:  # Swagger 2.0
                            parsed_response_schemas[status_code] = response_data['schema']
                        elif 'content' in response_data:  # OpenAPI 3.0
                            for media_type, media_type_content in response_data['content'].items():
                                if 'schema' in media_type_content:
                                    parsed_response_schemas[status_code] = media_type_content['schema']
                                    break

                    endpoints.append(SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        parameters=parameters,
                        responses=responses,
                        summary=summary,
                        operation_id=operation_id,
                        produces=produces,
                        consumes=consumes,
                        request_body=request_body_schema,
                        parsed_response_schemas=parsed_response_schemas
                    ))

        return endpoints
