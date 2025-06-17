import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwaggerEndpoint:
    """Represents a single API endpoint extracted from a Swagger/OpenAPI specification."""
    method: str  # e.g., 'get', 'post', 'put', 'delete'
    path: str  # e.g., '/pets/{id}'
    summary: Optional[str] = None
    description: Optional[str] = None
    operation_id: Optional[str] = None  # Added operation_id
    parameters: List[Dict[str, Any]] = field(default_factory=list)  # Includes path, query, header, formData parameters
    body_schema: Optional[
        Dict[str, Any]] = None  # For 'body' parameter schema (Swagger 2.0) or requestBody content schema (OpenAPI 3.0)
    responses: Dict[str, Any] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)  # For Swagger 2.0
    produces: List[str] = field(default_factory=list)  # For Swagger 2.0
    tags: List[str] = field(default_factory=list)


class SwaggerParser:
    """Parses a Swagger/OpenAPI specification to extract API endpoint details."""

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_data: Dict[str, Any] = self._fetch_swagger_spec()

    def _fetch_swagger_spec(self) -> Dict[str, Any]:
        """Fetches the Swagger/OpenAPI JSON specification from the given URL."""
        logger.info(f"Fetching Swagger spec from: {self.swagger_url}")
        try:
            response = requests.get(self.swagger_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            swagger_data = response.json()
            logger.info("Swagger spec fetched successfully.")
            return swagger_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Swagger spec from {self.swagger_url}: {e}")
            st.error(f"Error fetching Swagger spec: {e}. Please check the URL and your internet connection.")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding Swagger spec JSON from {self.swagger_url}: {e}")
            st.error(f"Error decoding Swagger spec: The response is not valid JSON. {e}")
            raise

    def get_full_swagger_spec(self) -> Dict[str, Any]:
        """Returns the full parsed Swagger/OpenAPI specification."""
        return self.swagger_data

    def _resolve_ref(self, ref_path: str) -> Optional[Dict[str, Any]]:
        """Resolves a JSON $ref path within the Swagger data."""
        parts = ref_path.split('/')
        # Remove '#', as the path is relative to the root of the document
        if parts[0] == '#':
            parts = parts[1:]

        current_data = self.swagger_data
        for part in parts:
            if part in current_data:
                current_data = current_data[part]
            else:
                logger.warning(f"Could not resolve reference part: {part} in {ref_path}")
                return None
        return current_data

    def _get_schema_from_param(self, param: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extracts the schema from a parameter definition, resolving $ref if necessary.
        Handles both Swagger 2.0 (param['schema']) and OpenAPI 3.0 (param['content']['application/json']['schema']).
        """
        if '$ref' in param:
            return self._resolve_ref(param['$ref'])
        if 'schema' in param:  # Swagger 2.0 body parameter
            if '$ref' in param['schema']:
                return self._resolve_ref(param['schema']['$ref'])
            return param['schema']

        # OpenAPI 3.0 requestBody structure
        if 'requestBody' in param:
            request_body = param['requestBody']
            if 'content' in request_body:
                # Try application/json first, then other content types
                for content_type in ['application/json', 'application/xml', 'application/x-www-form-urlencoded',
                                     'text/plain']:
                    if content_type in request_body['content']:
                        if 'schema' in request_body['content'][content_type]:
                            schema = request_body['content'][content_type]['schema']
                            if '$ref' in schema:
                                return self._resolve_ref(schema['$ref'])
                            return schema

        # For parameters that are not body/requestBody but might have inline schema
        if 'type' in param:
            return param  # Parameter itself defines the schema for simple cases

        return None

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        """
        Extracts and structures all API endpoints from the loaded Swagger data.
        Handles both Swagger 2.0 and OpenAPI 3.0 spec differences.
        """
        endpoints: List[SwaggerEndpoint] = []
        paths = self.swagger_data.get('paths', {})

        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']:
                    parameters = []
                    body_schema = None  # Will store the resolved schema for the request body

                    # Process parameters (path, query, header, formData)
                    # For OpenAPI 3.0, parameters can be directly under operation or in components/parameters
                    # For Swagger 2.0, they are directly under operation
                    if 'parameters' in details:
                        for param_def in details['parameters']:
                            if '$ref' in param_def:
                                resolved_param_def = self._resolve_ref(param_def['$ref'])
                                if resolved_param_def:
                                    param_def = resolved_param_def
                                else:
                                    logger.warning(f"Could not resolve parameter reference: {param_def['$ref']}")
                                    continue  # Skip if ref cannot be resolved

                            # Handle body parameter specifically for Swagger 2.0
                            if param_def.get('in') == 'body':
                                body_schema = self._get_schema_from_param(param_def)
                            else:
                                parameters.append(param_def)

                    # Handle requestBody for OpenAPI 3.0
                    if 'requestBody' in details:
                        request_body_content = details['requestBody'].get('content', {})
                        for content_type, media_type_obj in request_body_content.items():
                            if 'schema' in media_type_obj:
                                schema = media_type_obj['schema']
                                if '$ref' in schema:
                                    body_schema = self._resolve_ref(schema['$ref'])
                                else:
                                    body_schema = schema
                                break  # Take the first schema found (e.g., application/json)

                    # Handle responses
                    responses = {}
                    if 'responses' in details:
                        for status_code, response_obj in details['responses'].items():
                            if '$ref' in response_obj:
                                resolved_response_obj = self._resolve_ref(response_obj['$ref'])
                                if resolved_response_obj:
                                    responses[status_code] = resolved_response_obj
                                else:
                                    logger.warning(f"Could not resolve response reference: {response_obj['$ref']}")
                            else:
                                responses[status_code] = response_obj

                    endpoint = SwaggerEndpoint(
                        method=method.upper(),
                        path=path,
                        summary=details.get('summary'),
                        description=details.get('description'),
                        operation_id=details.get('operationId'),  # Extracted operationId
                        parameters=parameters,
                        body_schema=body_schema,
                        responses=responses,
                        security=details.get('security', []),
                        consumes=details.get('consumes', []),
                        produces=details.get('produces', []),
                        tags=details.get('tags', [])
                    )
                    endpoints.append(endpoint)
        return endpoints

