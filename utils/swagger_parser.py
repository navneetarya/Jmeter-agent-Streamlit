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
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    body_schema: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Converts the SwaggerEndpoint object to a dictionary, suitable for JSON serialization."""
        # Note: We're not deeply copying here for simplicity, direct access is fine for serialization.
        # If any fields were non-primitive objects that needed custom serialization, that would go here.
        return {
            "method": self.method,
            "path": self.path,
            "operation_id": self.operation_id,
            "summary": self.summary,
            "parameters": self.parameters,
            "body_schema": self.body_schema,
            "responses": self.responses
        }


class SwaggerParser:
    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_data: Dict[str, Any] = {}
        self._fetch_swagger_spec()

    def _fetch_swagger_spec(self):
        """Fetches the Swagger/OpenAPI specification from the given URL."""
        try:
            response = requests.get(self.swagger_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            self.swagger_data = response.json()
            logger.info(f"Successfully fetched Swagger spec from {self.swagger_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Swagger spec from {self.swagger_url}: {e}")
            raise ConnectionError(f"Could not connect to Swagger URL or invalid response: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Swagger spec at {self.swagger_url}: {e}")
            raise ValueError(f"Invalid JSON in Swagger spec: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching Swagger spec: {e}")
            raise

    def get_full_swagger_spec(self) -> Dict[str, Any]:
        """Returns the full fetched Swagger/OpenAPI specification."""
        return self.swagger_data

    def _resolve_ref(self, ref_path: str) -> Optional[Dict[str, Any]]:
        """Resolves a JSON schema $ref reference within the Swagger data."""
        # Example ref_path: '#/definitions/Pet'
        parts = ref_path.split('/')
        current_data = self.swagger_data
        for part in parts[1:]:  # Skip '#'
            if part in current_data:
                current_data = current_data[part]
            else:
                logger.warning(f"Reference part '{part}' not found in {ref_path}")
                return None
        return current_data

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        """Extracts API endpoints with their methods, paths, and relevant details."""
        endpoints: List[SwaggerEndpoint] = []
        paths = self.swagger_data.get('paths', {})

        for path, methods_data in paths.items():
            for method, details in methods_data.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    operation_id = details.get('operationId')
                    summary = details.get('summary')

                    parameters_raw = details.get('parameters', [])
                    parameters_processed: List[Dict[str, Any]] = []
                    body_schema_resolved: Optional[Dict[str, Any]] = None

                    for param in parameters_raw:
                        if '$ref' in param:
                            resolved_param = self._resolve_ref(param['$ref'])
                            if resolved_param:
                                parameters_processed.append(resolved_param)
                                # If it's a body parameter that was a $ref, extract its schema
                                if resolved_param.get('in') == 'body' and 'schema' in resolved_param:
                                    body_schema_raw = resolved_param['schema']
                                    if '$ref' in body_schema_raw:
                                        body_schema_resolved = self._resolve_ref(body_schema_raw['$ref'])
                                    else:
                                        body_schema_resolved = body_schema_raw
                            else:
                                logger.warning(f"Could not resolve parameter reference: {param['$ref']}")
                                parameters_processed.append(param)  # Include raw if resolution fails
                        else:
                            parameters_processed.append(param)
                            # Directly extract body schema if 'in' is 'body'
                            if param.get('in') == 'body' and 'schema' in param:
                                body_schema_raw = param['schema']
                                if '$ref' in body_schema_raw:
                                    body_schema_resolved = self._resolve_ref(body_schema_raw['$ref'])
                                else:
                                    body_schema_resolved = body_schema_raw

                    # Handle 'requestBody' for OpenAPI 3.0+ (Swagger 2.0 uses 'body' parameter)
                    if 'requestBody' in details:
                        request_body = details['requestBody']
                        content_types = request_body.get('content', {})
                        if 'application/json' in content_types:
                            schema_raw = content_types['application/json'].get('schema')
                            if schema_raw:
                                if '$ref' in schema_raw:
                                    body_schema_resolved = self._resolve_ref(schema_raw['$ref'])
                                else:
                                    body_schema_resolved = schema_raw
                        # Add more content types if needed (e.g., application/xml, application/x-www-form-urlencoded)

                    # Capture responses for potential extraction/assertions
                    responses_data = {}
                    for status_code, response_obj in details.get('responses', {}).items():
                        # Resolve $ref for response schemas if present
                        if 'schema' in response_obj and '$ref' in response_obj['schema']:
                            response_obj['schema'] = self._resolve_ref(response_obj['schema']['$ref'])
                        responses_data[status_code] = response_obj

                    endpoints.append(
                        SwaggerEndpoint(
                            method=method.upper(),
                            path=path,
                            operation_id=operation_id,
                            summary=summary,
                            parameters=parameters_processed,
                            body_schema=body_schema_resolved,
                            responses=responses_data
                        )
                    )
        logger.info(f"Extracted {len(endpoints)} API endpoints.")
        return endpoints

