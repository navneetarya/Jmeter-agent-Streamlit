import requests
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwaggerEndpoint:
    path: str
    method: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]
    # New fields for more detailed information
    request_headers: Dict[str, str] = field(default_factory=dict)
    security_schemes: List[Dict[str, Any]] = field(default_factory=list)
    required_parameters: List[str] = field(default_factory=list)
    success_responses: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    description: Optional[str] = None


class SwaggerParser:
    """Parse Swagger/OpenAPI specifications"""

    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_spec = None

    def fetch_swagger_spec(self) -> Dict[str, Any]:
        """Fetch and parse Swagger specification"""
        try:
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()
            self.swagger_spec = response.json()
            logger.info(f"Successfully fetched Swagger spec from {self.swagger_url}")
            return self.swagger_spec
        except Exception as e:
            logger.error(f"Failed to fetch Swagger spec from {self.swagger_url}: {e}")
            raise

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        """Extract API endpoints from Swagger spec with more details"""
        if not self.swagger_spec:
            try:
                self.fetch_swagger_spec()
            except Exception as e:
                logger.error(f"Cannot extract endpoints, failed to fetch Swagger spec: {e}")
                return []

        endpoints = []
        paths = self.swagger_spec.get('paths', {})
        security_definitions = self.swagger_spec.get('securityDefinitions', {})  # For Swagger 2.0
        components_security_schemes = self.swagger_spec.get('components', {}).get('securitySchemes',
                                                                                  {})  # For OpenAPI 3.x

        for path, path_data in paths.items():
            for method, method_data in path_data.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    parameters = method_data.get('parameters', [])
                    responses = method_data.get('responses', {})
                    summary = method_data.get('summary')
                    description = method_data.get('description')

                    # Extract required parameters
                    required_params = [
                        p.get('name') for p in parameters if p.get('required')
                    ]

                    # Extract success responses (e.g., 200, 201, 204)
                    success_resp = {}
                    for status_code, resp_data in responses.items():
                        if status_code.startswith('2'):
                            success_resp[status_code] = resp_data

                    # Extract request headers from consumes or explicit headers
                    req_headers = {}
                    if 'consumes' in method_data and method_data['consumes']:
                        req_headers['Content-Type'] = method_data['consumes'][0]

                    # Also check for parameters 'in': 'header'
                    for param in parameters:
                        if param.get('in') == 'header':
                            req_headers[param.get('name')] = f"{{{{{param.get('name')}}}}}"  # Placeholder for JMeter

                    # Extract security schemes for the endpoint
                    endpoint_security = []
                    # Swagger 2.0 security
                    if 'security' in method_data:
                        for security_req in method_data['security']:
                            for scheme_name in security_req.keys():
                                if scheme_name in security_definitions:
                                    endpoint_security.append({
                                        "name": scheme_name,
                                        "type": security_definitions[scheme_name].get('type'),
                                        "in": security_definitions[scheme_name].get('in'),
                                        "param_name": security_definitions[scheme_name].get('name')
                                    })
                    # OpenAPI 3.x security
                    elif 'security' in self.swagger_spec:  # Check global security if endpoint specific is not found
                        for security_req in self.swagger_spec['security']:
                            for scheme_name in security_req.keys():
                                if scheme_name in components_security_schemes:
                                    endpoint_security.append({
                                        "name": scheme_name,
                                        "type": components_security_schemes[scheme_name].get('type'),
                                        "scheme": components_security_schemes[scheme_name].get('scheme'),
                                        "bearerFormat": components_security_schemes[scheme_name].get('bearerFormat'),
                                        "in": components_security_schemes[scheme_name].get('in'),  # for apiKey
                                        "param_name": components_security_schemes[scheme_name].get('name')  # for apiKey
                                    })

                    endpoints.append(SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        parameters=parameters,
                        responses=responses,
                        request_headers=req_headers,
                        security_schemes=endpoint_security,
                        required_parameters=required_params,
                        success_responses=success_resp,
                        summary=summary,
                        description=description
                    ))

        logger.info(f"Extracted {len(endpoints)} detailed API endpoints.")
        return endpoints

