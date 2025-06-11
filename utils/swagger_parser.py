import requests
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwaggerEndpoint:
    path: str
    method: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]


class SwaggerParser:
    def __init__(self, swagger_url: str):
        self.swagger_url = swagger_url
        self.swagger_spec = None

    def fetch_swagger_spec(self) -> Dict[str, Any]:
        try:
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()
            self.swagger_spec = response.json()
            return self.swagger_spec
        except Exception as e:
            logger.error(f"Failed to fetch Swagger spec: {e}")
            raise

    def extract_endpoints(self) -> List[SwaggerEndpoint]:
        if not self.swagger_spec:
            self.fetch_swagger_spec()

        endpoints = []
        paths = self.swagger_spec.get('paths', {})

        for path, path_data in paths.items():
            for method, method_data in path_data.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    parameters = method_data.get('parameters', [])
                    responses = method_data.get('responses', {})

                    endpoints.append(SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        parameters=parameters,
                        responses=responses
                    ))

        return endpoints