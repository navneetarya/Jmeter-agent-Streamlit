import json
import dataclasses
from typing import List, Dict, Any, Optional
import os  # Added import for the 'os' module


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
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = dataclasses.field(default_factory=dict)
    produces: List[str] = dataclasses.field(default_factory=list)


class SwaggerParser:
    """
    Parses a Swagger/OpenAPI JSON file to extract API endpoints.
    """

    def __init__(self, swagger_file_path: str):
        self.swagger_file_path = swagger_file_path
        self.swagger_data = self._load_swagger_file()

    def _load_swagger_file(self) -> Dict[str, Any]:
        """Loads the Swagger JSON file."""
        if not os.path.exists(self.swagger_file_path):
            raise FileNotFoundError(f"Swagger file not found at: {self.swagger_file_path}")
        with open(self.swagger_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_swagger(self) -> List[SwaggerEndpoint]:
        """
        Parses the loaded Swagger data and extracts SwaggerEndpoint objects.
        """
        endpoints = []
        if "paths" not in self.swagger_data:
            return endpoints

        for path, methods in self.swagger_data["paths"].items():
            for method, details in methods.items():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    endpoint = SwaggerEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=details.get("summary"),
                        operation_id=details.get("operationId"),
                        parameters=details.get("parameters", []),
                        responses=details.get("responses", {}),
                        produces=details.get("produces", [])
                    )
                    # Extract request body if present (for OpenAPI 2.0, it's typically in parameters as 'in: body')
                    for param in endpoint.parameters:
                        if param.get("in") == "body" and "schema" in param:
                            endpoint.request_body = param["schema"]
                            break  # Assume only one body parameter

                    endpoints.append(endpoint)
        return endpoints
