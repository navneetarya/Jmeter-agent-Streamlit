import logging
from typing import Dict, List, Any, Optional  # Import Optional
import pandas as pd
import re

logger = logging.getLogger(__name__)


# Assuming SwaggerEndpoint is defined in swagger_parser or imported
# from .swagger_parser import SwaggerEndpoint # Uncomment if not already imported in app.py or if testing independently

class DataMapper:
    """
    Suggests mappings between Swagger API parameters and database table columns.
    It attempts to find best-fit matches based on naming conventions and provides
    different mapping sources (DB Sample, Generated, Static).
    """

    @staticmethod
    def suggest_mappings(swagger_endpoints: List[Any],
                         # Using Any to avoid circular import, assume SwaggerEndpoint structure
                         db_tables_schema: Dict[str, List[Dict[str, Any]]],
                         db_sampled_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Suggests data mappings for Swagger endpoint parameters from database schema and sampled data.
        Prioritizes exact name matches and common ID patterns.
        """
        mappings = {}
        for endpoint in swagger_endpoints:
            endpoint_key = f"{endpoint.method} {endpoint.path}"
            mappings[endpoint_key] = {}

            # Process parameters (path, query, header)
            for param in endpoint.parameters:
                param_name = param.get('name')
                if not param_name:
                    continue

                suggested_value, source = DataMapper._find_matching_db_column(
                    param_name, db_tables_schema, db_sampled_data
                )

                if suggested_value is not None:
                    mappings[endpoint_key][param_name] = {
                        "source": source,
                        "value": suggested_value,
                        "type": param.get('type')  # Store original swagger type for conversion later
                    }
                else:
                    # Provide a generic dummy or generated value if no DB match
                    mappings[endpoint_key][param_name] = {
                        "source": "Generated Value",
                        "value": DataMapper._generate_dummy_value(param.get('type')),
                        "type": param.get('type')
                    }
                logger.debug(f"Mapping for {endpoint_key} param '{param_name}': {mappings[endpoint_key][param_name]}")

            # Process request body parameters (if applicable, typically for POST/PUT)
            # The body schema is potentially nested, so a recursive approach is needed.
            if endpoint.body_schema:
                DataMapper._suggest_recursive_body_mappings(
                    endpoint.body_schema, endpoint_key, db_tables_schema, db_sampled_data, mappings[endpoint_key], []
                )
        return mappings

    @staticmethod
    def _suggest_recursive_body_mappings(schema: Dict[str, Any], endpoint_key: str,
                                         db_tables_schema: Dict[str, List[Dict[str, Any]]],
                                         db_sampled_data: Dict[str, pd.DataFrame],
                                         current_endpoint_mappings: Dict[str, Any],
                                         path_segments: List[str]):
        """
        Recursively suggests mappings for fields within a nested request body schema.
        `path_segments` keeps track of the current JSON path (e.g., ["user", "address", "street"]).
        """
        schema_type = schema.get('type', 'object')

        if schema_type == 'object':
            properties = schema.get('properties', {})
            for prop_name, prop_details in properties.items():
                full_param_name_path = path_segments + [prop_name]
                full_param_name_str = ".".join(full_param_name_path)  # e.g., "user.address.street"

                # Check if this property has already been explicitly mapped during prior processing
                if full_param_name_str in current_endpoint_mappings:
                    continue

                if prop_details.get('type') == 'object' or prop_details.get('type') == 'array':
                    # Recurse for nested structures
                    DataMapper._suggest_recursive_body_mappings(
                        prop_details, endpoint_key, db_tables_schema, db_sampled_data,
                        current_endpoint_mappings, full_param_name_path
                    )
                else:
                    # Try to find a DB match for primitive properties
                    suggested_value, source = DataMapper._find_matching_db_column(
                        prop_name, db_tables_schema, db_sampled_data
                    )
                    if suggested_value is not None:
                        current_endpoint_mappings[full_param_name_str] = {
                            "source": source,
                            "value": suggested_value,
                            "type": prop_details.get('type')
                        }
                    else:
                        # Default to generated value for primitive types if no DB match
                        current_endpoint_mappings[full_param_name_str] = {
                            "source": "Generated Value",
                            "value": DataMapper._generate_dummy_value(prop_details.get('type')),
                            "type": prop_details.get('type')
                        }
        elif schema_type == 'array' and 'items' in schema:
            # For arrays, we try to map the elements within the array.
            # This is simplified: it assumes elements of the array are of a consistent type/schema.
            # The key for array items is typically just the array name itself, or a combined path
            # to signify the 'item' within the array.

            # The mapping for the entire array (if it's simple, e.g., array of strings mapped to a single CSV column)
            # would be at the current `path_segments` level.
            full_param_name_str = ".".join(path_segments)

            # Check for direct mapping of the array itself (e.g., to a CSV list)
            suggested_value, source = DataMapper._find_matching_db_column(
                path_segments[-1] if path_segments else "",  # Use last segment for array name
                db_tables_schema, db_sampled_data,
                is_array=True
            )
            if suggested_value is not None:
                current_endpoint_mappings[full_param_name_str] = {
                    "source": source,
                    "value": suggested_value,  # This would ideally be a list of values
                    "type": schema.get('type')
                }
                logger.debug(f"Array '{full_param_name_str}' mapped directly to: {suggested_value}")
            else:
                # If no direct array mapping, try to map properties within the array items
                # We append "_item" to the path to indicate mapping for array elements
                item_path_segments = path_segments + ["_item"]
                DataMapper._suggest_recursive_body_mappings(
                    schema['items'], endpoint_key, db_tables_schema, db_sampled_data,
                    current_endpoint_mappings, item_path_segments
                )
        else:  # Primitive type at current level
            full_param_name_str = ".".join(path_segments)
            if full_param_name_str and full_param_name_str not in current_endpoint_mappings:
                suggested_value, source = DataMapper._find_matching_db_column(
                    full_param_name_str, db_tables_schema, db_sampled_data
                )
                if suggested_value is not None:
                    current_endpoint_mappings[full_param_name_str] = {
                        "source": source,
                        "value": suggested_value,
                        "type": schema.get('type')
                    }
                else:
                    current_endpoint_mappings[full_param_name_str] = {
                        "source": "Generated Value",
                        "value": DataMapper._generate_dummy_value(schema.get('type')),
                        "type": schema.get('type')
                    }

    @staticmethod
    def _find_matching_db_column(param_name: str,
                                 db_tables_schema: Dict[str, List[Dict[str, Any]]],
                                 db_sampled_data: Dict[str, pd.DataFrame],
                                 is_array: bool = False) -> (Any, str):
        """
        Finds a matching database column for a given API parameter name.
        Prioritizes exact matches, then case-insensitive matches, then common ID patterns.
        Returns the JMeter variable string (e.g., "${csv_users_id}") or sampled value, and source.
        """
        lower_param_name = param_name.lower()

        # Check for direct matches (case-insensitive)
        for table_name, columns_schema in db_tables_schema.items():
            for col_schema in columns_schema:
                col_name = col_schema['name']
                col_type = col_schema['type']

                # Exact match or common ID/name match
                if lower_param_name == col_name.lower():
                    # For arrays, if we have sampled data as a list, return that directly.
                    # Otherwise, provide a CSV variable for the first element.
                    if is_array and table_name in db_sampled_data and col_name in db_sampled_data[table_name].columns:
                        # Assuming for array, we might want the whole list of sampled data
                        return db_sampled_data[table_name][col_name].tolist(), "DB Sample (Array)"
                    elif table_name in db_sampled_data and col_name in db_sampled_data[table_name].columns and not \
                    db_sampled_data[table_name].empty:
                        # Return JMeter CSV variable reference
                        return f"${{csv_{table_name}_{col_name}}}", "DB Sample (CSV)"
                    else:
                        return f"dummy_{param_name}", "Generated Value"  # No data in sampled table, fall back

                # Check for common ID/name patterns
                if ("id" in lower_param_name and "id" in col_name.lower()) or \
                        ("name" in lower_param_name and "name" in col_name.lower()) or \
                        ("status" in lower_param_name and "status" in col_name.lower()):
                    if col_schema.get('is_primary_key') or col_schema.get('is_foreign_key'):
                        if table_name in db_sampled_data and col_name in db_sampled_data[table_name].columns and not \
                        db_sampled_data[table_name].empty:
                            return f"${{csv_{table_name}_{col_name}}}", "DB Sample (CSV)"

        # Check for special cases like "username" -> "users" table "username" column
        if lower_param_name == "username" and "users" in db_tables_schema:
            for col_schema in db_tables_schema["users"]:
                if col_schema['name'].lower() == "username" and "username" in db_sampled_data["users"].columns:
                    return f"${{csv_users_username}}", "DB Sample (CSV)"
        if lower_param_name == "password" and "users" in db_tables_schema:
            for col_schema in db_tables_schema["users"]:
                if col_schema['name'].lower() == "password" and "password" in db_sampled_data["users"].columns:
                    return f"${{csv_users_password}}", "DB Sample (CSV)"

        # Fallback to generating based on data type if no direct DB match found
        return None, "Generated Value"

    @staticmethod
    def _generate_dummy_value(param_type: Optional[str]) -> Any:
        """Generates a dummy value based on a given parameter type."""
        if param_type == 'string':
            return "dummy_string"
        elif param_type == 'integer':
            return 123
        elif param_type == 'boolean':
            return True
        elif param_type == 'number':
            return 123.45
        elif param_type == 'array':
            return []  # For array, just return empty list as dummy
        elif param_type == 'object':
            return {}  # For object, just return empty dict as dummy
        else:
            return "dummy_value"

