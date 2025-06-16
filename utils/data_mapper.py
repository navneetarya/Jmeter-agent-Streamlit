import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataMapper:
    """
    Maps API parameters and request body fields to database fields,
    using AI suggestions (inferred logic) and sampled data.
    """

    @staticmethod
    def suggest_mappings(endpoints: List[Any],  # Using Any to avoid circular import with SwaggerEndpoint in data_mapper
                         tables_schema: Dict[str, List[Dict[str, str]]],
                         db_sampled_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Suggests mappings between API parameters and database fields, prioritizing sampled DB data.
        Returns a nested dictionary: {endpoint_key: {param_name: {source: "...", value: "..."}}}
        """
        mappings = {}

        for endpoint in endpoints:
            endpoint_key = f"{endpoint.method} {endpoint.path}"
            mappings[endpoint_key] = {}

            # Process parameters from Swagger spec (path and query)
            for param in endpoint.parameters:
                if param.get('in') in ['query', 'path']:
                    param_name = param.get('name', '')
                    param_type = param.get('type', 'string')  # Default to string if type not found

                    mapping_info = DataMapper._determine_param_source(
                        param_name, param_type, tables_schema, db_sampled_data
                    )
                    mappings[endpoint_key][param_name] = mapping_info

            # Process parameters from request body schema (including nested properties)
            if endpoint.request_body:
                # Helper recursive function to process properties of a schema
                def process_body_schema_for_mapping(schema_part: Dict[str, Any], current_path: List[str]):
                    if not isinstance(schema_part, dict):
                        return

                    schema_type = schema_part.get('type', 'object')

                    if schema_type == 'object' and 'properties' in schema_part:
                        for prop_name, prop_details in schema_part['properties'].items():
                            full_param_name = ".".join(current_path + [prop_name])

                            if prop_details.get('type') == 'object':
                                process_body_schema_for_mapping(prop_details, current_path + [prop_name])
                            elif prop_details.get('type') == 'array':
                                # For arrays, we primarily map elements within the array if they are objects
                                # or if it's a primitive array, we try to map the array itself or its item type.
                                if 'items' in prop_details:
                                    if prop_details['items'].get('type') == 'object':
                                        # Recursively process items within an array of objects
                                        process_body_schema_for_mapping(prop_details['items'],
                                                                        current_path + [prop_name, "_item"])
                                    else:  # Array of primitives
                                        item_type = prop_details['items'].get('type', 'string')
                                        mapping_info = DataMapper._determine_param_source(
                                            full_param_name, item_type, tables_schema, db_sampled_data,
                                            is_body_param=True
                                        )
                                        mappings[endpoint_key][full_param_name] = mapping_info
                                else:  # Array with no items schema
                                    mapping_info = {"source": "Static/Dummy", "value": [], "type": "array"}
                                    mappings[endpoint_key][full_param_name] = mapping_info
                            else:  # Primitive type within an object
                                mapping_info = DataMapper._determine_param_source(
                                    full_param_name, prop_details.get('type', 'string'), tables_schema, db_sampled_data,
                                    is_body_param=True
                                )
                                mappings[endpoint_key][full_param_name] = mapping_info
                    elif schema_type == 'array' and 'items' in schema_part:
                        # If the entire request body is an array (e.g., createUsersWithList)
                        if schema_part['items'].get('type') == 'object':
                            process_body_schema_for_mapping(schema_part['items'], current_path + [
                                "_item"])  # Use a placeholder for array item path
                        else:  # Array of primitives at root level
                            item_type = schema_part['items'].get('type', 'string')
                            mapping_info = DataMapper._determine_param_source(
                                "".join(current_path), item_type, tables_schema, db_sampled_data, is_body_param=True
                            )
                            # Store it under the array's full path, indicating it applies to items
                            mappings[endpoint_key][".".join(current_path)] = mapping_info
                    elif current_path == []:  # Handle root level primitive body
                        mapping_info = DataMapper._determine_param_source(
                            "body", schema_part.get('type', 'string'), tables_schema, db_sampled_data,
                            is_body_param=True
                        )
                        mappings[endpoint_key]["body"] = mapping_info

                process_body_schema_for_mapping(endpoint.request_body, [])

        return mappings

    @staticmethod
    def _determine_param_source(param_name: str, param_type: str,
                                tables_schema: Dict[str, List[Dict[str, str]]],
                                db_sampled_data: Dict[str, pd.DataFrame],
                                is_body_param: bool = False) -> Dict[str, Any]:
        """Determines the best source for a parameter (DB, Generated, Static) and its value/placeholder."""
        param_name_lower = param_name.lower()

        # Strip path prefix (e.g., "category.id" becomes "id") for matching against DB columns
        clean_param_name_for_db_match = param_name_lower.split('.')[-1]

        # 1. Try to pull from DB (sampled data) - direct match (primary/foreign keys)
        for table_name, columns in tables_schema.items():
            for column in columns:
                column_name = column['name'].lower()
                # Check for exact match for primary/foreign keys
                if clean_param_name_for_db_match == column_name and (column.get('is_pk') or column.get('is_fk')):
                    if table_name in db_sampled_data and not db_sampled_data[table_name].empty and column['name'] in \
                            db_sampled_data[table_name].columns:
                        if is_body_param:  # For body, use a concrete sampled value
                            return {"source": "DB Sample (First Row)",
                                    "value": str(db_sampled_data[table_name][column['name']].iloc[0]),
                                    "type": param_type}
                        else:  # For path/query, use CSV variable
                            return {"source": "DB Sample (CSV)",
                                    "value": f"${{csv_{table_name.replace('.', '_')}_{column['name'].replace('.', '_')}}}",
                                    "type": param_type}

                # Check for `id` parameters matching `id` columns in related tables (e.g., `user_id` -> `users.id`)
                # This logic is already for the cleaned name
                if clean_param_name_for_db_match.endswith(
                        "_id") and column_name == "id" and clean_param_name_for_db_match.replace("_id",
                                                                                                 "") == table_name.lower():
                    if table_name in db_sampled_data and not db_sampled_data[table_name].empty and column['name'] in \
                            db_sampled_data[table_name].columns:
                        if is_body_param:
                            return {"source": "DB Sample (First Row)",
                                    "value": str(db_sampled_data[table_name][column['name']].iloc[0]),
                                    "type": param_type}
                        else:
                            return {"source": "DB Sample (CSV)",
                                    "value": f"${{csv_{table_name.replace('.', '_')}_{column['name'].replace('.', '_')}}}",
                                    "type": param_type}

        # 2. Try to pull from DB (sampled data) - partial match on clean name
        for table_name, columns in tables_schema.items():
            for column in columns:
                column_name = column['name'].lower()
                if clean_param_name_for_db_match in column_name or column_name in param_name_lower:  # Changed this condition slightly to use param_name_lower for broader match
                    if table_name in db_sampled_data and not db_sampled_data[table_name].empty and column['name'] in \
                            db_sampled_data[table_name].columns:
                        if is_body_param:
                            return {"source": "DB Sample (First Row)",
                                    "value": str(db_sampled_data[table_name][column['name']].iloc[0]),
                                    "type": param_type}
                        else:
                            return {"source": "DB Sample (CSV)",
                                    "value": f"${{csv_{table_name.replace('.', '_')}_{column['name'].replace('.', '_')}}}",
                                    "type": param_type}

        # 3. Auto-generate
        if "id" in clean_param_name_for_db_match or "uuid" in clean_param_name_for_db_match:
            return {"source": "Generated", "value": "${__UUID()}", "type": param_type}
        if "timestamp" in clean_param_name_for_db_match or "date" in clean_param_name_for_db_match:
            return {"source": "Generated", "value": "${__time()}",
                    "type": param_type}  # JMeter timestamp function (epoch milliseconds)

        # 4. Static value / Dummy (based on param_type)
        if param_type == 'string':
            return {"source": "Static/Dummy", "value": f"dummy_{clean_param_name_for_db_match}", "type": param_type}
        elif param_type == 'integer':
            return {"source": "Static/Dummy", "value": 123, "type": param_type}  # Changed to int for number types
        elif param_type == 'boolean':
            return {"source": "Static/Dummy", "value": True, "type": param_type}  # Changed to bool for boolean types
        elif param_type == 'array':
            # For arrays, provide a dummy array. The recursive builder will populate items.
            return {"source": "Static/Dummy", "value": [], "type": param_type}  # Default to empty list
        elif param_type == 'object':
            # For objects, provide a dummy empty object. The recursive builder will populate properties.
            return {"source": "Static/Dummy", "value": {}, "type": param_type}

        # 5. Fallback - No Match Found
        return {"source": "No Match Found", "value": "<<NO_MATCH_FOUND>>", "type": param_type}

