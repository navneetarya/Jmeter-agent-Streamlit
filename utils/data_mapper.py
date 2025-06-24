import logging
import json
from typing import Dict, List, Any
from groq import Groq

logger = logging.getLogger(__name__)


class DataMapper:
    @staticmethod
    def get_ai_powered_mappings(
            all_params: List[Dict[str, Any]],
            db_schema: Dict[str, List[Dict[str, Any]]],
            api_key: str
    ) -> Dict[str, Any]:
        """
        Calls an LLM to perform smart, semantic mapping between a list of API parameters and DB columns.
        This prompt is highly optimized to be small and fast.
        """
        db_schema_for_prompt = {table: [col['name'] for col in columns] for table, columns in db_schema.items()}

        prompt = f"""
        ## Task
        You are a data architect. Your task is to map the given API parameters to the most semantically appropriate database columns. You must handle abbreviations and different naming conventions (e.g., camelCase vs. snake_case).

        ## API Parameters to Map
        ```json
        {json.dumps(all_params, indent=2)}
        ```

        ## Available Database Schema
        ```json
        {json.dumps(db_schema_for_prompt, indent=2)}
        ```

        ## Instructions
        1.  Analyze each API parameter semantically.
        2.  Find the single best database table and column that matches its meaning.
            -   **Example 1:** "appcd" or "application_code" should map to a column named "ApplicationCD".
            -   **Example 2:** "userId" should map to a "User" table's "UserID" column.
        3.  If no logical database column exists, you MUST map its `mapped_table` and `mapped_column` to `null`.
        4.  Return a single JSON object with one key, "parameter_mappings". The value is an array of objects, each containing the original parameter name and its mapped table and column.

        ## Output Schema (JSON Only)
        ```json
        {{
            "parameter_mappings": [
                {{"parameter_name": "userId", "mapped_table": "User", "mapped_column": "UserID"}},
                {{"parameter_name": "appcd", "mapped_table": "Client", "mapped_column": "ApplicationCD"}}
            ]
        }}
        ```
        Respond with ONLY the raw JSON object and nothing else.
        """

        client = Groq(api_key=api_key)
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error getting AI-powered mappings: {e}")
            return {"parameter_mappings": []}