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
        This prompt is highly optimized to handle naming conventions and avoid inventing columns.
        """
        db_schema_for_prompt = {table: [col['name'] for col in columns] for table, columns in db_schema.items()}

        prompt = f"""
        ## Your Task
        You are an expert data mapping system. Your task is to map API parameters to the most semantically appropriate database columns from the provided schema.

        ## API Parameters to Map
        ```json
        {json.dumps(all_params, indent=2)}
        ```

        ## Available Database Schema (Tables and their EXACT columns)
        ```json
        {json.dumps(db_schema_for_prompt, indent=2)}
        ```

        ## CRITICAL Instructions & Rules
        1.  For each API parameter, find the single best database TABLE and COLUMN that matches its meaning.
        2.  **You MUST aggressively handle naming conventions.** For example:
            -   `appcd` (camelCase/abbreviation) should map to `Client.ClientID`.
            -   `user_id` (snake_case) should map to `User.UserID`.
            -   `referrer_id` (snake_case) should map to `UserDetail.ReferrerID` (PascalCase).
        3.  **DO NOT INVENT COLUMNS.** If you identify a likely table but NONE of its available columns are a good semantic match, you MUST NOT create a new column.
        4.  If no suitable column exists within the most likely table, or if no table is a good match, you MUST set `mapped_table` and `mapped_column` to `null`. This is a mandatory rule.
        5.  Your entire output must be a single, raw JSON object. Do not add any commentary.

        ## Required Output Format (JSON only)
        ```json
        {{
            "parameter_mappings": [
                {{"parameter_name": "string", "mapped_table": "string or null", "mapped_column": "string or null"}}
            ]
        }}
        ```
        """

        client = Groq(api_key=api_key)
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error getting AI-powered mappings: {e}", exc_info=True)
            return {"parameter_mappings": []}