### **Project Master Prompt: AI-Powered JMeter Script Generator**

**Project Overview:**
This document contains the complete and final state of the AI-Powered JMeter Script Generator project. The goal is to create a robust Streamlit application that uses a "Code-First, AI-Assisted" hybrid model to generate complex, data-driven JMeter test plans. The user provides an API specification, a database schema, and sample database data. The application then uses a single, focused AI call to perform semantic data mapping and uses deterministic Python code to assemble the final test plan and all associated artifacts.

**Final Architecture (Code-First, AI-Assist Hybrid Model):**
This architecture was chosen to be resilient against the token and rate limits of free-tier LLM APIs, following a rigorous debugging process.

1.  **Step 1: Code-Driven Endpoint Filtering & Parameter Collection (in `app.py`):** The application first reads the uploaded Swagger file using Python. It reliably filters the endpoints based on the user's text prompt (e.g., "all GET endpoints"). It then programmatically collects all unique parameter names from these filtered endpoints. This step uses no AI and is 100% accurate.

2.  **Step 2: Focused AI-Powered Data Mapping (in `utils/data_mapper.py`):** The application makes a **single, small, and fast AI call**.
    *   **Input:** Only the list of unique parameter names and the database schema.
    *   **Task:** The AI's only job is to return a JSON dictionary mapping the parameter names to the most semantically appropriate database table and column (e.g., mapping `appcd` to `Client.ApplicationCD`).
    *   **Benefit:** This prompt is extremely small and avoids all API token and rate limit errors.

3.  **Step 3: Code-Driven Test Case Assembly (in `app.py`):** With the AI's mapping dictionary, the application then programmatically loops through the filtered endpoints from Step 1. For each endpoint, it builds a complete test case object, including:
    *   A unique name in the format `METHOD - /path/to/endpoint`.
    *   Default headers (e.g., `Content-Type`).
    *   Placeholders for authentication (e.g., `Authorization: Bearer ${authToken}`).
    *   Default assertions (e.g., Response Code 200).
    *   Placeholder extractions for correlation.
    *   The correctly formatted `${csv_TableName_ColumnName}` variables for all mapped parameters.

4.  **Step 4: Artifact Generation (in `utils/jmeter_generator.py`):** The final, code-assembled test plan is passed to the JMX generator, which translates it into the `test_plan.jmx`, all necessary `.csv` files, and the YAML/JSON design documents. The JMX is generated with a compliant structure to avoid GUI errors in JMeter.
