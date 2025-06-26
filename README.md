## **Project Master Prompt: Intelligent JMeter Test Builder (Final Version)**

### **I. Project Goal & Core Philosophy**

The primary objective is to create an expert-level, AI-assisted Streamlit application that automates the creation of sophisticated, 99% production-ready JMeter test plans. The tool bridges the gap between API specifications and a fully functional performance test by intelligently handling data, logic, and JMeter-specific configurations.

The core philosophy is a **"Code-First, AI-Assist Hybrid Model"**. This means:
*   **Deterministic Code First:** All structural, repeatable, and rule-based tasks (parsing, JMX generation, UI logic) are handled by robust Python code. This ensures reliability and consistency.
*   **Targeted AI Assistance:** Artificial Intelligence is used *only* for tasks requiring semantic understanding and pattern recognition that are difficult to code deterministically (e.g., mapping `user_id` to `User.UserID`, suggesting correlations). This minimizes AI "hallucinations" and API dependency.

### **II. Finalized Feature Set & User Workflow**

The application is structured into a clear, multi-step workflow designed for performance engineers:

**Step 1: Design the Test Scenario (Dual-Mode)**
*   **UI:** A clean, two-panel layout.
*   **Auto-Build Mode:** A primary feature where the user clicks one button (`🚀 Auto-Build Intelligent Scenario`). The application analyzes all available APIs from the Swagger file and constructs a logically ordered scenario based on RESTful best practices (CRUD order: POSTs before GETs, GETs before PUTs/DELETEs). This provides an excellent starting point.
*   **Manual Build Mode:** For users requiring precise control, they can switch to a manual mode. Here, available endpoints are grouped by their API module (e.g., "User", "Client") in expandable sections. The user can add any endpoint to the scenario one by one.
*   **Full Control:** Regardless of the build mode, the user has complete control over the "Current Scenario Steps". They can **reorder** any step up or down and **remove** any step with intuitive buttons (⬆️, ⬇️, 🗑️).

**Step 2: Configure Logic (Correlation & Assertions)**
*   **Automatic Correlation Suggestions:** After the scenario is built, the application automatically analyzes the sequence of steps. It identifies common patterns (e.g., a POST followed by a GET with a path parameter like `/{id}/`) and suggests potential correlations.
*   **Correlation UI:**
    *   Each suggestion is displayed with a clear description (e.g., "Extract `userId` from Step 1 and use it in Step 2").
    *   The user can either click **"✅ Apply All Correlations"** for maximum speed or apply each one individually via a dedicated form that allows them to customize the JSON Path and variable name.
*   **Automatic Spec-Driven Assertions:**
    *   The application parses the expected success codes (e.g., `200`, `201`) for each endpoint directly from the Swagger file.
    *   This expected code is visually displayed next to each step in the scenario builder.
    *   A single **"🎯 Apply Auto-Assertions"** button adds a JMeter `ResponseAssertion` to every step, configured to validate against its specific, documented success code.

**Step 3: Map Data & Assemble Test Case**
*   This is a single, powerful action button (`Map Data & Assemble Scenario`). When clicked, it orchestrates the final data-driven assembly:
    *   **Intelligent Data Mapping:** It gathers all unique parameters from the scenario and sends them to the Groq LLM API with a highly-constrained prompt. This prompt explicitly instructs the AI to handle various naming conventions (`snake_case`, `PascalCase`) and forbids it from inventing columns that don't exist in the provided DB schema.
    *   **Code-Based Heuristics Fallback:** For any parameter the AI cannot map, or for common fields, a powerful, code-based heuristic function (`get_value_from_heuristics`) provides sensible default JMeter functions (e.g., `${__UUID()}` for IDs, `${__time()}` for dates). This is based on the comprehensive list we developed.
    *   **Final Assembly:** The tool builds the final test plan object, injecting the correct variables (`${csv_...}` for mapped data, `${__Random...}` for heuristic data, and `${extracted_...}` for correlated data) into every request's path, query parameters, and body.
*   **Verification Previews:** After assembly, the UI displays previews of the AI mapping results and the generated CSV data files, allowing the user to verify everything before the final step.

**Step 4: Generate JMeter Package**
*   The final step is a single `🚀 Generate Full Test Plan` button.
*   It takes the fully assembled and data-driven test plan from the previous step.
*   It generates a complete, downloadable `.zip` package containing:
    *   `test_plan.jmx`: The clean, well-structured, and fully functional JMeter test plan.
    *   `data/` (folder): Contains all necessary `.csv` files for data-driven testing.
    *   `design.json` / `design.yaml`: Human-readable representations of the final assembled test case for documentation and review.

