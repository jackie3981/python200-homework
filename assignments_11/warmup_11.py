# --- Prefect Orchestration ---
"""
Difference between @task and @flow in Prefect:

@task: Decorates a single, focused unit of work (e.g.loading data, calling an API, writing a file.)
       Tasks are the atomic building blocks that Prefect tracks its execution, state, and logs.

@flow: Decorates the orchestrator function that calls tasks in sequence and manages 
       the entire pipeline run as a whole. Flows is a top-level function that wires tasks together. entry point Prefect tracks as a run.

Would I decorate the Celsius-to-Fahrenheit helper with @task?

No, I would NOT decorate it with @task. Why not:

1. It's a pure, synchronous, in-memory calculation with no I/O operations
2. It has no failure modes that retries would solve.
3. Adding @task introduces unnecessary overhead (state tracking, serialization) without 
   providing any benefit
"""

# Prefect Question 2
# @task(retries=3, retry_delay_seconds=30)

# Prefect Question 3
"""
Where to look in the Prefect UI:

Navigate to the specific flow run (click on it from the main dashboard). Then:

1. PRIMARY LOCATION - The Task Run View:
   - Click on the failed 'transform' task in the task list visualization
   - This will open the task run details

2. SPECIFIC INFORMATION TO LOOK FOR:
   In the task run details, check:
   - Status: Should show 'Failed' (red indicator)
   - Error Message/Exception: The exact Python exception
   - Stack Trace: The full execution trace showing exactly where and why the task failed
   
   In the Logs tab:
   - All captured output from the transform task, including:
     * print() statements (if log_prints=True)
     * Any structured logs from get_run_logger()
     * The exact error message with context about what input caused the failure

Why load never ran:
Because transform failed, Prefect correctly short-circuited the flow. Load depends 
on transform's output, Prefect marks the flow as failed without proceeding to dependent tasks.
"""

# --- Production Patterns ---
# Production Question 1
"""
raise_for_status() lanza una excepción HTTPError si el código es 400 o 500.
Con if + print: la tarea sigue ejecutándose con datos malos, se marca como Completed,
y las tareas downstream corren con datos corruptos.

Con raise_for_status(): la tarea se marca como Failed, las tareas downstream NO corren,
y el pipeline falla de forma visible en la UI.

El principio del fichero: "visible failures are safer than silent corruption".

What raise_for_status() does:
raise_for_status() checks the HTTP response status code. If the code is 400 or higher, it raises a requests.exceptions.HTTPError 
exception. If the code is 200-399 (success), it does nothing and the pipeline continues.

Why it's better than manual if-check + print:

MANUAL CHECK (if status != 200: print("error")):
- Only prints a message but the task continues executing
- The task is marked as Completed (success) in Prefect
- Bad/empty data continues to downstream tasks
- The pipeline appears successful even though data is corrupted

WITH raise_for_status():
- Immediately raises an HTTPError exception
- Prefect catches the exception and marks the task as Failed
- Downstream tasks NEVER run
- The failure is explicit and visible in the UI

What happens when API returns a 500 error:

WITH MANUAL CHECK:
- extract task: prints "error" but completes successfully (green in UI)
- transform task: receives bad/empty data from extract
- load task: may run with corrupted data, writing bad results to storage
- RESULT: Silent data corruption, hard to detect, may go unnoticed for days

WITH raise_for_status():
- extract task: raises HTTPError, marked as Failed (red in UI)
- transform task: NEVER runs (skipped)
- load task: NEVER runs (skipped)
- RESULT: Visible failure, easy to debug, no bad data written

Key principle from the lesson:
"In data pipelines, visible failures are usually safer than silent corruption."
"""
# Production Question 2
"""
What overwrite=True protects you from in this scenario:

You run the pipeline → crashes halfway through transform → fix the bug → re-run.

WITH overwrite=True:
- Second run successfully completes
- The new upload_blob() call replaces the previous (possibly incomplete/corrupted) blob
- The file at final/{today}/weather_etl.json contains the correct, complete data
- The pipeline is IDEMPOTENT - running it twice produces the same correct result

WITHOUT overwrite=True (default overwrite=False):
- a second run would fail because the blob already exists
- I have to manually delete the blob or change the path
- The pipeline is NOT idempotent - can't safely re-run
"""

# Production Question 3
from prefect import task
from prefect.logging import get_run_logger

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")