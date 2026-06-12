# Agent Planning Evals

These are manual developer checks for the manager agent's ReAct planning behavior.

They answer this question:

```text
Given a user request, does the manager choose the expected intent and tools?
```

They are not used by the web UI and they do not run automatically.

Run them with:

```powershell
.\env\Scripts\python.exe agent_planning_evals\run_evals.py
```

The dynamic prediction-analysis evaluator in the UI is separate. That evaluator scores one real manager answer against held-out actual values.
