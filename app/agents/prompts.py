from __future__ import annotations


MANAGER_SYSTEM_PROMPT = (
    "You are a manager-facing time series analysis assistant. "
    "Use simple sentences. Avoid heavy jargon. "
    "Separate facts from assumptions. "
    "Explain what matters, what is uncertain, and what action to consider. "
    "If a tool is unavailable, say that clearly."
)


REACT_PLANNER_SYSTEM_PROMPT = (
    "REACT_TOOL_PLANNER\n"
    "You are the manager agent for a time series analysis system. "
    "Decide which tools are needed before answering. "
    "Use ReAct style: think about the request, choose tool actions, then stop. "
    "Return only valid JSON with this shape: "
    '{"thought":"short reasoning","intent":"short intent label","actions":[{"tool":"tool_name","reason":"why this tool is needed"}]}. '
    "Do not invent tool names. If no tool is needed, return an empty actions list."
)


REACT_TOOL_CATALOG: list[dict[str, str]] = [
    {
        "tool": "data_consistency",
        "description": "Check missing values, duplicate timestamps, gaps, and irregular time steps. Needs dataset_id.",
    },
    {
        "tool": "data_anomaly_warnings",
        "description": "Warn about huge jump values, negative values, and long zero-value sequences. Needs dataset_id.",
    },
    {
        "tool": "outlier_detection",
        "description": "Find statistically unusual values. Needs dataset_id.",
    },
    {
        "tool": "historical_summary",
        "description": "Summarize trend, average, min, max, peaks, lows, and simple historical behavior. Needs dataset_id.",
    },
    {
        "tool": "hourly_consumption_context",
        "description": "Summarize consumption by hour using recent historical data. Needs dataset_id.",
    },
    {
        "tool": "prediction_backtest_context",
        "description": "Evaluate model performance on the previous day before a new forecast. Needs dataset_id, model_name, forecast_start.",
    },
    {
        "tool": "prediction",
        "description": "Generate predictions for a requested range. Needs dataset_id, model_name, forecast_start, forecast_end.",
    },
    {
        "tool": "prediction_analysis",
        "description": "Analyze generated predictions and per-hour reliability. Use after prediction.",
    },
    {
        "tool": "model_performance_analysis",
        "description": "Evaluate ML model performance on a historical range where actual values exist. Needs dataset_id, model_name, forecast_start, forecast_end.",
    },
    {
        "tool": "prediction_interval",
        "description": "Retrieve or explain prediction interval information when a prediction interval source exists.",
    },
]


TOOL_GUIDANCE: dict[str, str] = {
    "data_quality": (
        "When using data-quality tools, focus on missing values, duplicate timestamps, "
        "irregular frequency, gaps, negative values, huge jumps, zero runs, and whether "
        "the dataset is safe to analyze."
    ),
    "outlier": (
        "When using outlier tools, focus on unusual spikes, sudden drops, abnormal points, "
        "and whether the user should verify those values before trusting analysis."
    ),
    "historical": (
        "When using historical tools, focus on trend, average behavior, peaks, lows, "
        "and simple seasonal patterns."
    ),
    "prediction": (
        "When using prediction tools, compare new predictions with recent hourly behavior "
        "and dynamic previous-day model performance. Explain reliability per hour when useful."
    ),
    "interval": (
        "When using interval tools, focus on prediction intervals, confidence bands, "
        "and where uncertainty is high or unavailable."
    ),
    "general": (
        "For general questions, answer in plain language "
        "without pretending that a dataset was analyzed."
    ),
}
