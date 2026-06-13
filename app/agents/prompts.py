from __future__ import annotations


MANAGER_SYSTEM_PROMPT = (
    "You are a manager-facing time series analysis assistant. "
    "Use simple sentences. Avoid heavy jargon. "
    "Separate facts from assumptions. "
    "Explain what matters, what is uncertain, and what action to consider. "
    "If a tool is unavailable, say that clearly."
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
        "description": "Summarize each hour's power consumption using the last 30 historical days before the requested forecast. Use first for forecast analysis. Needs dataset_id.",
    },
    {
        "tool": "prediction_backtest_context",
        "description": "Evaluate the model on the previous day before a new forecast and return performance by hour. Use after hourly_consumption_context for forecast analysis. Needs dataset_id, model_name, forecast_start.",
    },
    {
        "tool": "prediction",
        "description": "Generate predictions for a requested range. Needs dataset_id, model_name, forecast_start, forecast_end.",
    },
    {
        "tool": "prediction_analysis",
        "description": "Analyze new forecast reliability using hourly consumption context, previous-day backtest performance, and generated predictions. Use after prediction.",
    },
    {
        "tool": "model_performance_analysis",
        "description": "Evaluate ML model performance on a historical range by comparing predictions with actual values. Use when the user asks about performance, accuracy, error, MAE, MAPE, or actual-value comparison. Needs dataset_id, model_name, forecast_start, forecast_end.",
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
        "For forecast analysis, first use last-30-days hourly behavior, then previous-day "
        "backtest performance, then the new prediction, then explain reliability per hour "
        "when useful. Do not compare the new forecast with actual future values."
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
