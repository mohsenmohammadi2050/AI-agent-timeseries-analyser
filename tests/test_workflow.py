from __future__ import annotations

from app.agents.workflow import _fallback_answer_from_tools


def test_fallback_answer_uses_tool_results_for_prediction_analysis() -> None:
    answer = _fallback_answer_from_tools(
        [
            {
                "name": "prediction",
                "status": "ok",
                "summary": "Generated 24 predicted values with linear_regression_simple.",
                "data": {},
            },
            {
                "name": "prediction_analysis",
                "status": "ok",
                "summary": "The prediction average is 120.00.",
                "data": {
                    "low_reliability_count": 1,
                    "medium_reliability_count": 1,
                    "per_hour_reliability": [
                        {
                            "timestamp": "2026-01-03T18:00:00",
                            "reliability": "low",
                            "reasons": ["prediction is above the recent observed range"],
                        },
                        {
                            "timestamp": "2026-01-03T19:00:00",
                            "reliability": "medium",
                            "reasons": ["recent backtest error for this hour is moderate"],
                        },
                    ],
                },
            },
        ]
    )

    assert "Generated 24 predicted values" in answer
    assert "Reliability check" in answer
    assert "2026-01-03T18:00:00" in answer
    assert "shorter question" not in answer
