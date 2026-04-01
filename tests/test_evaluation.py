from emoji_bench.evaluation import EvalPrediction, normalize_prediction, score_prediction, summarize_scores


def test_normalize_prediction_accepts_boolean_and_null_step():
    prediction = normalize_prediction({"has_error": False, "error_step": 3})

    assert prediction == EvalPrediction(has_error=False, error_step=None)


def test_normalize_prediction_accepts_string_values():
    prediction = normalize_prediction({"has_error": "yes", "error_step": "4"})

    assert prediction == EvalPrediction(has_error=True, error_step=4)


def test_score_prediction_marks_joint_correct_when_both_fields_match():
    record = {
        "example_id": "ex-1",
        "difficulty": "medium",
        "error_type": "E-CASC",
        "has_error": True,
        "expected_error_step": 3,
    }

    scored = score_prediction(record, EvalPrediction(has_error=True, error_step=3))

    assert scored.has_error_correct is True
    assert scored.error_step_correct is True
    assert scored.joint_correct is True


def test_score_prediction_marks_clean_example_wrong_when_step_is_predicted():
    record = {
        "example_id": "ex-2",
        "difficulty": "easy",
        "error_type": None,
        "has_error": False,
        "expected_error_step": None,
    }

    scored = score_prediction(record, EvalPrediction(has_error=True, error_step=1))

    assert scored.has_error_correct is False
    assert scored.error_step_correct is False
    assert scored.joint_correct is False


def test_summarize_scores_reports_overall_and_per_difficulty_metrics():
    scored_predictions = [
        score_prediction(
            {
                "example_id": "ex-1",
                "difficulty": "easy",
                "error_type": None,
                "has_error": False,
                "expected_error_step": None,
            },
            EvalPrediction(has_error=False, error_step=None),
        ),
        score_prediction(
            {
                "example_id": "ex-2",
                "difficulty": "hard",
                "error_type": "E-CASC",
                "has_error": True,
                "expected_error_step": 2,
            },
            EvalPrediction(has_error=True, error_step=4),
        ),
    ]

    summary = summarize_scores(scored_predictions)

    assert summary["total_examples"] == 2
    assert summary["has_error_accuracy"] == 1.0
    assert summary["error_step_accuracy"] == 0.5
    assert summary["joint_accuracy"] == 0.5
    assert summary["by_difficulty"]["easy"]["joint_accuracy"] == 1.0
    assert summary["by_difficulty"]["hard"]["joint_accuracy"] == 0.0
