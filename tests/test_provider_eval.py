from types import SimpleNamespace

from emoji_bench.model_registry import get_model_config
from emoji_bench.provider_eval import (
    _openai_missing_output_error,
    _request_openai_prediction,
)


class _FakeParsedModel:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return dict(self._payload)


class _FakeResponsesAPI:
    def __init__(self, queued_responses: list[object]):
        self._queued_responses = list(queued_responses)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._queued_responses.pop(0)


class _FakeOpenAIClient:
    def __init__(self, queued_responses: list[object]):
        self.responses = _FakeResponsesAPI(queued_responses)


def test_request_openai_prediction_extracts_nested_parsed_content():
    response = SimpleNamespace(
        id="resp_nested",
        output_parsed=None,
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text='{"has_error": true, "error_step": 2}',
                        parsed=_FakeParsedModel({"has_error": True, "error_step": 2}),
                    )
                ],
            )
        ],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    client = _FakeOpenAIClient([response])

    provider_response = _request_openai_prediction(
        client=client,
        model_config=get_model_config("gpt-5.4-mini"),
        prompt="example",
        max_output_tokens=50,
        reasoning_effort=None,
    )

    assert provider_response.prediction_payload == {"has_error": True, "error_step": 2}
    assert provider_response.response_id == "resp_nested"
    assert provider_response.raw_output_text == '{"has_error": true, "error_step": 2}'
    assert len(client.responses.calls) == 1


def test_request_openai_prediction_retries_once_when_incomplete_for_max_output_tokens():
    first_response = SimpleNamespace(
        id="resp_incomplete",
        output_parsed=None,
        output_text="",
        output=[SimpleNamespace(type="reasoning")],
        usage=None,
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    second_response = SimpleNamespace(
        id="resp_complete",
        output_parsed=_FakeParsedModel({"has_error": False, "error_step": None}),
        output_text='{"has_error": false, "error_step": null}',
        output=[],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    client = _FakeOpenAIClient([first_response, second_response])

    provider_response = _request_openai_prediction(
        client=client,
        model_config=get_model_config("gpt-5.4-mini"),
        prompt="example",
        max_output_tokens=50,
        reasoning_effort=None,
    )

    assert provider_response.prediction_payload == {"has_error": False, "error_step": None}
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["max_output_tokens"] == 50
    assert client.responses.calls[1]["max_output_tokens"] == 200


def test_openai_missing_output_error_includes_status_and_retry_hint():
    message = _openai_missing_output_error(
        response=SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="message")],
        )
    )

    assert "status=incomplete" in message
    assert "incomplete_reason=max_output_tokens" in message
    assert "Retry with a higher --max-output-tokens value." in message
