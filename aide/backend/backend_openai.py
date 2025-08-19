"""Backend for OpenAI API."""

import json
import logging
import time
import os
import re

from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_BASE_URL = "https://api.openai.com/v1"

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

_client: openai.OpenAI = None  # type: ignore
_custom_client: openai.OpenAI = None  # type: ignore

@once
def _setup_openai_client():
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL, max_retries=0)

@once
def _setup_custom_client():
    global _custom_client
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        _custom_client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    model_name = filtered_kwargs.get("model", "")
    is_openai_model = re.match(r"^(gpt-|o\d-|codex-mini-latest$)", model_name)
    use_chat_api = os.getenv("OPENAI_BASE_URL") is not None

    if use_chat_api:
        _setup_custom_client()
        messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
    else:
        messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
        for i in range(len(messages)):
            messages[i]["content"] = [
                {"type": "input_text", "text": messages[i]["content"]}
            ]
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_responses_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_responses_tool_choice_dict


    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
