import inspect
import logging
import os
from typing import Any, Dict, Optional, Sequence, Union

from omegaconf import DictConfig

from megamem.core.general_api import (
    GeneralAPIClient,
    GeneralContentFilterError,
)
from megamem.utils.token_usage import TokenUsageCallback

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Optional[DictConfig], section: str, key: str, default: str = "") -> str:
    if cfg is None:
        return default
    block = getattr(cfg, section, None)
    if block is None:
        return default
    return getattr(block, key, default)


def get_general_chat_completion_client(
    cfg: Optional[DictConfig] = None,
) -> GeneralAPIClient:
    """Build a client for the general model gateway."""
    base_url = os.getenv("LLM_API_BASE") or _cfg_get(cfg, "llm", "api_base")
    api_key = os.getenv("LLM_API_KEY") or _cfg_get(cfg, "llm", "api_key")
    if not base_url or not api_key:
        raise RuntimeError(
            "General LLM API base/key not configured. "
            "Set LLM_API_BASE and LLM_API_KEY before running."
        )
    return GeneralAPIClient(base_url=base_url, api_key=api_key)


class ChatCompletionModel:
    """Unified chat-completion frontend.

    The default backend is a general chat API. Set ``cfg.llm.backend`` to
    ``huggingface`` to run a local causal LM instead.
    """

    def __init__(self, cfg: DictConfig, token_usage_callback=None):
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback

        model_name = cfg.llm.model
        self.model_type = self._determine_model_type(model_name)

        if self.model_type == "huggingface":
            self._load_hf_model(model_name)
            self.client = None
        else:
            self.client = get_general_chat_completion_client(cfg)
            self.hf_model = None
            self.hf_tokenizer = None

    def _determine_model_type(self, model_name: str) -> str:
        """Classify the configured backend."""
        backend = str(self.cfg.llm.get("backend", "api")).lower()
        if backend in {"hf", "huggingface", "local"}:
            return "huggingface"
        return "api"

    def _load_hf_model(self, model_name: str):
        """Load a Hugging Face causal LM and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if model_name.startswith("hf:"):
            model_name = model_name[3:]

        print(f"Loading Hugging Face model: {model_name}")

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        device = next(self.hf_model.parameters()).device
        print(f"Model loaded successfully on device: {device}")

    def invoke(
        self,
        input: Union[str, Sequence[str]],
        prompt_args: Optional[Dict] = None,
        response_format: Any = None,
        source: str = "Unknown",
        **kwargs: Any,
    ) -> str:
        """Run a single completion against the configured backend."""
        if isinstance(input, str):
            rendered = input.format(**prompt_args) if prompt_args else input
            messages = [{"role": "user", "content": rendered}]
        elif isinstance(input, list):
            messages = input
        else:
            raise ValueError(
                "Input must be a string or a sequence of LLMMessage objects."
            )

        if self.model_type == "huggingface":
            return self._invoke_hf(messages, source, **kwargs)
        return self._invoke_api(messages, response_format, source, **kwargs)

    def _invoke_hf(self, messages: list, source: str = "Unknown", **kwargs) -> str:
        """Run a Hugging Face completion with retries on transient errors."""
        import time
        import torch

        max_retries = 3
        retry_delay = 1.0
        max_delay = 10.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                prompt_text = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                tokenized = self.hf_tokenizer(
                    [prompt_text], return_tensors="pt"
                ).to(self.hf_model.device)

                seed = self.cfg.llm.get("seed", 42)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                max_new_tokens = kwargs.pop(
                    "max_tokens", kwargs.pop("max_new_tokens", 512)
                )
                temperature = kwargs.pop("temperature", 0.7)

                with torch.no_grad():
                    generated_ids = self.hf_model.generate(
                        **tokenized,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                    )

                generated_ids = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(tokenized.input_ids, generated_ids)
                ]

                result = self.hf_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                if self.token_usage_callback:
                    prompt_tokens = len(tokenized.input_ids[0])
                    completion_tokens = len(generated_ids[0])
                    self.token_usage_callback.update(
                        prompt_tokens,
                        completion_tokens,
                        model=model_name_for_usage(self.cfg.llm.model),
                        source=source,
                    )

                return result

            except Exception as exc:
                last_exception = exc
                err_text = str(exc)
                err_lower = err_text.lower()
                retryable = (
                    "out of memory" in err_lower
                    or "cuda" in err_lower
                    or "timeout" in err_lower
                )

                if attempt < max_retries - 1 and retryable:
                    backoff = min(retry_delay * (2 ** attempt), max_delay)
                    print(
                        f"HF model error (attempt {attempt + 1}/{max_retries}): "
                        f"{err_text[:100]}"
                    )
                    print(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)

                    if "out of memory" in err_lower and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    if attempt == max_retries - 1:
                        print(
                            f"Error in HF model invocation after {max_retries} "
                            f"attempts: {exc}"
                        )
                    else:
                        print(f"Non-retryable error in HF model invocation: {exc}")
                    raise exc

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("HF model invocation failed without an exception.")

    def _invoke_api(
        self, messages: list, response_format: Any, source: str, **kwargs
    ) -> str:
        """Run a general API completion with retries on transient errors."""
        import time

        max_retries = 3
        retry_delay = 1.0
        max_delay = 10.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                if response_format:
                    response = self.client.chat.completions.parse(
                        messages=messages,
                        model=self.cfg.llm.model,
                        response_format=response_format,
                        seed=self.cfg.llm.get("seed", 42),
                        **kwargs,
                    )
                else:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=self.cfg.llm.model,
                        seed=self.cfg.llm.get("seed", 42),
                        **kwargs,
                    )
                break

            except GeneralContentFilterError as exc:
                logger.warning(
                    "Content filter blocked request. Returning empty response. "
                    f"Messages length: {len(messages)}, Response format: "
                    f"{response_format.__name__ if response_format else 'None'}"
                )

                if response_format:
                    try:
                        return response_format(entries=[])
                    except Exception:
                        try:
                            return response_format()
                        except Exception:
                            logger.error(
                                f"Could not create empty response for format "
                                f"{response_format}"
                            )
                            raise exc
                return ""

            except Exception as exc:
                last_exception = exc
                err_text = str(exc)
                err_lower = err_text.lower()
                retryable = (
                    "rate" in err_lower
                    or "429" in err_text
                    or "timeout" in err_lower
                    or "503" in err_text
                    or "502" in err_text
                    or "500" in err_text
                    or "connection" in err_lower
                )

                if attempt < max_retries - 1 and retryable:
                    backoff = min(retry_delay * (2 ** attempt), max_delay)
                    print(
                        f"LLM API error (attempt {attempt + 1}/{max_retries}): "
                        f"{err_text[:100]}"
                    )
                    print(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    if attempt == max_retries - 1:
                        print(
                            f"Error in LLM API invocation after "
                            f"{max_retries} attempts: {exc}"
                        )
                    else:
                        print(f"Non-retryable error in LLM API invocation: {exc}")
                    raise exc

        if "response" not in locals():
            raise last_exception

        if response_format:
            result = response.choices[0].message.parsed
        else:
            result = response.choices[0].message.content

        if self.token_usage_callback:
            usage = response.usage
            assert isinstance(self.token_usage_callback, TokenUsageCallback)

            if source == "Unknown":
                try:
                    caller = inspect.stack()[1].frame.f_locals.get("self", None)
                    source = caller.__class__.__name__ if caller else "Unknown"
                except Exception:
                    source = "Unknown"

            self.token_usage_callback.update(
                getattr(usage, "prompt_tokens", 0),
                getattr(usage, "completion_tokens", 0),
                model=model_name_for_usage(self.cfg.llm.model),
                source=source,
            )

        return result


def model_name_for_usage(model_name: str) -> str:
    """Normalize a model identifier for token-usage logs."""
    return str(model_name).replace("hf:", "")
