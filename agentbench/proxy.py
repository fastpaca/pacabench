"""FastAPI-based OpenAI proxy for intercepting and tracking LLM calls."""

import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from genai_prices import Usage, calc_price
from loguru import logger
from openai import APIStatusError, AsyncOpenAI


class MetricsCollector:
    """Thread-safe metrics collector for tracking LLM usage per case."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "llm_call_count": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_cache_read_tokens": 0,
                "llm_cache_write_tokens": 0,
                "llm_total_cost_usd": 0.0,
                "llm_latency_ms": [],
                "llm_calls": [],
            }
        )

    def record_call(
        self,
        case_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        latency_ms: float,
        ttft_ms: float | None = None,
        status_code: int = 200,
        error: str | None = None,
        provider: str = "openai",
    ) -> None:
        """Record a single LLM call's metrics."""
        with self._lock:
            metrics = self._metrics[case_id]
            metrics["llm_call_count"] += 1
            metrics["llm_input_tokens"] += input_tokens
            metrics["llm_output_tokens"] += output_tokens
            metrics["llm_cache_read_tokens"] += cache_read_tokens
            metrics["llm_cache_write_tokens"] += cache_write_tokens
            metrics["llm_latency_ms"].append(latency_ms)

            cost = self._calculate_cost(
                model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            )
            metrics["llm_total_cost_usd"] += cost

            metrics["llm_calls"].append(
                {
                    "timestamp": time.time(),
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                    "status_code": status_code,
                    "error": error,
                    "provider": provider,
                    "cost_usd": cost,
                }
            )

    def get_metrics(self, case_id: str) -> dict[str, Any]:
        """Get aggregated metrics for a case."""
        with self._lock:
            return dict(self._metrics[case_id])

    def clear_metrics(self, case_id: str) -> None:
        """Clear metrics for a case."""
        with self._lock:
            if case_id in self._metrics:
                del self._metrics[case_id]

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> float:
        """Calculate cost using genai-prices."""
        try:
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            price_calc = calc_price(usage, model)
            return float(price_calc.total_price)
        except Exception:
            return 0.0


class ProxyServer:
    """OpenAI-compatible proxy server for tracking LLM calls."""

    def __init__(
        self,
        port: int = 8000,
        openai_api_key: str | None = None,
        upstream_base_url: str | None = None,
        provider: str = "openai",
    ) -> None:
        self.port = port
        self.metrics = MetricsCollector()
        self._openai_api_key = openai_api_key or ""
        self._request_timeout = float(os.getenv("LLM_PROXY_TIMEOUT", "60"))
        self._provider = provider
        upstream_base = (
            upstream_base_url
            or os.getenv("UPSTREAM_OPENAI_BASE_URL")
            or self._default_base_for_provider(provider)
        ).rstrip("/")
        self._upstream_base_url = upstream_base
        self._upstream_api_url = f"{self._upstream_base_url}/v1"
        self.openai_client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY") or "dummy",
            base_url=self._upstream_api_url,
        )
        self._beta_chat_url = f"{self._upstream_api_url}/beta/chat/completions"
        self._server_thread: threading.Thread | None = None
        self._should_stop = False
        self._server = None
        self._active_case_id = "_current"

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        self.app = FastAPI(title="AgentBench LLM Proxy", lifespan=lifespan)
        self._setup_routes()

    def _default_base_for_provider(self, provider: str) -> str:
        if provider == "openai":
            return "https://api.openai.com"
        return "https://api.openai.com"

    def _setup_routes(self) -> None:
        @self.app.post("/v1/chat/completions")
        @self.app.post("/v1/case/{case_id}/chat/completions")
        async def chat_completions(
            request: Request,
            case_id: str | None = None,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy chat completions to OpenAI and track metrics."""
            body = await request.json()
            model = body.get("model", "gpt-4o-mini")

            case_id = x_case_id or case_id or self._active_case_id
            self._log_request("/v1/chat/completions", body, case_id)

            start_time = time.time()
            try:
                # If key is missing, try to use env var or what was passed to init
                # OpenAI client will look for env var OPENAI_API_KEY
                response = await self.openai_client.chat.completions.create(**body)
                latency_ms = (time.time() - start_time) * 1000

                self._record_usage(
                    case_id,
                    model,
                    usage=response.usage,
                    latency_ms=latency_ms,
                    status_code=200,
                )

                return JSONResponse(content=response.model_dump())
            except APIStatusError as e:
                latency_ms = (time.time() - start_time) * 1000
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=latency_ms,
                    status_code=e.status_code,
                    error=e.message,
                )
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": {"message": e.message, "type": "api_error"}},
                )
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=latency_ms,
                    status_code=500,
                    error=str(e),
                )
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )

        @self.app.post("/v1/beta/chat/completions")
        @self.app.post("/v1/case/{case_id}/beta/chat/completions")
        async def beta_chat_completions(
            request: Request,
            case_id: str | None = None,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy beta chat completions (structured outputs) to OpenAI."""

            body = await request.json()
            model = body.get("model", "gpt-4o-mini")
            case_id = x_case_id or case_id or self._active_case_id
            self._log_request("/v1/beta/chat/completions", body, case_id)

            # Manual fetch using httpx because openai python client might not fully support beta paths via same client easily
            # Actually new client does, but let's keep the logic if it works.

            api_key = self._openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Proxy missing OPENAI_API_KEY for upstream requests."},
                )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            for header_name in ("OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project"):
                header_value = request.headers.get(header_name)
                if header_value:
                    headers[header_name] = header_value

            start_time = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self._beta_chat_url,
                        json=body,
                        headers=headers,
                        timeout=self._request_timeout,
                    )
            except httpx.HTTPError as exc:
                logger.error(f"Proxy beta request failed: {exc}")
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=(time.time() - start_time) * 1000,
                    status_code=500,
                    error=str(exc),
                )
                return JSONResponse(status_code=500, content={"error": str(exc)})

            latency_ms = (time.time() - start_time) * 1000

            try:
                response_data = response.json()
            except ValueError:
                response_data = {"error": response.text}

            if response.status_code >= 400:
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error=str(response_data.get("error", response.text)),
                )
                return JSONResponse(status_code=response.status_code, content=response_data)

            self._record_usage(
                case_id,
                model,
                usage=response_data.get("usage"),
                latency_ms=latency_ms,
                status_code=response.status_code,
            )

            return JSONResponse(content=response_data)

        @self.app.post("/v1/embeddings")
        @self.app.post("/v1/case/{case_id}/embeddings")
        async def embeddings(
            request: Request,
            case_id: str | None = None,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy embeddings to OpenAI."""
            body = await request.json()
            case_id = x_case_id or case_id or self._active_case_id
            self._log_request("/v1/embeddings", body, case_id)
            model = body.get("model", "text-embedding-ada-002")

            start_time = time.time()
            try:
                response = await self.openai_client.embeddings.with_raw_response.create(**body)
                latency_ms = (time.time() - start_time) * 1000
                try:
                    response_data = response.http_response.json()
                    usage = response_data.get("usage")
                except ValueError:
                    response_data = {"error": response.http_response.text}
                    usage = None

                self._record_usage(
                    case_id,
                    model,
                    usage=usage,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
                return JSONResponse(status_code=response.status_code, content=response_data)

            except APIStatusError as e:
                latency_ms = (time.time() - start_time) * 1000
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=latency_ms,
                    status_code=e.status_code,
                    error=e.message,
                )
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": {"message": e.message, "type": "api_error"}},
                )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self._record_usage(
                    case_id,
                    model,
                    usage=None,
                    latency_ms=latency_ms,
                    status_code=500,
                    error=str(e),
                )
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

    def start(self) -> None:
        """Start the proxy server in a background thread."""
        import asyncio

        def run_server():
            asyncio.run(self._run_server())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready
        for _ in range(30):
            try:
                response = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.1)

        raise RuntimeError(f"Proxy server failed to start on port {self.port}")

    async def _run_server(self) -> None:
        """Run the uvicorn server."""
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    def stop(self) -> None:
        """Stop the proxy server."""
        self._should_stop = True
        if self._server:
            self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=2.0)

    def set_active_case(self, case_id: str) -> None:
        """Set the case id the proxy should attribute metrics to when no header is provided."""
        self._active_case_id = case_id

    def _record_usage(
        self,
        case_id: str,
        model: str,
        usage: Any,
        latency_ms: float,
        status_code: int,
        error: str | None = None,
    ) -> None:
        """Record token usage for metrics from OpenAI responses."""

        def _get(value: Any, attr: str) -> Any:
            if value is None:
                return None
            if isinstance(value, dict):
                return value.get(attr)
            return getattr(value, attr, None)

        input_tokens = _get(usage, "prompt_tokens") or 0
        output_tokens = _get(usage, "completion_tokens") or 0
        cache_details = _get(usage, "prompt_tokens_details")
        cache_read_tokens = _get(cache_details, "cached_tokens") or 0

        self.metrics.record_call(
            case_id=case_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            status_code=status_code,
            error=error,
            provider=self._provider,
        )

    def _log_request(self, route: str, body: dict[str, Any], case_id: str) -> None:
        """Emit minimal debug logging for proxied OpenAI calls."""
        try:
            model = body.get("model", "unknown")
            metadata = body.get("metadata")
            log_metadata = metadata if isinstance(metadata, list) else []
            logger.debug(
                "[proxy] {route} case={case_id} model={model} metadata={metadata}",
                route=route,
                case_id=case_id,
                model=model,
                metadata=log_metadata,
            )
        except Exception as exc:
            logger.debug(f"Failed to log proxy request: {exc}")
