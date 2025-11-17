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
from openai import OpenAI


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
    ) -> None:
        self.port = port
        self.metrics = MetricsCollector()
        self._openai_api_key = openai_api_key or ""
        self._request_timeout = float(os.getenv("LLM_PROXY_TIMEOUT", "60"))
        upstream_base = (
            upstream_base_url or os.getenv("UPSTREAM_OPENAI_BASE_URL") or "https://api.openai.com"
        ).rstrip("/")
        self._upstream_base_url = upstream_base
        self._upstream_api_url = f"{self._upstream_base_url}/v1"
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=self._upstream_api_url,
        )
        self._beta_chat_url = f"{self._upstream_api_url}/beta/chat/completions"
        self._server_thread: threading.Thread | None = None
        self._should_stop = False

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        self.app = FastAPI(title="AgentBench LLM Proxy", lifespan=lifespan)
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy chat completions to OpenAI and track metrics."""
            body = await request.json()
            model = body.get("model", "gpt-4o-mini")

            case_id = x_case_id or "_current"
            self._log_request("/v1/chat/completions", body, case_id)

            start_time = time.time()
            try:
                response = self.openai_client.chat.completions.create(**body)
                latency_ms = (time.time() - start_time) * 1000

                self._record_usage(case_id, model, usage=response.usage, latency_ms=latency_ms)

                return JSONResponse(content=response.model_dump())
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )

        @self.app.post("/v1/beta/chat/completions")
        async def beta_chat_completions(
            request: Request,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy beta chat completions (structured outputs) to OpenAI."""

            body = await request.json()
            model = body.get("model", "gpt-4o-mini")
            case_id = x_case_id or "_current"
            self._log_request("/v1/beta/chat/completions", body, case_id)

            if not self._openai_api_key:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Proxy missing OPENAI_API_KEY for upstream requests."},
                )

            headers = {
                "Authorization": f"Bearer {self._openai_api_key}",
                "Content-Type": "application/json",
            }
            for header_name in ("OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project"):
                header_value = request.headers.get(header_name)
                if header_value:
                    headers[header_name] = header_value

            start_time = time.time()
            try:
                response = httpx.post(
                    self._beta_chat_url,
                    json=body,
                    headers=headers,
                    timeout=self._request_timeout,
                )
            except httpx.HTTPError as exc:
                logger.error(f"Proxy beta request failed: {exc}")
                return JSONResponse(status_code=500, content={"error": str(exc)})

            latency_ms = (time.time() - start_time) * 1000

            try:
                response_data = response.json()
            except ValueError:
                response_data = {"error": response.text}

            if response.status_code >= 400:
                return JSONResponse(status_code=response.status_code, content=response_data)

            self._record_usage(
                case_id,
                model,
                usage=response_data.get("usage"),
                latency_ms=latency_ms,
            )

            return JSONResponse(content=response_data)

        @self.app.post("/v1/embeddings")
        async def embeddings(
            request: Request,
            x_case_id: str | None = Header(None, alias="X-Case-ID"),
        ) -> JSONResponse:
            """Proxy embeddings to OpenAI."""
            body = await request.json()
            case_id = x_case_id or "_current"
            self._log_request("/v1/embeddings", body, case_id)

            try:
                response = self.openai_client.embeddings.with_raw_response.create(**body)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)},
                )
            try:
                response_data = response.http_response.json()
            except ValueError:
                response_data = {"error": response.http_response.text}

            return JSONResponse(status_code=response.status_code, content=response_data)

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

        raise RuntimeError("Proxy server failed to start")

    async def _run_server(self) -> None:
        """Run the uvicorn server."""
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        await server.serve()

    def stop(self) -> None:
        """Stop the proxy server."""
        self._should_stop = True

    def _record_usage(self, case_id: str, model: str, usage: Any, latency_ms: float) -> None:
        """Record token usage for metrics from OpenAI responses."""
        if usage is None:
            return

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
