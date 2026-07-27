"""REST API tool: makes HTTP requests via async httpx with SSRF protection enforced on every hop of a redirect chain.

SSRF protection only checking the *initial* URL is a common, real vulnerability:
an external, allowlisted-looking URL can 30x-redirect to an internal address
(e.g. the cloud metadata endpoint). This module re-validates every redirect
target before following it, not just the URL the caller supplied.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, field_validator

from src.base import BaseTool
from src.config import ALLOWED_DOMAINS, BLOCKED_DOMAINS, MAX_RESPONSE_SIZE_MB
from src.errors import ToolExecutionError, ToolSecurityError, ToolTimeoutError

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5


class RestApiInput(BaseModel):
    url: str = Field(..., min_length=1)
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    timeout_seconds: int = Field(default=30, ge=1, le=60)

    @field_validator("url")
    @classmethod
    def _validate_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"URL must use http or https, got: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("URL must include a host")
        return v


class RestApiOutput(BaseModel):
    status_code: int
    headers: Dict[str, str]
    body: Union[Dict[str, Any], List[Any], str]
    response_time_ms: int
    content_type: str


def _check_ssrf(url: str) -> None:
    """Raise ToolSecurityError if url's host is blocked by name, not allowlisted, or resolves to a private/reserved IP."""
    hostname = urlparse(url).hostname
    if not hostname:
        raise ToolSecurityError(f"URL has no resolvable host: {url}")
    hostname_lower = hostname.lower()

    if hostname_lower in BLOCKED_DOMAINS:
        raise ToolSecurityError(f"Host is blocked: {hostname}")

    if ALLOWED_DOMAINS and not any(
        hostname_lower == d or hostname_lower.endswith(f".{d}") for d in ALLOWED_DOMAINS
    ):
        raise ToolSecurityError(f"Host is not in the configured allowlist: {hostname}")

    # Resolve DNS and check every returned address -- catches both IP-literal
    # SSRF attempts (http://127.0.0.1/) and DNS-based ones (a public-looking
    # hostname that resolves to an internal or link-local/metadata address).
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ToolExecutionError(f"Could not resolve host {hostname!r}: {e}") from e

    for _family, _type, _proto, _canon, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ToolSecurityError(f"Host {hostname!r} resolves to a blocked internal address: {sockaddr[0]}")


class RestApiTool(BaseTool):
    """Makes an HTTP request and returns structured status/headers/body, with SSRF protection on the URL and every redirect hop."""

    name = "rest_api"
    description = "Make an HTTP request to a REST API and return status code, headers, and parsed body."
    input_schema = RestApiInput
    output_schema = RestApiOutput

    async def execute(self, inputs: RestApiInput) -> RestApiOutput:
        """Send the request, following up to 5 redirects with SSRF re-validation on each hop, and stream the body with a hard size cap.

        Example:
            result = await RestApiTool().run(url="https://api.github.com/repos/anthropics/anthropic-sdk-python")
        """
        _check_ssrf(inputs.url)
        max_bytes = MAX_RESPONSE_SIZE_MB * 1024 * 1024
        current_url = inputs.url

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=inputs.timeout_seconds) as client:
                redirects_followed = 0
                while True:
                    async with client.stream(
                        inputs.method,
                        current_url,
                        headers=inputs.headers,
                        params=inputs.params,
                        json=inputs.body,
                        follow_redirects=False,
                    ) as response:
                        is_redirect = 300 <= response.status_code < 400 and "location" in response.headers
                        status_code = response.status_code
                        resp_headers = dict(response.headers)

                        if is_redirect:
                            location = response.headers["location"]
                        else:
                            body_bytes = bytearray()
                            async for chunk in response.aiter_bytes():
                                body_bytes.extend(chunk)
                                if len(body_bytes) > max_bytes:
                                    raise ToolExecutionError(
                                        f"Response exceeded the {MAX_RESPONSE_SIZE_MB}MB limit"
                                    )
                            final_body_bytes = bytes(body_bytes)

                    if not is_redirect:
                        break
                    if redirects_followed >= _MAX_REDIRECTS:
                        raise ToolExecutionError(f"Exceeded max redirects ({_MAX_REDIRECTS}) for {inputs.url}")

                    next_url = str(httpx.URL(current_url).join(location))
                    _check_ssrf(next_url)
                    current_url = next_url
                    redirects_followed += 1
        except httpx.TimeoutException as e:
            raise ToolTimeoutError(f"Request to {inputs.url} timed out after {inputs.timeout_seconds}s: {e}") from e
        except httpx.ConnectError as e:
            raise ToolExecutionError(f"Connection failed for {inputs.url}: {e}") from e
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"Request to {inputs.url} failed: {e}") from e

        response_time_ms = int((time.perf_counter() - start) * 1000)
        content_type = resp_headers.get("content-type", "")

        body: Union[Dict[str, Any], List[Any], str]
        if "application/json" in content_type:
            try:
                import json as _json

                body = _json.loads(final_body_bytes)
            except ValueError:
                body = final_body_bytes.decode("utf-8", errors="replace")
        else:
            body = final_body_bytes.decode("utf-8", errors="replace")

        logger.info(
            "rest_api: method=%s url=%r status=%d latency_ms=%d",
            inputs.method, current_url, status_code, response_time_ms,
        )

        return RestApiOutput(
            status_code=status_code,
            headers=resp_headers,
            body=body,
            response_time_ms=response_time_ms,
            content_type=content_type,
        )
