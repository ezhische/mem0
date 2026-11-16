"""MCP server (streamable HTTP) for a self-hosted Mem0 REST server.

Exposes memory tools to any MCP client and forwards them to the REST API.
All clients share one memory space (MCP_DEFAULT_USER_ID), which is the point:
the same memories are visible from every assistant connected to this server.

Authentication (at least one must be configured):
  * MCP_AUTH_TOKEN  - clients send `Authorization: Bearer <token>` to /mcp
  * MCP_URL_SECRET  - clients that cannot send headers (web connectors)
                      connect to /mcp/<secret> instead
"""

import json
import os
import secrets
from typing import Any, Optional

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

MEM0_API_URL = os.environ.get("MEM0_API_URL", "http://mem0:8000").rstrip("/")
MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
DEFAULT_USER_ID = os.environ.get("MCP_DEFAULT_USER_ID") or "default"
AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")
URL_SECRET = os.environ.get("MCP_URL_SECRET", "")
MIN_SECRET_LENGTH = 32
MCP_PATH = "/mcp"

for _name, _value in (("MCP_AUTH_TOKEN", AUTH_TOKEN), ("MCP_URL_SECRET", URL_SECRET)):
    if _value and len(_value) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"{_name} must be at least {MIN_SECRET_LENGTH} characters (try `openssl rand -hex 32`).")
if not AUTH_TOKEN and not URL_SECRET:
    raise RuntimeError("Set MCP_AUTH_TOKEN and/or MCP_URL_SECRET; the MCP server refuses to run without auth.")

mcp = FastMCP(
    "mem0",
    instructions=(
        "Long-term memory shared between all of the user's AI assistants. "
        "Search it when earlier context could help: the user's preferences, projects, "
        "decisions, or anything they refer to as already known. "
        "Save durable facts, preferences, and decisions that will matter in later conversations; "
        "do not save secrets or throwaway details."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path=MCP_PATH,
    # Requests arrive through a reverse proxy under a public hostname and are
    # authenticated by token, so Host-header allow-listing is not used.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_client = httpx.AsyncClient(base_url=MEM0_API_URL, headers={"X-API-Key": MEM0_API_KEY}, timeout=180.0)


async def _api(method: str, path: str, **kwargs: Any) -> Any:
    response = await _client.request(method, path, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Mem0 API returned HTTP {response.status_code}: {detail}")
    return response.json()


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


@mcp.tool()
async def add_memory(text: str, metadata: Optional[dict] = None, infer: bool = True) -> str:
    """Save information to long-term memory.

    Pass the facts in plain language with enough context to be understood later
    (who/what/why, not just a bare value). By default the server extracts and
    de-duplicates individual facts from the text; set infer=false to store the
    text verbatim as a single memory.
    """
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": text}],
        "user_id": DEFAULT_USER_ID,
        "infer": infer,
    }
    if metadata:
        body["metadata"] = metadata
    return _dump(await _api("POST", "/memories", json=body))


@mcp.tool()
async def search_memories(query: str, top_k: int = 10) -> str:
    """Semantic search over long-term memory. Returns the most relevant memories with ids and scores.

    Phrase the query as a natural-language question or topic.
    """
    body = {"query": query, "filters": {"user_id": DEFAULT_USER_ID}, "top_k": max(1, min(top_k, 50))}
    return _dump(await _api("POST", "/search", json=body))


@mcp.tool()
async def get_memories(top_k: int = 100) -> str:
    """List stored memories (most useful for reviewing or cleaning up what is remembered)."""
    params = {"user_id": DEFAULT_USER_ID, "top_k": max(1, min(top_k, 1000))}
    return _dump(await _api("GET", "/memories", params=params))


@mcp.tool()
async def get_memory(memory_id: str) -> str:
    """Fetch a single memory by id."""
    return _dump(await _api("GET", f"/memories/{memory_id}"))


@mcp.tool()
async def update_memory(memory_id: str, text: str) -> str:
    """Replace the text of an existing memory (use when a remembered fact has changed)."""
    return _dump(await _api("PUT", f"/memories/{memory_id}", json={"text": text}))


@mcp.tool()
async def delete_memory(memory_id: str) -> str:
    """Permanently delete one memory by id."""
    return _dump(await _api("DELETE", f"/memories/{memory_id}"))


class AuthMiddleware:
    """Bearer token on /mcp, or a secret path segment (/mcp/<secret>) for header-less clients."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope["path"].rstrip("/")
        if path == "/health":
            return await self._respond(send, 200, {"status": "ok"})

        if path == MCP_PATH:
            headers = dict(scope["headers"])
            supplied = headers.get(b"authorization", b"").decode("latin-1")
            if AUTH_TOKEN and secrets.compare_digest(supplied, f"Bearer {AUTH_TOKEN}"):
                return await self.app(scope, receive, send)
            return await self._respond(
                send, 401, {"error": "Authentication required."}, [(b"www-authenticate", b"Bearer")]
            )

        prefix = MCP_PATH + "/"
        if URL_SECRET and path.startswith(prefix) and secrets.compare_digest(path[len(prefix) :], URL_SECRET):
            scope = dict(scope, path=MCP_PATH, raw_path=MCP_PATH.encode())
            return await self.app(scope, receive, send)

        return await self._respond(send, 404, {"error": "Not found."})

    @staticmethod
    async def _respond(send, status: int, body: dict, extra_headers: Optional[list] = None):
        payload = json.dumps(body).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
        await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
        await send({"type": "http.response.body", "body": payload})


app = AuthMiddleware(mcp.streamable_http_app())

if __name__ == "__main__":
    # No access log: the secret URL path must not end up in logs.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "8765")), access_log=False)
