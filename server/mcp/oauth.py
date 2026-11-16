"""OAuth 2.1 authorization server for the Mem0 MCP server.

Implements the provider interface of the MCP Python SDK, which supplies the
HTTP endpoints (/authorize, /token, /register, /revoke, metadata) and the PKCE
checks. This module adds what the SDK leaves out:

  * storage in Postgres (clients, pending requests, codes, tokens);
  * the login + consent page, which checks the password against the Mem0
    REST API so the account is the same one as the dashboard's;
  * an allow-list for client redirect URIs, because dynamic client
    registration is open to the internet.

Tokens and codes are random strings stored only as SHA-256 hashes.
"""

import asyncio
import hashlib
import html
import logging
import secrets
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

logger = logging.getLogger("mem0.mcp.oauth")

REQUEST_TTL = 10 * 60
CODE_TTL = 5 * 60
ACCESS_TOKEN_TTL = 60 * 60
REFRESH_TOKEN_TTL = 30 * 24 * 60 * 60
UNUSED_CLIENT_TTL = 24 * 60 * 60
MAX_CLIENTS = 500
CLEANUP_INTERVAL = 10 * 60

# Failed logins are counted globally: there is one account, and requests arrive
# through a proxy, so per-IP counting would add little.
MAX_FAILED_LOGINS = 8
FAILED_LOGIN_WINDOW = 10 * 60

LOGIN_PATH = "/oauth/login"

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS mcp_oauth_clients (
        client_id text PRIMARY KEY,
        info jsonb NOT NULL,
        created_at bigint NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS mcp_oauth_requests (
        id_hash text PRIMARY KEY,
        client_id text NOT NULL,
        params jsonb NOT NULL,
        expires_at bigint NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS mcp_oauth_codes (
        code_hash text PRIMARY KEY,
        client_id text NOT NULL,
        data jsonb NOT NULL,
        expires_at bigint NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS mcp_oauth_tokens (
        token_hash text PRIMARY KEY,
        kind text NOT NULL,
        grant_id text NOT NULL,
        client_id text NOT NULL,
        subject text,
        scopes jsonb NOT NULL,
        expires_at bigint NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS mcp_oauth_tokens_grant_idx ON mcp_oauth_tokens (grant_id)",
    "CREATE INDEX IF NOT EXISTS mcp_oauth_tokens_client_idx ON mcp_oauth_tokens (client_id)",
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _now() -> int:
    return int(time.time())


def _same_resource(a: str, b: str) -> bool:
    return a.rstrip("/").lower() == b.rstrip("/").lower()


class RedirectAllowList:
    """Entries are `scheme://host` (port ignored) or `scheme:` for app-specific schemes."""

    def __init__(self, entries: list[str]):
        self.entries = []
        for entry in entries:
            parsed = urlparse(entry.strip())
            if parsed.scheme:
                self.entries.append((parsed.scheme.lower(), (parsed.hostname or "").lower() or None))

    def allows(self, uri: str) -> bool:
        parsed = urlparse(uri)
        scheme, host = parsed.scheme.lower(), (parsed.hostname or "").lower()
        return any(scheme == s and (h is None or host == h) for s, h in self.entries)


class Mem0OAuthProvider:
    def __init__(
        self,
        *,
        conninfo: str,
        public_url: str,
        resource_url: str,
        api_client: httpx.AsyncClient,
        static_tokens: list[str],
        default_subject: str,
        redirect_allow_list: RedirectAllowList,
    ):
        self.public_url = public_url.rstrip("/")
        self.resource_url = resource_url
        self.api_client = api_client
        self.static_tokens = [t for t in static_tokens if t]
        self.default_subject = default_subject
        self.redirect_allow_list = redirect_allow_list
        self._pool = AsyncConnectionPool(conninfo, min_size=1, max_size=4, open=False, kwargs={"autocommit": True})
        self._ready = False
        self._ready_lock = asyncio.Lock()
        self._last_cleanup = 0
        self._failed_logins: list[float] = []

    # --- storage -----------------------------------------------------------

    async def _db(self) -> AsyncConnectionPool:
        if not self._ready:
            async with self._ready_lock:
                if not self._ready:
                    await self._pool.open()
                    async with self._pool.connection() as conn:
                        for statement in SCHEMA:
                            await conn.execute(statement)
                    self._ready = True
        if _now() - self._last_cleanup > CLEANUP_INTERVAL:
            self._last_cleanup = _now()
            await self._cleanup()
        return self._pool

    async def _execute(self, query: str, params: tuple = ()) -> list[tuple]:
        pool = await self._db()
        async with pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchall() if cursor.description else []

    async def _cleanup(self) -> None:
        now = _now()
        async with self._pool.connection() as conn:
            for table in ("mcp_oauth_requests", "mcp_oauth_codes", "mcp_oauth_tokens"):
                await conn.execute(f"DELETE FROM {table} WHERE expires_at < %s", (now,))
            # Registered but never authorized (or long since expired) clients.
            await conn.execute(
                """DELETE FROM mcp_oauth_clients c WHERE c.created_at < %s
                   AND NOT EXISTS (SELECT 1 FROM mcp_oauth_tokens t WHERE t.client_id = c.client_id)
                   AND NOT EXISTS (SELECT 1 FROM mcp_oauth_requests r WHERE r.client_id = c.client_id)
                   AND NOT EXISTS (SELECT 1 FROM mcp_oauth_codes k WHERE k.client_id = c.client_id)""",
                (now - UNUSED_CLIENT_TTL,),
            )

    # --- clients -----------------------------------------------------------

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        rows = await self._execute("SELECT info FROM mcp_oauth_clients WHERE client_id = %s", (client_id,))
        return OAuthClientInformationFull.model_validate(rows[0][0]) if rows else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            if not self.redirect_allow_list.allows(str(uri)):
                logger.warning("Rejected client registration: redirect URI %s is not in the allow-list", uri)
                raise RegistrationError("invalid_redirect_uri", f"Redirect URI is not allowed: {uri}")

        count = (await self._execute("SELECT count(*) FROM mcp_oauth_clients"))[0][0]
        if count >= MAX_CLIENTS:
            await self._cleanup()
            count = (await self._execute("SELECT count(*) FROM mcp_oauth_clients"))[0][0]
            if count >= MAX_CLIENTS:
                raise RegistrationError("invalid_client_metadata", "Too many registered clients; try again later.")

        await self._execute(
            "INSERT INTO mcp_oauth_clients (client_id, info, created_at) VALUES (%s, %s, %s)",
            (client_info.client_id, Jsonb(client_info.model_dump(mode="json")), _now()),
        )
        logger.info("Registered OAuth client %s (%s)", client_info.client_id, client_info.client_name)

    # --- authorization -----------------------------------------------------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if params.resource and not _same_resource(params.resource, self.resource_url):
            raise AuthorizeError("invalid_request", f"Unknown resource: {params.resource}")

        request_id = secrets.token_urlsafe(32)
        await self._execute(
            "INSERT INTO mcp_oauth_requests (id_hash, client_id, params, expires_at) VALUES (%s, %s, %s, %s)",
            (_hash(request_id), client.client_id, Jsonb(params.model_dump(mode="json")), _now() + REQUEST_TTL),
        )
        return f"{self.public_url}{LOGIN_PATH}?request={request_id}"

    async def _load_request(self, request_id: str) -> Optional[tuple[OAuthClientInformationFull, AuthorizationParams]]:
        rows = await self._execute(
            "SELECT client_id, params FROM mcp_oauth_requests WHERE id_hash = %s AND expires_at >= %s",
            (_hash(request_id), _now()),
        )
        if not rows:
            return None
        client = await self.get_client(rows[0][0])
        if client is None:
            return None
        return client, AuthorizationParams.model_validate(rows[0][1])

    async def _consume_request(self, request_id: str) -> bool:
        rows = await self._execute(
            "DELETE FROM mcp_oauth_requests WHERE id_hash = %s AND expires_at >= %s RETURNING client_id",
            (_hash(request_id), _now()),
        )
        return bool(rows)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        rows = await self._execute(
            "SELECT data FROM mcp_oauth_codes WHERE code_hash = %s AND client_id = %s",
            (_hash(authorization_code), client.client_id),
        )
        return AuthorizationCode.model_validate({**rows[0][0], "code": authorization_code}) if rows else None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single use: whoever deletes the row gets the tokens.
        rows = await self._execute(
            "DELETE FROM mcp_oauth_codes WHERE code_hash = %s AND client_id = %s RETURNING client_id",
            (_hash(authorization_code.code), client.client_id),
        )
        if not rows:
            raise TokenError("invalid_grant", "Authorization code has already been used.")
        return await self._issue_tokens(
            client.client_id, authorization_code.scopes, authorization_code.subject, str(uuid.uuid4())
        )

    # --- tokens ------------------------------------------------------------

    async def _issue_tokens(self, client_id: str, scopes: list[str], subject: Optional[str], grant_id: str) -> OAuthToken:
        access, refresh, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), _now()
        for token, kind, ttl in ((access, "access", ACCESS_TOKEN_TTL), (refresh, "refresh", REFRESH_TOKEN_TTL)):
            await self._execute(
                """INSERT INTO mcp_oauth_tokens (token_hash, kind, grant_id, client_id, subject, scopes, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (_hash(token), kind, grant_id, client_id, subject, Jsonb(scopes), now + ttl),
            )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=refresh,
            scope=" ".join(scopes) if scopes else None,
        )

    async def _load_token(self, token: str, kind: str) -> Optional[dict[str, Any]]:
        rows = await self._execute(
            """SELECT grant_id, client_id, subject, scopes, expires_at FROM mcp_oauth_tokens
               WHERE token_hash = %s AND kind = %s AND expires_at >= %s""",
            (_hash(token), kind, _now()),
        )
        if not rows:
            return None
        grant_id, client_id, subject, scopes, expires_at = rows[0]
        return {"grant_id": grant_id, "client_id": client_id, "subject": subject, "scopes": scopes, "expires_at": expires_at}

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        for static in self.static_tokens:
            if secrets.compare_digest(token, static):
                return AccessToken(
                    token=token,
                    client_id="static-token",
                    scopes=[],
                    resource=self.resource_url,
                    subject=self.default_subject,
                )
        row = await self._load_token(token, "access")
        if row is None:
            return None
        return AccessToken(
            token=token,
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=row["expires_at"],
            resource=self.resource_url,
            subject=row["subject"],
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> Optional[RefreshToken]:
        row = await self._load_token(refresh_token, "refresh")
        if row is None or row["client_id"] != client.client_id:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=row["expires_at"],
            resource=self.resource_url,
            subject=row["subject"],
        )

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        # Rotation: the old refresh token and its access token stop working.
        rows = await self._execute(
            """DELETE FROM mcp_oauth_tokens WHERE token_hash = %s AND kind = 'refresh' AND client_id = %s
               RETURNING grant_id""",
            (_hash(refresh_token.token), client.client_id),
        )
        if not rows:
            raise TokenError("invalid_grant", "Refresh token has already been used.")
        grant_id = rows[0][0]
        await self._execute("DELETE FROM mcp_oauth_tokens WHERE grant_id = %s", (grant_id,))
        return await self._issue_tokens(client.client_id, scopes, refresh_token.subject, grant_id)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        rows = await self._execute("SELECT grant_id FROM mcp_oauth_tokens WHERE token_hash = %s", (_hash(token.token),))
        if rows:
            await self._execute("DELETE FROM mcp_oauth_tokens WHERE grant_id = %s", (rows[0][0],))

    # --- login and consent page ---------------------------------------------

    def _login_blocked(self) -> bool:
        cutoff = time.time() - FAILED_LOGIN_WINDOW
        self._failed_logins = [t for t in self._failed_logins if t > cutoff]
        return len(self._failed_logins) >= MAX_FAILED_LOGINS

    async def _check_password(self, email: str, password: str) -> bool:
        response = await self.api_client.post("/auth/login", json={"email": email, "password": password})
        return response.status_code == 200

    async def login_endpoint(self, request: Request) -> Response:
        if request.method == "GET":
            request_id = request.query_params.get("request", "")
            loaded = await self._load_request(request_id) if request_id else None
            if loaded is None:
                return _page(_message("Запрос на доступ не найден или истёк. Начните подключение в приложении заново."), 400)
            return _page(_form(request_id, *loaded))

        form = await request.form()
        request_id = str(form.get("request", ""))
        loaded = await self._load_request(request_id) if request_id else None
        if loaded is None:
            return _page(_message("Запрос на доступ не найден или истёк. Начните подключение в приложении заново."), 400)
        client, params = loaded
        redirect_uri = str(params.redirect_uri)

        if form.get("action") != "approve":
            await self._consume_request(request_id)
            return RedirectResponse(
                construct_redirect_uri(redirect_uri, error="access_denied", state=params.state),
                status_code=302,
                headers={"Cache-Control": "no-store"},
            )

        if self._login_blocked():
            return _page(_form(request_id, client, params, "Слишком много неудачных попыток. Подождите 10 минут."), 429)

        email, password = str(form.get("email", "")).strip(), str(form.get("password", ""))
        if not email or not password or not await self._check_password(email, password):
            self._failed_logins.append(time.time())
            logger.warning("Failed OAuth login for client %s", client.client_id)
            return _page(_form(request_id, client, params, "Неверный email или пароль."), 401)

        if not await self._consume_request(request_id):
            return _page(_message("Запрос на доступ уже использован. Начните подключение заново."), 400)

        code = secrets.token_urlsafe(32)
        data = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + CODE_TTL,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=self.resource_url,
            subject=email,
        ).model_dump(mode="json")
        data.pop("code")
        await self._execute(
            "INSERT INTO mcp_oauth_codes (code_hash, client_id, data, expires_at) VALUES (%s, %s, %s, %s)",
            (_hash(code), client.client_id, Jsonb(data), _now() + CODE_TTL),
        )
        logger.info("OAuth access granted to client %s (%s)", client.client_id, client.client_name)
        return RedirectResponse(
            construct_redirect_uri(redirect_uri, code=code, state=params.state),
            status_code=302,
            headers={"Cache-Control": "no-store"},
        )


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Доступ к памяти</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f5f4; --card:#fff; --text:#1c1917; --muted:#57534e; --line:#d6d3d1; --accent:#1d4ed8; --danger:#b91c1c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --card:#292524; --text:#f5f5f4; --muted:#a8a29e; --line:#44403c; --accent:#60a5fa; --danger:#f87171; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; padding:16px; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }}
main {{ width:100%; max-width:420px; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:24px; }}
h1 {{ font-size:20px; margin:0 0 12px; }}
p {{ margin:0 0 12px; color:var(--muted); }}
dl {{ margin:0 0 16px; padding:12px; border:1px solid var(--line); border-radius:8px; }}
dt {{ font-size:13px; color:var(--muted); }} dd {{ margin:0 0 8px; overflow-wrap:anywhere; }} dd:last-child {{ margin:0; }}
label {{ display:block; font-size:14px; margin:12px 0 4px; }}
input {{ width:100%; padding:10px; font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:8px; }}
.row {{ display:flex; gap:8px; margin-top:20px; }}
button {{ flex:1; padding:10px; font:inherit; border-radius:8px; border:1px solid var(--line); background:transparent; color:inherit; cursor:pointer; }}
button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.error {{ color:var(--danger); }}
</style></head><body><main>{body}</main></body></html>"""


def _page(body: str, status_code: int = 200) -> HTMLResponse:
    return HTMLResponse(
        PAGE.format(body=body),
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _message(text: str) -> str:
    return f"<h1>Доступ к памяти</h1><p>{html.escape(text)}</p>"


def _form(
    request_id: str, client: OAuthClientInformationFull, params: AuthorizationParams, error: Optional[str] = None
) -> str:
    redirect = urlparse(str(params.redirect_uri))
    destination = f"{redirect.scheme}://{redirect.netloc}" if redirect.netloc else f"{redirect.scheme}:"
    error_html = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    return f"""<h1>Доступ к памяти</h1>
<p>Приложение просит доступ к вашей долговременной памяти: чтение, запись и удаление воспоминаний.</p>
<dl>
<dt>Приложение (название указало само приложение)</dt><dd>{html.escape(client.client_name or "без названия")}</dd>
<dt>После входа вы вернётесь на</dt><dd>{html.escape(destination)}</dd>
</dl>
<p>Разрешайте, только если вы сами начали подключение и узнаёте адрес выше.</p>
{error_html}
<form method="post" action="{LOGIN_PATH}">
<input type="hidden" name="request" value="{html.escape(request_id)}">
<label for="email">Email</label>
<input id="email" name="email" type="email" autocomplete="username" required autofocus>
<label for="password">Пароль</label>
<input id="password" name="password" type="password" autocomplete="current-password" required>
<div class="row">
<button type="submit" name="action" value="deny" formnovalidate>Отказать</button>
<button type="submit" name="action" value="approve" class="primary">Войти и разрешить</button>
</div>
</form>"""
