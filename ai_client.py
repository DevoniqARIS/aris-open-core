"""Responses API client with safe diagnostics and strict structured-output handling."""

import json
import httpx


class ModelUnavailable(RuntimeError):
    pass


def _safe_api_error(response: httpx.Response) -> str:
    """Return useful OpenAI error info without ever including the API key."""
    status = response.status_code
    try:
        body = response.json()
        err = body.get("error") or {}
        message = str(err.get("message") or "").strip()
        code = str(err.get("code") or err.get("type") or "").strip()
        parts = [f"OpenAI HTTP {status}"]
        if code:
            parts.append(code)
        if message:
            # Keep UI/log output concise and avoid dumping provider payloads.
            parts.append(message[:500])
        return ": ".join(parts)
    except Exception:
        return f"OpenAI HTTP {status}"


def _structured_schema(schema: dict) -> dict:
    """
    Convert Pydantic JSON Schema into OpenAI strict Structured Outputs schema.

    In strict mode OpenAI requires every key declared in an object's `properties`
    to also appear in that object's `required` array. "Optional" fields therefore
    remain required at the JSON level but accept null through Pydantic's anyOf
    [value_type, null] schema.

    ARIS still enforces numeric/string bounds locally with Pydantic and the
    deterministic safety gate before any aircraft movement.
    """
    unsupported = {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minItems", "maxItems", "minLength", "maxLength",
        "pattern", "format", "default",
    }

    def walk(value):
        if isinstance(value, dict):
            out = {
                k: walk(v)
                for k, v in value.items()
                if k not in unsupported
            }

            # OpenAI strict Structured Outputs: every property must be listed
            # as required. Nullable fields are represented with anyOf(..., null).
            props = out.get("properties")
            if isinstance(props, dict):
                out["required"] = list(props.keys())
                out["additionalProperties"] = False

            return out

        if isinstance(value, list):
            return [walk(v) for v in value]

        return value

    return walk(schema)


class ResponsesJSON:
    def __init__(self, key, model, client=None):
        key = (key or "").strip().strip('"').strip("'")
        model = (model or "").strip().strip('"').strip("'")
        if not key or not model:
            raise ModelUnavailable("API key and model name must be configured")
        self.key = key
        self.model = model
        self.client = client

    async def generate_with_meta(self, instructions, content, schema, schema_name="aris_assessment"):
        payload = {
            "model": self.model,
            "store": False,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": _structured_schema(schema),
                }
            },
        }

        async def call(c):
            try:
                r = await c.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": "Bearer " + self.key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TimeoutException as e:
                raise ModelUnavailable("OpenAI request timed out") from e
            except httpx.RequestError as e:
                raise ModelUnavailable(
                    f"OpenAI network error: {type(e).__name__}"
                ) from e

            if r.status_code >= 400:
                raise ModelUnavailable(_safe_api_error(r))
            return r.json()

        try:
            if self.client is not None:
                body = await call(self.client)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=10)) as c:
                    body = await call(c)

            if body.get("status") != "completed":
                detail = body.get("incomplete_details") or body.get("error") or body.get("status")
                raise ModelUnavailable(f"OpenAI response did not complete: {str(detail)[:400]}")

            texts = []
            for item in body.get("output", []):
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise ModelUnavailable("OpenAI model refused the structured request")
                    if part.get("type") == "output_text":
                        texts.append(part.get("text", ""))

            if len(texts) != 1:
                raise ModelUnavailable(
                    f"OpenAI returned {len(texts)} output_text blocks; expected exactly 1"
                )

            try:
                parsed = json.loads(texts[0])
            except json.JSONDecodeError as e:
                raise ModelUnavailable(
                    f"OpenAI returned non-JSON structured output: {e.msg}"
                ) from e

            meta = {
                "response_id": body.get("id"),
                "model": body.get("model") or self.model,
                "status": body.get("status"),
            }
            return parsed, meta
        except ModelUnavailable:
            raise
        except (ValueError, KeyError, TypeError) as e:
            raise ModelUnavailable(
                f"OpenAI response parsing failed: {type(e).__name__}: {str(e)[:300]}"
            ) from e

    async def generate(self, instructions, content, schema):
        parsed, _ = await self.generate_with_meta(instructions, content, schema)
        return parsed
