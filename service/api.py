from __future__ import annotations

import argparse
from datetime import date
import logging
import re
import threading

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from service.errors import RequestError, ServiceError


MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 64 * 1024
LOGGER = logging.getLogger(__name__)
ALLOWED_FILE_TYPES = {
    "text/csv", "application/csv", "text/plain", "application/octet-stream",
    "application/vnd.ms-excel",
}

UPLOAD_CONTRACT = {
    "requestBody": {
        "required": True,
        "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["file"], "additionalProperties": False,
            "properties": {
                "file": {"type": "string", "format": "binary",
                         "description": "UTF-8 CSV (optional BOM), maximum 25 MiB: Country,Month,Brand,Model,Sales."},
                "origin": {"type": "string", "pattern": r"^[0-9]{4}-[0-9]{2}$",
                           "description": "Shared last observation month; defaults to latest valid Month in the CSV."},
                "horizon_months": {"type": "integer", "minimum": 1, "maximum": 24,
                                   "default": 24, "description": "Number of monthly forecasts; not cumulative sales."},
            },
        }}},
    },
}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


class _TransportError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


async def _bounded_request_body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        if re.fullmatch(r"[0-9]+", declared) is None:
            raise _TransportError(422, "INVALID_REQUEST", "Invalid Content-Length header.")
        if int(declared) > MAX_REQUEST_BYTES:
            raise _TransportError(413, "REQUEST_TOO_LARGE", "Multipart request exceeds the size limit.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise _TransportError(413, "REQUEST_TOO_LARGE", "Multipart request exceeds the size limit.")
        body.extend(chunk)
    return bytes(body)


async def _parse_upload(request: Request) -> tuple[bytes, str | None, int]:
    if request.query_params:
        raise _TransportError(422, "INVALID_REQUEST", "Pass only file, origin and horizon_months as multipart fields.")
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise _TransportError(415, "UNSUPPORTED_MEDIA_TYPE", "Use multipart/form-data with a CSV file.")


    body = await _bounded_request_body(request)
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    bounded = Request(request.scope, receive=receive)
    try:
        async with bounded.form(max_files=4, max_fields=8, max_part_size=4096) as form:
            names = [name for name, _ in form.multi_items()]
            if len(names) != len(set(names)) or set(names) - {"file", "origin", "horizon_months"}:
                raise _TransportError(422, "INVALID_REQUEST", "Unknown or duplicate multipart field.")
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise _TransportError(422, "INVALID_REQUEST", "Exactly one uploaded CSV file is required in field 'file'.")
            file_type = (upload.content_type or "application/octet-stream").partition(";")[0].strip().lower()
            if file_type not in ALLOWED_FILE_TYPES:
                raise _TransportError(415, "UNSUPPORTED_MEDIA_TYPE", "Uploaded file must be a UTF-8 CSV.")

            horizon_text = form.get("horizon_months", "24")
            if not isinstance(horizon_text, str) or re.fullmatch(r"[0-9]{1,2}", horizon_text) is None:
                raise _TransportError(422, "INVALID_HORIZON", "horizon_months must be an integer from 1 to 24.")
            horizon = int(horizon_text)
            if not 1 <= horizon <= 24:
                raise _TransportError(422, "INVALID_HORIZON", "horizon_months must be an integer from 1 to 24.")
            origin = form.get("origin")
            if "origin" in form:
                if not isinstance(origin, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}", origin) is None:
                    raise _TransportError(422, "INVALID_ORIGIN", "origin must be a valid month in YYYY-MM format.")
                try:
                    date.fromisoformat(origin + "-01")
                except ValueError:
                    raise _TransportError(422, "INVALID_ORIGIN", "origin must be a valid month in YYYY-MM format.") from None

            content = await upload.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise _TransportError(413, "FILE_TOO_LARGE", "CSV file exceeds the 25 MiB limit.")
            if not content:
                raise _TransportError(422, "EMPTY_FILE", "CSV file is empty.")
            try:
                content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise _TransportError(422, "INVALID_ENCODING", "CSV must use UTF-8 encoding; a UTF-8 BOM is accepted.") from None
            return content, origin, horizon
    except HTTPException:
        raise _TransportError(422, "INVALID_REQUEST", "Malformed multipart upload.") from None


def create_app(predictor=None, config_path=None) -> FastAPI:

    app = FastAPI(
        title="DAIZEN monthly automobile sales forecast", version="4.0.0",
        description="Saved original models only. No request-time training or model selection. Sales forecasts are monthly, not cumulative.",
    )
    app.state.predictor = predictor
    initialization_lock = threading.Lock()

    def get_predictor():
        with initialization_lock:
            if app.state.predictor is None:
                from service.inference import Predictor
                app.state.predictor = Predictor(config_path=config_path)
            return app.state.predictor

    @app.exception_handler(RequestError)
    async def request_error_handler(_request, exc):
        return _error(422, exc.code, exc.message)

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request, exc):
        return _error(503, exc.code, exc.message)

    @app.exception_handler(_TransportError)
    async def transport_error_handler(_request, exc):
        return _error(exc.status, exc.code, exc.message)

    @app.get("/health", description='Check HTTP reachability and model configuration, without claiming weights are loaded.')
    async def health():

        try:
            await run_in_threadpool(get_predictor)
        except ServiceError:
            raise
        except Exception:
            LOGGER.exception("Service configuration could not be initialized")
            return _error(503, "INFERENCE_UNAVAILABLE", "Forecast service is unavailable. Check the server logs.")
        return {"status": "ok", "service": "daizen-sales-forecast",
                "configuration": "ready", "model_loading": "on-demand"}

    @app.post("/predict", openapi_extra=UPLOAD_CONTRACT,
              responses={422: {"description": "Invalid request or CSV"},
                         413: {"description": "Upload too large"},
                         415: {"description": "Unsupported media type"},
                         503: {"description": "Model or inference service unavailable"}})
    async def predict(request: Request):
        content, origin, horizon = await _parse_upload(request)

        def run_prediction():
            return get_predictor().predict_csv(content, origin=origin, horizon_months=horizon)

        try:
            result = await run_in_threadpool(run_prediction)
            return JSONResponse(content=result)
        except (RequestError, ServiceError):
            raise
        except Exception:
            LOGGER.exception("Forecast request failed")
            return _error(503, "INFERENCE_UNAVAILABLE", "Forecast service is unavailable. Check the server logs.")

    return app


def main(argv=None) -> None:
    from service.console import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Run DAIZEN saved-model inference API (one worker).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Trusted server-side model configuration JSON.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    import uvicorn
    uvicorn.run(create_app(config_path=args.config), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
