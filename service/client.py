from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid


class ClientError(RuntimeError):
    def __init__(self, message, *, status=None, response=None):
        super().__init__(message)
        self.status = status
        self.response = response


def build_multipart(csv_content, *, origin=None, horizon_months=24):
    boundary = "daizen-" + uuid.uuid4().hex
    chunks = []
    fields = {"horizon_months": str(horizon_months)}
    if origin is not None:
        fields["origin"] = origin
    for name, value in fields.items():
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\""
                       f"\r\n\r\n{value}\r\n").encode("utf-8"))

    chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                   "filename=\"sales.csv\"\r\nContent-Type: text/csv\r\n\r\n").encode("ascii"))
    chunks += [csv_content, f"\r\n--{boundary}--\r\n".encode("ascii")]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def predict_file(csv_path, *, url="http://127.0.0.1:8000/predict", origin=None,
                 horizon_months=24, timeout=600):
    body, content_type = build_multipart(Path(csv_path).read_bytes(), origin=origin,
                                         horizon_months=horizon_months)
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": content_type, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(text)
        except ValueError:
            details = text
        raise ClientError(f"Prediction service returned HTTP {exc.code}",
                          status=exc.code, response=details) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ClientError(f"Could not connect to prediction service: {exc}") from exc


def main():
    from service.console import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Minimal backend integration client, using only Python's standard library.\n\npython -m service.client service/examples/sample_sales.csv --origin 2026-07\n")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--origin")
    parser.add_argument("--horizon-months", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, help="New response JSON path; existing files are not overwritten")
    args = parser.parse_args()
    try:
        response = predict_file(args.csv, url=args.url, origin=args.origin,
                                horizon_months=args.horizon_months, timeout=args.timeout)
        content = json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(f"Response saved: {args.output.resolve()}")
        else:
            print(content, end="")
    except ClientError as exc:
        print(json.dumps({"error": str(exc), "http_status": exc.status,
                          "response": exc.response}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Client error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
