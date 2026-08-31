# Copyright (c) 2026 Tristan Penman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Send a query to a running vlm-rknn-server instance.

The query may optionally include an image. When `--image` is supplied, the
image is read locally, base64-encoded, and uploaded in the request body as
`image_data`, so the client does not need to share a filesystem with the
server. The server limits uploads to 1 MB once decoded. When `--image` is
omitted, the query is sent as a plain text request.

Examples:
    python -m vlm_rknn.send_query "Tell me a joke."
    python -m vlm_rknn.send_query --image data/cell.png "What is in the image?"
    python -m vlm_rknn.send_query --host 192.168.1.50 --model-id qwen2-vl \\
        --image data/pythagoras.png "Transcribe any text in the image."
"""

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request

# Mirrors kMaxUploadedImageBytes in cpp/src/server.cc.
MAX_UPLOAD_BYTES = 1024 * 1024

# The placeholder the prompt must contain for the server to run the vision
# encoder on the supplied image.
IMAGE_PLACEHOLDER = "<image>"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Send a query to vlm-rknn-server, optionally with an image.")
    parser.add_argument("query", help="Text query to send to the server.")
    parser.add_argument("--image",
                        help="Path to a local image file (PNG or JPEG) to "
                             "include with the query. If omitted, the query is "
                             "sent as a plain text request.")
    parser.add_argument("--model-id",
                        help="Model id to use; defaults to the server's default model.")
    parser.add_argument("--host", default="localhost",
                        help="Server host (default: %(default)s).")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port (default: %(default)s).")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Request timeout in seconds (default: %(default)s).")
    return parser.parse_args(argv)


def build_request_body(args):
    if args.image is None:
        body = {"prompt": args.query}
        if args.model_id:
            body["model_id"] = args.model_id
        return body

    with open(args.image, "rb") as image_file:
        image_bytes = image_file.read()

    if not image_bytes:
        raise ValueError("image file is empty: " + args.image)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "image is {} bytes; the server rejects uploads over {} bytes".format(
                len(image_bytes), MAX_UPLOAD_BYTES))

    # Ensure the prompt contains the placeholder, otherwise the server will not
    # process the image.
    prompt = args.query
    if IMAGE_PLACEHOLDER not in prompt:
        prompt = IMAGE_PLACEHOLDER + prompt

    body = {
        "prompt": prompt,
        "image_data": base64.b64encode(image_bytes).decode("ascii"),
    }
    if args.model_id:
        body["model_id"] = args.model_id
    return body


def main(argv):
    args = parse_args(argv)

    try:
        body = build_request_body(args)
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    url = "http://{}:{}/query".format(args.host, args.port)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            pass
        print("error: server returned {}: {}".format(error.code, detail),
              file=sys.stderr)
        return 1
    except urllib.error.URLError as error:
        print("error: could not reach {}: {}".format(url, error.reason),
              file=sys.stderr)
        return 1

    print(payload.get("text", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
