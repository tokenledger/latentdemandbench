"""Run run_v0 with an uncompressed OpenAI HTTP response transport.

The local HTTP decoder can fail with an incompatible Brotli installation:
Decompressor.decompress() got an unexpected keyword argument output_buffer_limit.
This wrapper avoids changing shared Python packages or experiment parameters.
All CLI arguments are passed unchanged to the canonical runner; only the HTTP
Accept-Encoding header differs. Use the same --backend openai/model/effort/K.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai

import run_v0
from ldb.backends import openai_api


def main():
    previous = openai_api._CLIENT
    with openai.OpenAI(default_headers={"Accept-Encoding": "identity"}) as client:
        openai_api._CLIENT = client
        try:
            run_v0.main()
        finally:
            openai_api._CLIENT = previous


if __name__ == "__main__":
    main()
