"""Bench client against a local server that streams like vLLM's /v1/completions."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import bench_client  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    mode = "stream"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body["max_tokens"]
        if self.mode == "error":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "bad request"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        sent = n - 1 if self.mode == "short" else n
        events = [{"choices": [{"index": 0, "text": f" t{i}"}]} for i in range(sent)]
        if self.mode == "buffered":  # the whole answer in one event
            events = [{"choices": [{"index": 0, "text": "".join(e["choices"][0]["text"] for e in events)}]}]
        events.append({"choices": [], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": sent}})
        for e in events:
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/completions"
    srv.shutdown()


def test_prompts_are_seeded_distinct_and_exact():
    a = bench_client.prompts(128, 4096, 151936, seed=0)
    assert a == bench_client.prompts(128, 4096, 151936, seed=0)
    assert len({tuple(p) for p in a}) == 128 and all(len(p) == 4096 for p in a)
    assert min(min(p) for p in a) >= 1000 and max(max(p) for p in a) < 151936 - 1000
    assert bench_client.prompts(1, 8, 151936, seed=1) != bench_client.prompts(1, 8, 151936, seed=0)


def test_body_asks_for_exact_output():
    b = bench_client.body("m", [1, 2, 3], 512)
    assert b["ignore_eos"] is True and b["stream"] is True and b["stream_options"] == {"include_usage": True}


def test_stream_one_times_tokens(server):
    Handler.mode = "stream"
    r = bench_client.stream_one(server, "k", bench_client.body("m", [5] * 16, 32))
    assert r["ok"] and r["out_tokens"] == 32 and r["chunks"] == 32 and r["prompt_tokens"] == 16
    assert r["ttft_s"] is not None and r["tpot_s"] is not None


def test_short_answer_is_a_failure(server):
    Handler.mode = "short"
    r = bench_client.stream_one(server, "k", bench_client.body("m", [5] * 16, 32))
    assert not r["ok"] and "got 31 output tokens" in r["error"]


def test_http_error_is_recorded(server):
    Handler.mode = "error"
    r = bench_client.stream_one(server, "k", bench_client.body("m", [5] * 16, 32))
    assert not r["ok"] and r["error"].startswith("HTTP 400")


def test_buffered_stream_shows_in_chunks_per_token(server):
    Handler.mode = "buffered"
    summary, _ = bench_client.run_level(server, "k", "m", [[5] * 16] * 4, 2, 32)
    assert summary["failed"] == 0 and summary["chunks_per_token"] < 0.1


def test_run_level_summary(server):
    Handler.mode = "stream"
    summary, recs = bench_client.run_level(server, "k", "m", bench_client.prompts(10, 16, 32000), 4, 64)
    assert summary["requests"] == 10 and summary["failed"] == 0 and summary["output_tokens"] == 640
    assert summary["output_tok_s"] > 0 and summary["chunks_per_token"] == 1.0
    assert len(recs) == 10


def test_percentiles():
    assert bench_client.pct([1, 2, 3, 4], 50) == 2
    assert bench_client.pct(list(range(1, 121)), 99) == 119
    assert bench_client.pct([None], 50) is None


def test_a_cut_stream_is_a_failed_request_not_a_crash():
    """A chunked stream that stops mid-chunk raises http.client.IncompleteRead inside urllib."""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     b"40\r\ndata: {\"choices\": [{\"text\": \" a\"}]}\n\n")  # promises 0x40 bytes, sends fewer
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    url = f"http://127.0.0.1:{srv.getsockname()[1]}/v1/completions"
    r = bench_client.stream_one(url, "k", bench_client.body("m", [5] * 16, 32), timeout=10)
    assert not r["ok"] and r["error"]
