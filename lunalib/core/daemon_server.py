import json
import threading
import gzip
import os

try:
    import msgpack  # type: ignore
    _HAS_MSGPACK = True
except Exception:
    msgpack = None
    _HAS_MSGPACK = False
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse, parse_qs


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DaemonHTTPHandler(BaseHTTPRequestHandler):
    daemon_ref = None
    blockchain_ref = None
    mempool_ref = None

    def log_message(self, format, *args):
        return

    def _json(self, status: int, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if self.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        content_type = self.headers.get("Content-Type", "").lower()
        if "application/msgpack" in content_type and _HAS_MSGPACK:
            return msgpack.unpackb(raw, raw=False)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/system/health", "/health"):
            return self._json(200, {"status": "ok"})

        if parsed.path in ("/blockchain", "/api/blockchain/full", "/blockchain/blocks"):
            blocks = self.blockchain_ref.blocks if hasattr(self.blockchain_ref, "blocks") else []
            return self._json(200, {"blocks": blocks})

        if parsed.path == "/api/blockchain/latest":
            latest = self.blockchain_ref.get_latest_block()
            return self._json(200, latest or {})

        if parsed.path.startswith("/blockchain/block/"):
            try:
                height = int(parsed.path.split("/")[-1])
                block = self.blockchain_ref.blocks[height]
                return self._json(200, block)
            except Exception:
                return self._json(404, {"error": "block not found"})

        if parsed.path == "/blockchain/range":
            qs = parse_qs(parsed.query)
            start = int(qs.get("start", [0])[0])
            end = int(qs.get("end", [0])[0])
            blocks = self.blockchain_ref.blocks[start : end + 1]
            return self._json(200, {"blocks": blocks})

        if parsed.path == "/mempool":
            mempool_list = []
            if hasattr(self.mempool_ref, "local_mempool"):
                mempool_list = [v["transaction"] for v in self.mempool_ref.local_mempool.values()]
            return self._json(200, mempool_list)

        if parsed.path == "/api/peers/list":
            peers = self.daemon_ref.get_peer_list()
            return self._json(200, {"peers": peers})

        return self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/mempool/add", "/api/mempool/add"):
            tx = self._read_json()
            validation = self.daemon_ref.validate_transaction(tx)
            if validation.get("valid"):
                self.mempool_ref.add_transaction(tx)
                return self._json(200, {"success": True, "transaction_hash": tx.get("hash")})
            return self._json(400, {"success": False, "error": validation.get("message")})

        if parsed.path in ("/mempool/add/batch", "/api/mempool/add/batch"):
            payload = self._read_json()
            txs = payload.get("transactions", []) if isinstance(payload, dict) else payload
            result = self.daemon_ref.process_incoming_transactions_batch(txs)
            return self._json(200, result)

        if parsed.path == "/api/transactions/new":
            tx = self._read_json()
            result = self.daemon_ref.process_incoming_transaction(tx)
            return self._json(200, result)

        if parsed.path == "/api/transactions/new/batch":
            payload = self._read_json()
            txs = payload.get("transactions", []) if isinstance(payload, dict) else payload
            result = self.daemon_ref.process_incoming_transactions_batch(txs)
            return self._json(200, result)

        if parsed.path == "/api/blocks/new":
            block = self._read_json()
            result = self.daemon_ref.process_incoming_block(block)
            return self._json(200, result)

        if parsed.path == "/api/blocks/validate":
            block = self._read_json()
            return self._json(200, self.daemon_ref.validate_block(block))

        if parsed.path == "/api/transactions/validate":
            tx = self._read_json()
            return self._json(200, self.daemon_ref.validate_transaction(tx))

        if parsed.path == "/api/peers/register":
            peer = self._read_json()
            return self._json(200, self.daemon_ref.register_peer(peer))

        if parsed.path == "/blockchain/submit-block":
            block = self._read_json()
            success = self.blockchain_ref.submit_mined_block(block)
            return self._json(200, {"success": success, "block_hash": block.get("hash")})

        return self._json(404, {"error": "not found"})


class DaemonHTTPServer:
    """HTTP server exposing daemon endpoints, including batch mempool ingestion."""

    def __init__(self, daemon, blockchain, mempool, host: str = "0.0.0.0", port: int = 8765):
        self.daemon = daemon
        self.blockchain = blockchain
        self.mempool = mempool
        self.host = host
        self.port = port
        self._server: Optional[ThreadedHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        DaemonHTTPHandler.daemon_ref = self.daemon
        DaemonHTTPHandler.blockchain_ref = self.blockchain
        DaemonHTTPHandler.mempool_ref = self.mempool
        self._server = ThreadedHTTPServer((self.host, self.port), DaemonHTTPHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
