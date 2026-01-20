import time

import pytest

from lunalib.core.blockchain import BlockchainManager
from lunalib.core.daemon import BlockchainDaemon
from lunalib.core.p2p import P2PClient


class DummyBlockchainManager:
    def __init__(self, latest_block):
        self._latest_block = latest_block
        self.submitted_blocks = []

    def get_latest_block(self):
        return self._latest_block

    def submit_mined_block(self, block):
        self.submitted_blocks.append(block)
        return True


def _make_block(index, previous_hash, difficulty=1, reward=1.0, transactions=None):
    return {
        "index": index,
        "previous_hash": previous_hash,
        "timestamp": time.time(),
        "transactions": transactions or [
            {
                "type": "reward",
                "to": "LUN_test",
                "amount": reward,
                "timestamp": time.time(),
                "hash": "txhash",
            }
        ],
        "miner": "miner-test",
        "difficulty": difficulty,
        "nonce": 1,
        "hash": "0" * difficulty + "a" * (64 - difficulty),
        "reward": reward,
    }


def _response(status_code, payload):
    class Response:
        def __init__(self, code, data):
            self.status_code = code
            self._data = data
            self.text = str(data)

        def json(self):
            return self._data

    return Response(status_code, payload)


def test_daemon_validates_and_accepts_block():
    latest = _make_block(0, "0" * 64)
    blockchain = DummyBlockchainManager(latest)
    daemon = BlockchainDaemon(blockchain, mempool_manager=None)

    new_block = _make_block(1, latest["hash"], difficulty=1, reward=1.0)
    result = daemon.validate_block(new_block)

    assert result["valid"] is True
    processed = daemon.process_incoming_block(new_block, from_peer="peer-1")
    assert processed.get("success") is True
    assert blockchain.submitted_blocks


def test_daemon_rejects_invalid_hash():
    latest = _make_block(0, "0" * 64)
    blockchain = DummyBlockchainManager(latest)
    daemon = BlockchainDaemon(blockchain, mempool_manager=None)

    bad_block = _make_block(1, latest["hash"], difficulty=2, reward=10.0)
    bad_block["hash"] = "b" * 64
    result = daemon.validate_block(bad_block)

    assert result["valid"] is False
    assert any("difficulty" in err.lower() for err in result.get("errors", []))


def test_blockchain_scan_chain_falls_back_to_peers(monkeypatch):
    manager = BlockchainManager(endpoint_url="https://bank.linglin.art")
    peer_url = "https://peer.example"

    def fake_get(url, timeout=30):
        if url.startswith("https://bank.linglin.art"):
            return _response(500, {"error": "down"})
        if url == f"{peer_url}/blockchain":
            return _response(200, {"blocks": [{"index": 0}]})
        return _response(404, {})

    manager._session.get = fake_get

    data = manager.scan_chain(peer_urls=[peer_url])
    assert data
    assert data.get("blocks")


def test_p2p_initial_sync_prefers_primary(monkeypatch):
    primary = "https://bank.linglin.art"
    peer = "https://peer.example"

    def fake_get(url, timeout=30):
        if url == f"{primary}/blockchain":
            return _response(200, {"blocks": ["primary"]})
        if url == f"{peer}/blockchain":
            return _response(200, {"blocks": ["peer"]})
        return _response(404, {})

    monkeypatch.setattr("requests.get", fake_get)

    client = P2PClient(primary, peer_seed_urls=[peer], prefer_peers=False)
    data = client._initial_sync()
    assert data["blocks"] == ["primary"]


def test_p2p_initial_sync_prefers_peers(monkeypatch):
    primary = "https://bank.linglin.art"
    peer = "https://peer.example"

    def fake_get(url, timeout=30):
        if url == f"{primary}/blockchain":
            return _response(200, {"blocks": ["primary"]})
        if url == f"{peer}/blockchain":
            return _response(200, {"blocks": ["peer"]})
        return _response(404, {})

    monkeypatch.setattr("requests.get", fake_get)

    client = P2PClient(primary, peer_seed_urls=[peer], prefer_peers=True)
    data = client._initial_sync()
    assert data["blocks"] == ["peer"]