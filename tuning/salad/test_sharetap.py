# Tests for sharetap.py's decoder: a tap log made here, with shares in the PlainProof format.
# Run: python3 -m unittest discover -s tuning/salad
import base64, gzip, importlib.util, json, os, struct, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sharetap", os.path.join(HERE, "sharetap.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def u64(*v):
    return b"".join(struct.pack("<Q", x) for x in v)


def matrix_proof(rows, fill, leaves=2, total=64):
    """A MatrixMerkleProof as plainProof.js writes it: leaves, leaf indices, total, root, siblings, rows."""
    return (u64(leaves) + b"".join(u64(1024) + bytes([fill & 0xFF]) * 1024 for _ in range(leaves))
            + u64(leaves, *range(leaves)) + u64(total) + b"\0" * 32 + u64(3) + b"\1" * 96 + u64(len(rows), *rows))


def proof(m, n, k, a_rows, b_rows, fill=48, gz=True):
    raw = u64(m, n, k, 128) + matrix_proof(a_rows, fill) + matrix_proof(b_rows, fill) + b"\0"
    return base64.b64encode(gzip.compress(raw) if gz else raw).decode()


def tap_log(recs):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "tap.jsonl")
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def line(t, c, d, msg):
    return {"t": t, "c": c, "d": d, "line": json.dumps(msg)}


class TestDecode(unittest.TestCase):
    def test_a_share_shows_its_shape_tile_and_operand_bytes_but_never_the_wallet(self):
        p = tap_log([
            {"t": 100, "c": 1, "d": "open"},
            line(100, 1, "up", {"id": 1, "method": "mining.authorize", "params": ["prl1secretwallet.bench-peak", "x"]}),
            line(101, 1, "up", {"id": 7, "method": "mining.submit",
                                "params": {"job_id": "j1", "plain_proof": proof(2048, 1024, 8192, [5, 6, 9], [40, 48])}}),
            line(102, 1, "down", {"id": 7, "result": True, "error": None}),
        ])
        out = st.decode(p)
        text = "\n".join(out)
        self.assertNotIn("prl1secretwallet", text)
        self.assertIn("bench-peak", out[0])
        self.assertIn(" ok m=2048 n=1024 k=8192 r=128 tile=3x2 gz=1", out[0])
        self.assertIn("A rows 3@5+[0,1,4] bytes distinct=1 range=48..48 top=48:100.0%", out[1])
        self.assertIn("Bt rows 2@40+[0,8]", out[2])
        self.assertEqual(out[-1], "bench-peak summary m=2048 n=1024 k=8192 r=128 tile=3x2:1 ok:1")

    def test_since_drops_connections_an_earlier_miner_opened(self):
        p = tap_log([
            {"t": 100, "c": 1, "d": "open"},
            line(100, 1, "up", {"id": 1, "method": "mining.authorize", "params": {"login": "prl1w", "worker": "bench-ours"}}),
            line(101, 1, "up", {"id": 7, "method": "mining.submit", "params": {"plain_proof": proof(1, 2, 2048, [0], [0])}}),
            {"t": 200, "c": 2, "d": "open"},
            line(200, 2, "up", {"id": 1, "method": "mining.authorize", "params": ["prl1w.bench-srb"]}),
            line(201, 2, "up", {"id": 3, "method": "mining.submit",
                                "params": ["prl1w.bench-srb", "j", proof(4, 8, 2048, [1], [2], fill=-3, gz=False)]}),
            line(202, 2, "down", {"id": 3, "result": None, "error": {"code": 23, "message": "low difficulty"}}),
        ])
        out = st.decode(p, since=150)
        self.assertTrue(all("bench-ours" not in l for l in out))
        self.assertIn("bench-srb", out[0])
        self.assertIn('rej:{"code": 23', out[0])
        self.assertIn("m=4 n=8 k=2048", out[0])
        self.assertIn("gz=0", out[0])
        self.assertIn("params=[", out[0])
        self.assertIn("range=-3..-3 top=-3:100.0%", out[1])

    def test_an_undecodable_share_says_so_and_a_log_with_none_says_no_shares(self):
        p = tap_log([
            {"t": 1, "c": 1, "d": "open"},
            line(1, 1, "up", {"id": 1, "method": "mining.authorize", "params": ["w.bench-peak"]}),
            line(2, 1, "up", {"id": 2, "method": "mining.submit", "params": {"plain_proof": "AAAA"}}),
        ])
        out = st.decode(p)
        self.assertIn("undecoded", out[0])
        self.assertIn("undecoded:1", out[-1])
        self.assertEqual(st.decode(p, want="bench-ours"), ["bench-ours: no shares"])

    def test_every_line_fits_the_log_service(self):
        rows = list(range(0, 4000, 7))
        p = tap_log([
            {"t": 1, "c": 1, "d": "open"},
            line(1, 1, "up", {"id": 1, "method": "mining.authorize", "params": ["w.bench-peak"]}),
            line(2, 1, "up", {"id": 2, "method": "mining.submit", "params": {"plain_proof": proof(8, 8, 2048, rows, rows)}}),
        ])
        self.assertTrue(all(len(l) <= 440 for l in st.decode(p)))


if __name__ == "__main__":
    unittest.main()
