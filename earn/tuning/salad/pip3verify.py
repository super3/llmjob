"""pip3verify.py DIR: check proofs written by `fp8-mine.js --dump DIR` with Pearl's own verifier.

Run it with the Python of a built pearl checkout (branch fp8-scheme), which has pearl_mining,
the pyo3 binding of zk-pow:  /root/pearl/.venv/bin/python pip3verify.py /root/proofs

Each file is {"header": base64 of the 76-byte proposed header, "plain_proof": base64 of the
PlainProofV4, "nbits": the target the proof was mined at}. Prints one line a proof and exits
0 only if every proof verifies.
"""
import base64
import glob
import json
import sys

from pearl_mining import IncompleteBlockHeader, PlainProofV4, verify_plain_proof_for_cert_version

CERT_VERSION_PLAIN_FP8 = 4

files = sorted(glob.glob(sys.argv[1] + "/proof-*.json"))
bad = 0
for name in files:
    d = json.load(open(name))
    header = IncompleteBlockHeader.from_bytes(base64.b64decode(d["header"]))
    proof = PlainProofV4.from_base64(d["plain_proof"])
    try:
        ok, why = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, header, proof, d["nbits"])
    except Exception as e:  # the binding raises on a malformed proof
        ok, why = False, repr(e)
    bad += 0 if ok else 1
    print("PASS" if ok else "FAIL", name, why)
print(json.dumps({"proofs": len(files), "failed": bad}))
sys.exit(1 if bad or not files else 0)
