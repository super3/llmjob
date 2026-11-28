"""pip3load.py: two helpers for pip3box.sh's "vload" stage. A measuring tool: nothing here ships.

    python pip3load.py patch PEARL_DIR
        Makes the reference miner write its hashrate. The miner only keeps a local count of credited work
        (m*n*k a launch) and prints it at shutdown. This adds a line to /root/hash.log every 10 s:
        "<unix time> <T-MAC/s> T-MAC/s total <count>". It also makes pearl-gateway save the first six
        submitPlainProof requests to /root/proofs/, so a CPU verifier can replay real proofs from this card.

    python pip3load.py load URL MODEL THREADS SECONDS [REPEATS]
        Keeps vLLM prefilling: THREADS clients each send a prompt of REPEATS * 10 random token ids (700 by
        default, so 7,000 tokens) and ask for one token. Random ids, so the prefix cache never serves one. The reference miner mines on prefills of 1,024 tokens or more, so this is
        the load that makes it mine all the time. Prints "LOAD" lines with the prompt tokens a second.
"""
import json
import os
import py_compile
import random
import subprocess
import sys
import threading
import time
import urllib.request

COUNTER = "miner/miner-base/src/miner_base/async_loop_manager.py"
SERVER = "miner/pearl-gateway/src/pearl_gateway/miner_rpc/server.py"


def patch(root):
    subprocess.run(["git", "-C", root, "checkout", "--", COUNTER, SERVER], check=True)
    path = os.path.join(root, COUNTER)
    src = open(path).read()
    old = "            self._inner_hash_counter += count\n"
    new = old + (
        "            import time as _t\n"
        "            _now = _t.time()\n"
        "            if not hasattr(self, '_f_t0'):\n"
        "                self._f_t0 = _now\n"
        "                self._f_c0 = self._inner_hash_counter\n"
        "            elif _now - self._f_t0 >= 10:\n"
        "                _r = (self._inner_hash_counter - self._f_c0) / (_now - self._f_t0) / 1e12\n"
        "                with open('/root/hash.log', 'a') as _f:\n"
        "                    print(int(_now), round(_r, 3), 'T-MAC/s total', self._inner_hash_counter, file=_f)\n"
        "                self._f_t0 = _now\n"
        "                self._f_c0 = self._inner_hash_counter\n"
    )
    if src.count(old) != 1:
        sys.exit("the hash counter is not where this patch expects it")
    open(path, "w").write(src.replace(old, new))
    py_compile.compile(path, doraise=True)

    path = os.path.join(root, SERVER)
    src = open(path).read()
    old = "                plain_proof = proof_type.from_base64(params[\"plain_proof\"])\n"
    new = (
        "                try:\n"
        "                    import json as _j, os as _o\n"
        "                    _o.makedirs('/root/proofs', exist_ok=True)\n"
        "                    _n = len(_o.listdir('/root/proofs'))\n"
        "                    if _n < 6:\n"
        "                        with open('/root/proofs/proof-' + str(_n) + '.json', 'w') as _f:\n"
        "                            _j.dump(params, _f)\n"
        "                except Exception:\n"
        "                    pass\n"
    ) + old
    if src.count(old) != 1:
        sys.exit("the gateway's submit handler is not where this patch expects it")
    open(path, "w").write(src.replace(old, new))
    py_compile.compile(path, doraise=True)
    print("patched")


def load(url, model, threads, secs, reps=700):
    ok, err, toks, last = [0], [0], [0], [""]
    end = time.time() + secs

    def work(i):
        r = random.Random(i)
        while time.time() < end:
            # Token ids, not text: tokenizing 7,000-token prompts in the API server held the load to about
            # 12,000 tokens a second on a 112-thread host, with the card at 20 to 30% use.
            prompt = [r.randrange(1000, 20000) for _ in range(reps * 10)]
            req = urllib.request.Request(url + "/v1/completions",
                                         json.dumps({"model": model, "prompt": prompt, "max_tokens": 1}).encode(),
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    toks[0] += json.load(resp)["usage"]["prompt_tokens"]
                    ok[0] += 1
            except Exception as e:  # noqa: BLE001
                err[0] += 1
                last[0] = str(e)[:150]
                time.sleep(1)

    ts = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    while time.time() < end:
        time.sleep(min(30, max(1, end - time.time())))
        print(f"LOAD {time.time() - t0:.0f}s ok {ok[0]} err {err[0]} prompt tok/s {toks[0] / (time.time() - t0):.0f} "
              f"{last[0]}", flush=True)
    for t in ts:
        t.join(timeout=700)
    print(f"LOAD done ok {ok[0]} err {err[0]} prompt tok/s {toks[0] / (time.time() - t0):.0f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "patch":
        patch(sys.argv[2])
    elif len(sys.argv) >= 6 and sys.argv[1] == "load":
        load(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 700)
    else:
        sys.exit(__doc__)
