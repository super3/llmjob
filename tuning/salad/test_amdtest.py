# Tests for amdtest.py's pure parts: names, classes, the create and update bodies, time parsing, log lines and
# the patch it sends, with no Salad calls. Also checks amdtest.sh's syntax and the task line parsing.
# Run: python3 -m unittest discover -s tuning/salad
import base64, contextlib, gzip, importlib.util, io, os, shutil, subprocess, sys, tempfile, unittest

os.environ["SALAD_MINE_DIR"] = tempfile.mkdtemp()     # never an exported folder, where live files may be
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("amdtest", os.path.join(HERE, "amdtest.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def read(path):
    with open(path) as f:
        return f.read()


def item(t, text, mach="m1234567-aaaa"):
    return {"time": t, "text_log": text, "resource": {"labels": {"machine_id": mach}}}


class TestAmdtest(unittest.TestCase):
    def test_only_the_two_test_groups(self):
        self.assertEqual(a.check_name("amdtest-rdna3"), "amdtest-rdna3")
        self.assertEqual(a.check_name("amdtest-rdna4"), "amdtest-rdna4")
        # Salad holds a deleted name for hours, so a numbered one can stand in.
        self.assertEqual(a.check_name("amdtest-rdna4-2"), "amdtest-rdna4-2")
        self.assertEqual(a.check_name("amdtest-rdna3-17"), "amdtest-rdna3-17")
        for bad in ("biz-rtx5060-56", "amdtest-rdna5", "bench-rx9070xt-1", "", "amdtest-rdna4-1", "amdtest-rdna4-0",
                    "amdtest-rdna4-100", "amdtest-rdna4-2x", "biz-amdtest-rdna4", "amdtest-rdna4-"):
            with self.assertRaises(SystemExit):
                a.check_name(bad)

    def test_every_command_refuses_another_name_before_calling_salad(self):
        calls = []
        old = a.w.http
        a.w.http = lambda *x, **k: calls.append(x) or (200, {})
        try:
            for argv in (["status", "biz-x"], ["delete", "biz-x"], ["realloc", "biz-x"], ["stop", "biz-x"],
                         ["set-env", "biz-x", "A=1"], ["logs", "biz-x"], ["create", "biz-x", "9070xt", "--sha", "x"]):
                with self.assertRaises(SystemExit):
                    a.main(argv)
        finally:
            a.w.http = old
        self.assertEqual(calls, [])

    def test_delete_says_the_name_stays_held(self):
        old = a.w.http
        a.w.http = lambda *x, **k: (202, {})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                a.main(["delete", "amdtest-rdna3"])
        finally:
            a.w.http = old
        self.assertIn("name_conflict", out.getvalue())

    def test_short_class_names_never_mix_xt_and_xtx(self):
        self.assertEqual(a.class_name("7900xtx"), "AMD RX 7900 XTX (24GB)")
        self.assertEqual(a.class_name("7900xt"), "AMD RX 7900 XT (20GB)")
        self.assertEqual(a.class_name("RX 9070 XT"), "AMD RX 9070 XT (16GB)")
        self.assertEqual(a.class_name("AMD RX 9060 XT (16GB)"), "AMD RX 9060 XT (16GB)")

    def test_create_body(self):
        b = a.body("amdtest-rdna3", "cid", {"BASE_SHA": "abc", "TASKS": "card,hip"})
        c = b["container"]
        self.assertEqual((b["replicas"], b["restart_policy"]), (1, "never"))
        self.assertEqual((c["image"], c["priority"]), ("rocm/dev-ubuntu-24.04:7.2", "high"))
        self.assertEqual(c["resources"], {"cpu": 4, "memory": 8192, "gpu_classes": ["cid"],
                                          "storage_amount": 10 * 1024 ** 3})
        self.assertEqual(c["command"][:2], ["/bin/sh", "-c"])
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/tuning/salad/amdtest.sh", c["command"][2])
        self.assertEqual(c["environment_variables"]["TASKS"], "card,hip")

    def test_boot_command_is_valid_sh(self):
        sh = shutil.which("dash") or shutil.which("sh")
        r = subprocess.run([sh, "-n", "-c", a.BOOT], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_boot_without_a_sha_idles(self):
        sh = shutil.which("dash") or shutil.which("sh")
        boot = a.BOOT.replace("exec sleep infinity", "exit 0")
        r = subprocess.run([sh, "-c", boot], capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
        self.assertRegex(r.stdout, r"\[amdtest\] \d\d:\d\d:\d\d boot no BASE_SHA; idle")

    def test_env_patch_sends_the_whole_map(self):
        # Salad replaces the map on a PATCH, so every variable that stays is sent again; a stored null is dropped.
        cur = {"BASE_SHA": "a", "PATCH_GZ_B64": "xx", "TASKS": "card", "WORKER": "amdtest-rdna3", "OLD": None}
        self.assertEqual(a.env_patch(cur, {"TASKS": "build"}, ["PATCH_GZ_B64", "NOPE"]),
                         {"container": {"environment_variables": {"BASE_SHA": "a", "TASKS": "build",
                                                                  "WORKER": "amdtest-rdna3"}}})

    def test_parse_kv(self):
        self.assertEqual(a.parse_kv(["TASKS=a,b:1", "X=y=z"]), {"TASKS": "a,b:1", "X": "y=z"})
        for bad in (["NOEQUALS"], ["1X=2"], ["A-B=1"]):
            with self.assertRaises(SystemExit):
                a.parse_kv(bad)

    def test_since(self):
        now = a.w.parse_iso("2026-10-22T13:30:00Z")
        self.assertEqual(a.since_t("30m", now), now - 1800)
        self.assertEqual(a.since_t("2h", now), now - 7200)
        self.assertEqual(a.since_t("13:05", now), a.w.parse_iso("2026-10-22T13:05:00Z"))
        self.assertEqual(a.since_t("13:05:30", now), a.w.parse_iso("2026-10-22T13:05:30Z"))
        self.assertEqual(a.since_t("23:50", now), a.w.parse_iso("2026-10-21T23:50:00Z"))   # yesterday
        self.assertEqual(a.since_t("2026-10-22T12:00:00Z", now), now - 5400)

    def test_pick_lines_keeps_amdtest_lines_of_one_machine_in_order(self):
        items = [item("2026-10-22T13:02:00Z", "[amdtest] 13:01:58 hip | PASS"),
                 item("2026-10-22T13:01:00Z", "x [amdtest] 13:00:58 boot container started"),
                 item("2026-10-22T13:01:30Z", "apt noise"),
                 item("2026-10-22T13:01:40Z", "[amdtest] 13:01:38 card target gfx1100", mach="other-pc"),
                 item("2026-10-22T13:02:00Z", "[amdtest] 13:01:58 hip | PASS")]       # a repeat
        got = a.pick_lines(items, machine="m1234567-aaaa")
        self.assertEqual([t for _, _, t in got], ["[amdtest] 13:00:58 boot container started",
                                                  "[amdtest] 13:01:58 hip | PASS"])
        self.assertEqual(len(a.pick_lines(items)), 3)
        self.assertEqual(len(a.pick_lines(items, grep=r"card")), 1)
        self.assertEqual(a.fmt_line(*got[0]), "m1234567 [amdtest] 13:00:58 boot container started")

    def test_env_summary_hides_payloads(self):
        self.assertEqual(a.env_summary({"PATCH_GZ_B64": "x" * 50, "BASE_SHA": "abc"}),
                         "BASE_SHA=abc PATCH_GZ_B64=<50 chars>")

    def test_make_patch_carries_changed_and_new_files_and_applies(self):
        d = tempfile.mkdtemp()
        g = lambda *x: subprocess.run(["git", "-C", d, *x], check=True, capture_output=True)
        g("init", "-q")
        g("config", "user.email", "t@t")
        g("config", "user.name", "t")
        with open(os.path.join(d, "f.txt"), "w") as f:
            f.write("one\n")
        g("add", "f.txt")
        g("commit", "-qm", "base")
        base = a.head(d)
        with open(os.path.join(d, "f.txt"), "w") as f:
            f.write("two\n")
        with open(os.path.join(d, "new.txt"), "w") as f:
            f.write("new\n")
        b64, diff = a.make_patch(d, base)
        self.assertEqual(gzip.decompress(base64.b64decode(b64)), diff)
        self.assertIn(b"new.txt", diff)
        # The repo's own index is untouched: new.txt is still untracked.
        st = subprocess.run(["git", "-C", d, "status", "--porcelain"], capture_output=True, text=True).stdout
        self.assertIn("?? new.txt", st)
        # And it applies to a clean copy of base, as the box does.
        e = tempfile.mkdtemp()
        subprocess.run(["git", "clone", "-q", d, e], check=True)
        subprocess.run(["git", "-C", e, "checkout", "-q", base], check=True)
        subprocess.run(["git", "-C", e, "apply"], input=diff, check=True)
        self.assertEqual(read(os.path.join(e, "f.txt")), "two\n")
        self.assertEqual(read(os.path.join(e, "new.txt")), "new\n")

    def test_box_script_syntax_and_line_shape(self):
        sh = os.path.join(HERE, "amdtest.sh")
        r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # emit cuts a long line into pieces of at most 400 characters, each a whole [amdtest] line.
        src = read(sh)
        fns = src[src.index("say() {"):src.index("idle() {")]
        long = "x" * 1000
        r = subprocess.run(["bash", "-c", fns + f'\nprintf "short\\n\\n{long}\\n" | emit tag'], capture_output=True,
                           text=True)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertRegex(lines[0], r"^\[amdtest\] \d\d:\d\d:\d\d tag \| short$")
        self.assertTrue(all(len(l.split(" | ", 1)[1]) <= 401 for l in lines))
        self.assertTrue(lines[2].split(" | ", 1)[1].startswith("+x"))
        self.assertEqual(sum(l.split(" | ", 1)[1].lstrip("+").count("x") for l in lines[1:]), 1000)

    def test_lib_stage_only_defines_the_functions(self):
        # bench.sh sources it this way for setup, fetch, the build, the hit check and the CLI: nothing may run.
        sh = os.path.join(HERE, "amdtest.sh")
        r = subprocess.run(["bash", "-c", f'AMDTEST_STAGE=lib . {sh}; echo "rc=$?"; '
                            "declare -F setup fetch target t_hip t_build t_verify mine_cli"],
                           capture_output=True, text=True, env={"PATH": os.environ["PATH"], "BASE_SHA": "abc"})
        self.assertEqual(r.stdout.splitlines(), ["rc=0", "setup", "fetch", "target", "t_hip", "t_build", "t_verify",
                                                 "mine_cli"])
        self.assertEqual(r.stderr, "")

    def test_nvtest_names_and_classes(self):
        for good in ("nvtest-rtx3090", "nvtest-rtx3090ti", "nvtest-rtx3060ti-2", "nvtest-rtx3070-17"):
            self.assertEqual(a.check_name(good), good)
        for bad in ("nvtest-rtx2080ti", "nvtest-rtx3050", "nvtest-rtx3090-1", "nvtest-3090", "nvtest-rtx3090tix",
                    "biz-nvtest-rtx3090", "nvtest-rtx3090-100"):
            with self.assertRaises(SystemExit):
                a.check_name(bad)
        self.assertEqual(a.class_name("3090", nv=True), "RTX 3090 (24 GB)")
        self.assertEqual(a.class_name("3090ti", nv=True), "RTX 3090 Ti (24 GB)")
        self.assertEqual(a.class_name("RTX 3060 Ti", nv=True), "RTX 3060 Ti (8 GB)")
        self.assertEqual(a.class_name("RTX 3080 (10 GB)", nv=True), "RTX 3080 (10 GB)")   # Salad's own name
        self.assertEqual(a.class_name("7900xt"), "AMD RX 7900 XT (20GB)")                 # AMD's unchanged

    def test_nvtest_create_body_runs_nvtest_sh_on_the_fleet_image(self):
        c = a.body("nvtest-rtx3090", "cid", {"BASE_SHA": "abc"})["container"]
        self.assertEqual(c["image"], a.w.IMAGE)
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/tuning/salad/nvtest.sh", c["command"][2])
        self.assertNotIn("amdtest", c["command"][2])
        self.assertEqual(c["resources"]["gpu_classes"], ["cid"])
        self.assertEqual(a.body("amdtest-rdna3", "cid", {})["container"]["image"], a.IMAGE)

    def test_nvtest_create_asks_for_the_rtx_class(self):
        calls = []

        def http(method, url, body=None, **k):
            calls.append((method, url, body))
            if url.endswith("/gpu-classes"):
                return 200, {"items": [{"name": "RTX 3090 (24 GB)", "id": "c3090"},
                                       {"name": "RTX 3090 Ti (24 GB)", "id": "c3090ti"}]}
            return (201, {}) if method == "POST" else (200, {"current_state": {"status": "running"}})
        old, old_sleep = a.w.http, a.time.sleep
        a.w.http, a.time.sleep = http, lambda s: None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a.main(["create", "nvtest-rtx3090", "3090", "--sha", "f" * 40, "--env", "VARIANTS_SPEC=base|;x|-DX"])
        finally:
            a.w.http, a.time.sleep = old, old_sleep
        posts = [b for m, u, b in calls if m == "POST" and u == a.w.CB]
        self.assertEqual(len(posts), 1)
        c = posts[0]["container"]
        self.assertEqual(c["resources"]["gpu_classes"], ["c3090"])
        self.assertEqual(c["environment_variables"], {"VARIANTS_SPEC": "base|;x|-DX", "WORKER": "nvtest-rtx3090",
                                                      "BASE_SHA": "f" * 40})

    def test_nvtest_boot_is_valid_sh_and_idles_without_a_sha(self):
        sh = shutil.which("dash") or shutil.which("sh")
        r = subprocess.run([sh, "-n", "-c", a.NV_BOOT], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([sh, "-c", a.NV_BOOT.replace("exec sleep infinity", "exit 0")], capture_output=True,
                           text=True, env={"PATH": os.environ["PATH"]})
        self.assertRegex(r.stdout, r"\[nvtest\] \d\d:\d\d:\d\d boot no BASE_SHA; idle")

    def test_nvtest_lines_are_read_by_their_tag(self):
        items = [item("2026-10-29T05:01:00Z", "[nvtest] 05:00:58 boot sha abc"),
                 item("2026-10-29T05:02:00Z", "[amdtest] 05:01:58 hip | PASS"),
                 item("2026-10-29T05:03:00Z", "x [nvtest] 05:02:58 [compilers] bench r1 base 124.9 TH/s")]
        self.assertEqual([t for _, _, t in a.pick_lines(items, tag=a.tag("nvtest-rtx3090"))],
                         ["[nvtest] 05:00:58 boot sha abc", "[nvtest] 05:02:58 [compilers] bench r1 base 124.9 TH/s"])
        self.assertEqual(len(a.pick_lines(items, tag=a.tag("amdtest-rdna3"))), 1)

    def test_nvtest_sh_syntax_functions_and_line_shape(self):
        sh = os.path.join(HERE, "nvtest.sh")
        r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "nvidia-smi"), "w") as f:
            f.write("#!/bin/bash\necho '8.6'\n")
        os.chmod(os.path.join(d, "nvidia-smi"), 0o755)
        long = "x" * 1000
        # The script's path as an argument, not in the command: a Windows path's backslashes would be escapes there.
        r = subprocess.run(["bash", "-c", 'NVTEST_STAGE=lib . "$0"; echo "rc=$?"; declare -F setup cuda fetch card_gc; '
                            f'card_gc; printf "short\\n\\n{long}\\n" | emit', sh],
                           capture_output=True, text=True, env={"PATH": d + os.pathsep + os.environ["PATH"]})
        lines = r.stdout.splitlines()
        self.assertEqual(lines[:6], ["rc=0", "setup", "cuda", "fetch", "card_gc",
                                     "-gencode arch=compute_86,code=sm_86"])
        self.assertRegex(lines[6], r"^\[nvtest\] \d\d:\d\d:\d\d short$")
        rest = [l.split(" ", 2)[2] for l in lines[7:]]
        self.assertEqual(len(rest), 3)
        self.assertTrue(all(len(t) <= 401 for t in rest) and rest[1].startswith("+x"))
        self.assertEqual(sum(t.lstrip("+").count("x") for t in rest), 1000)
        self.assertEqual(r.stderr, "")

    def test_nvtest_sh_pins_cuda_12_8_1(self):
        src = read(os.path.join(HERE, "nvtest.sh"))
        for part in ("cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz|9961b3484b6b",
                     "cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz|8d566b5fe745",
                     "cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz|0740e9e01e4f",
                     "cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz|dbf41355a0f7"):
            self.assertIn(part, src)
        self.assertIn('echo "$sha  $f" | sha256sum -c --quiet -', src)
        self.assertIn('[ "$("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE \'V[0-9.]+\' | tail -1)" != V12.8.93 ]',
                      src)
        self.assertIn("export GC=${GC:-$(card_gc)} FOLD=${FOLD:-pearl_tile_fold_tall}", src)


    def test_mine_runs_the_cli_through_mine_cli(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        body = src[src.index("mine_cli() {"):src.index("\n}\n", src.index("mine_cli() {"))]
        self.assertIn("PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$A/build/pearl_core_hip.node NODE_PATH=$NM/node_modules", body)
        self.assertIn('--address "$WALLET" --mode mining --pool "$POOL" -w "$WORKER"', body)
        self.assertIn("(mine_cli) > /root/mine.log 2>&1 &", src)


class TestCompilersMemoryReadings(unittest.TestCase):
    """compilers.sh (earn/native/probes/turing), which nvtest.sh runs: med.py's and sum.py's memory columns."""
    SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SRC = read(os.path.join(HERE, "..", "..", "earn", "native", "probes", "turing", "compilers.sh"))

    def script(self, name):
        start = self.SRC.index(f'cat > "$OUT/{name}" <<\'EOF\'\n') + len(f'cat > "$OUT/{name}" <<\'EOF\'\n')
        body = self.SRC[start:self.SRC.index("\nEOF\n", start)]
        d = tempfile.mkdtemp()
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(body + "\n")
        return p, d

    def run_py(self, *args):
        return subprocess.run([sys.executable, "-I", *args], capture_output=True, text=True, check=True).stdout

    def test_the_sampler_asks_for_the_memory_fields_med_py_reads(self):
        self.assertIn("--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.memory,memory.used,clocks.mem", self.SRC)
        med, d = self.script("med.py")
        smi = os.path.join(d, "smi")
        with open(smi, "w") as f:
            f.write("100.0,1620,350.1,70,22,1366,9501\n101.0,1635,349.9,70,20,1366,9501\n"
                    "102.0,1620,350.0,71,[N/A],1366,9501\n103,1600,349,70\n")
        self.assertEqual(self.run_py(med, smi, "100", "103").split(), "4 1620.0 349.9 70.0 1618.8 21 1366 9501".split())
        with open(smi, "w") as f:
            f.write("100.0,1620,350.1,70\n")      # a card that gives no memory readings
        self.assertEqual(self.run_py(med, smi, "100", "101").split(), "1 1620.0 350.1 70.0 1620.0 - - -".split())
        self.assertEqual(self.run_py(med, smi, "200", "201").split(), "0 0 0 0 0 - - -".split())

    def test_sum_py_shows_memory_where_rounds_have_it(self):
        s, d = self.script("sum.py")
        t = os.path.join(d, "b.tsv")
        with open(t, "w") as f:
            f.write("1\tbase\t124.9\t1620\t350\t70\t77.1\t1622\t22\t1366\t9501\n"
                    "1\tdram\t130.0\t1680\t350\t70\t77.4\t1681\t1\t1300\t9501\n"
                    "2\tdram\t130.2\t1680\t350\t70\t77.5\t1680\t1\t1300\t9501\n"
                    "2\tbase\t125.0\t1620\t350\t70\t77.2\t1621\t20\t1366\t9501\n")
        out = self.run_py(s, t, "bench-sum", os.path.join(d, "best")).splitlines()
        self.assertIn("77.150 TH/s/GHz mem 21% 1366 MiB", out[0])
        self.assertIn("77.450 TH/s/GHz mem 1% 1300 MiB vs base: raw +4.12%", out[1])
        with open(os.path.join(d, "best")) as f:
            self.assertEqual(f.read(), "dram")
        with open(t, "a") as f:           # a -DPEARL_TALL_CYC build's cycles a tile, in column 12
            f.write("1\tcyc\t124.0\t1620\t350\t70\t76.5\t1620\t22\t1366\t9501\t102655\n"
                    "2\tcyc\t124.1\t1620\t350\t70\t76.6\t1620\t22\t1366\t9501\t102661\n")
        out = self.run_py(s, t, "bench-sum", os.path.join(d, "best")).splitlines()
        self.assertIn("76.550 TH/s/GHz mem 22% 1366 MiB cyc 102658 (95.76%) vs base", out[2])
        self.assertNotIn("cyc", out[0])
        with open(t, "w") as f:           # a table from before: no memory columns, and no "mem" in the line
            f.write("1\tbase\t124.9\t1620\t350\t70\t77.1\n1\tx\t125.9\t1620\t350\t70\t77.7\n")
        out = self.run_py(s, t, "hashrate-sum", os.path.join(d, "best2"))
        self.assertNotIn(" mem ", out)
        self.assertIn("x n=1 125.900 TH/s", out)

    def test_gc_and_fold_default_to_sm_75_and_its_fold(self):
        self.assertIn('GC=${GC:-"-gencode arch=compute_75,code=sm_75"}', self.SRC)
        self.assertIn("FOLD=${FOLD:-pearl_tile_fold_wmma}", self.SRC)
        self.assertIn('[ "$SM" = 75 ] && rg=$(rg_hash "$K/build/Release/pearl_core.node")', self.SRC)
        self.assertIn('if has probe && [ -f "$OUT/v/base/ok" ] && [ "$SM" = 75 ]; then', self.SRC)
        r = subprocess.run(["bash", "-c", "GC='-gencode arch=compute_86,code=sm_86'; "
                            + self.SRC[self.SRC.index("SM=$("):self.SRC.index("\n", self.SRC.index("SM=$("))]
                            + '; echo "$SM"'], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "86")


if __name__ == "__main__":
    unittest.main()
