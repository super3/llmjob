# Tests for amdtest.py's pure parts: names, classes, the create and update bodies, time parsing, log lines and
# the patch it sends, with no Salad calls. Also checks amdtest.sh's syntax and the task line parsing.
# Run: python3 -m unittest discover -s earn/tuning/salad
import base64, contextlib, gzip, importlib.util, io, os, shutil, subprocess, sys, tempfile, time, unittest

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
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/earn/tuning/salad/amdtest.sh", c["command"][2])
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

    def test_read_logs_splits_until_no_window_is_full(self):
        """Salad gives at most 100 lines a call. 5 hours at a line every 2.4 s (an nvtest race with its clock lines)
        all come back. At 6 splits, as before, each 281 s window held about 117, and 1100 lines were lost."""
        t0 = a.w.parse_iso("2026-11-01T00:00:00Z")
        times = [t0 + 2.4 * i for i in range(7500)]

        def http(method, url, body=None, **k):
            s, e = a.w.parse_iso(body["start_time"]), a.w.parse_iso(body["end_time"])
            return 200, {"items": [{"time": t} for t in times if s <= t < e][:100]}
        old = a.w.http
        a.w.http = http
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                got = a.read_logs("nvtest-rtx5090", t0, t0 + 5 * 3600)
                self.assertEqual(err.getvalue(), "")
                self.assertEqual([x["time"] for x in got], times)
                # 300 lines in one second cannot be split: 100 of them, and a warning.
                times = [t0 + i / 300 for i in range(300)]
                self.assertEqual(len(a.read_logs("nvtest-rtx5090", t0, t0 + 1)), 100)
        finally:
            a.w.http = old
        self.assertIn("still has 100+ lines after 0 splits; some may be missed", err.getvalue())

    def test_env_summary_hides_payloads(self):
        self.assertEqual(a.env_summary({"PATCH_GZ_B64": "x" * 50, "BASE_SHA": "abc"}),
                         "BASE_SHA=abc PATCH_GZ_B64=<50 chars>")

    def test_env_summary_never_prints_a_wallet_key_or_token(self):
        s = a.env_summary({"WALLET": "prl1secretish", "SALAD_API_KEY": "k", "BOT_TOKEN": "t", "POOL_SECS": "600"})
        self.assertEqual(s, "BOT_TOKEN=<set> POOL_SECS=600 SALAD_API_KEY=<set> WALLET=<set>")

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
        # Forward slashes: Git Bash on Windows (rig01) can't source a path written with backslashes.
        sh = os.path.join(HERE, "amdtest.sh").replace("\\", "/")
        r = subprocess.run(["bash", "-c", f'AMDTEST_STAGE=lib . {sh}; echo "rc=$?"; '
                            "declare -F setup fetch target t_hip t_build t_verify mine_cli"],
                           capture_output=True, text=True, env={"PATH": os.environ["PATH"], "BASE_SHA": "abc"})
        self.assertEqual(r.stdout.splitlines(), ["rc=0", "setup", "fetch", "target", "t_hip", "t_build", "t_verify",
                                                 "mine_cli"])
        self.assertEqual(r.stderr, "")

    def test_pool_wallet_and_worker_reach_child_processes(self):
        # verify-pool.js reads POOL, WALLET and WORKER from its environment, so the script must export them.
        sh = os.path.join(HERE, "amdtest.sh").replace("\\", "/")
        child = 'bash -c \'echo "${POOL:-none} ${WORKER:-none} ${WALLET:0:8}"\''
        base = {"PATH": os.environ["PATH"], "BASE_SHA": "abc"}
        r = subprocess.run(["bash", "-c", f"AMDTEST_STAGE=lib . {sh}; {child}"], capture_output=True, text=True,
                           env=base)
        self.assertEqual(r.stdout.strip(), "prl-us.kryptex.network:7048 amdtest prl1pw88")
        r = subprocess.run(["bash", "-c", f"AMDTEST_STAGE=lib . {sh}; {child}"], capture_output=True, text=True,
                           env={**base, "WALLET": "prl1given", "WORKER": "w2"})
        self.assertEqual(r.stdout.strip(), "prl-us.kryptex.network:7048 w2 prl1give")

    def test_nvtest_names_and_classes(self):
        for good in ("nvtest-rtx3090", "nvtest-rtx3090ti", "nvtest-rtx3060ti-2", "nvtest-rtx3070-17", "nvtest-rtx4070",
                     "nvtest-rtx4060ti-3", "nvtest-rtx2080ti", "nvtest-rtx2070-2", "nvtest-rtx5090", "nvtest-rtx5070ti-2",
                     "nvtest-rtx5060"):
            self.assertEqual(a.check_name(good), good)
        for bad in ("nvtest-rtx1080ti", "nvtest-rtx3050", "nvtest-rtx3090-1", "nvtest-3090", "nvtest-rtx3090tix",
                    "biz-nvtest-rtx3090", "nvtest-rtx3090-100", "nvtest-rtx5050", "nvtest-rtx6090"):
            with self.assertRaises(SystemExit):
                a.check_name(bad)
        self.assertEqual(a.class_name("3090", nv=True), "RTX 3090 (24 GB)")
        self.assertEqual(a.class_name("3090ti", nv=True), "RTX 3090 Ti (24 GB)")
        self.assertEqual(a.class_name("4070", nv=True), "RTX 4070 (12 GB)")
        self.assertEqual(a.class_name("4060ti", nv=True), "RTX 4060 Ti (16 GB)")
        self.assertEqual(a.class_name("2080ti", nv=True), "RTX 2080 Ti (11 GB)")
        self.assertEqual(a.class_name("5090", nv=True), "RTX 5090 (32 GB)")
        self.assertEqual(a.class_name("5070ti", nv=True), "RTX 5070 Ti (16 GB)")
        self.assertEqual(a.class_name("RTX 5060 Ti", nv=True), "RTX 5060 Ti (16 GB)")
        # Every RTX 50 class by Salad's own name, as the watcher knows it.
        rtx50 = {k: v for k, v in a.NV_CLASSES.items() if k.startswith("50")}
        self.assertEqual(rtx50, {"5090": "RTX 5090 (32 GB)", "5080": "RTX 5080 (16 GB)", "5070ti": "RTX 5070 Ti (16 GB)",
                                 "5070": "RTX 5070 (12 GB)", "5060ti": "RTX 5060 Ti (16 GB)", "5060": "RTX 5060 (8 GB)"})
        for name in rtx50.values():
            self.assertIn(name, a.w.KNOWN)
        self.assertEqual(a.class_name("RTX 3060 Ti", nv=True), "RTX 3060 Ti (8 GB)")
        self.assertEqual(a.class_name("RTX 3080 (10 GB)", nv=True), "RTX 3080 (10 GB)")   # Salad's own name
        self.assertEqual(a.class_name("7900xt"), "AMD RX 7900 XT (20GB)")                 # AMD's unchanged

    def test_nvtest_gtx16_names_and_classes(self):
        # GTX 16 boxes, for the dp4a fold on sm_75 without tensor cores (PEARL_DP4A_TURING).
        for good in ("nvtest-gtx1650", "nvtest-gtx1660", "nvtest-gtx1660super", "nvtest-gtx1660super-2",
                     "nvtest-gtx1650-17"):
            self.assertEqual(a.check_name(good), good)
        for bad in ("nvtest-gtx1660ti", "nvtest-gtx1650super", "nvtest-gtx1080", "nvtest-gtx1660s", "nvtest-gtx1660-1",
                    "nvtest-gtx1660-100", "biz-nvtest-gtx1660", "nvtest-rtx1660", "nvtest-gtx2060"):
            with self.assertRaises(SystemExit):
                a.check_name(bad)
        # Salad's names, as its gpu-classes listed them on 6 November 2026; exact short names only.
        self.assertEqual(a.class_name("1650", nv=True), "GTX 1650 (4 GB)")
        self.assertEqual(a.class_name("1660", nv=True), "GTX 1660 (6 GB)")
        self.assertEqual(a.class_name("1660super", nv=True), "GTX 1660 Super (6 GB)")
        self.assertEqual(a.class_name("GTX 1660 Super", nv=True), "GTX 1660 Super (6 GB)")
        self.assertEqual(a.class_name("NVIDIA GeForce GTX 1650", nv=True), "GTX 1650 (4 GB)")
        self.assertEqual(a.class_name("GTX 1660 (6 GB)", nv=True), "GTX 1660 (6 GB)")       # Salad's own name
        self.assertEqual(a.class_name("1660s", nv=True), "1660s")                           # not a short name
        # The RTX names still map as before.
        self.assertEqual(a.class_name("2060", nv=True), "RTX 2060 (6 GB)")
        c = a.body("nvtest-gtx1660super", "cid", {"BASE_SHA": "abc"})["container"]
        self.assertEqual(c["image"], a.NV_IMAGE)
        self.assertIn("nvtest.sh", c["command"][2])

    def test_nvtest_create_body_runs_nvtest_sh_on_the_fleet_image(self):
        c = a.body("nvtest-rtx3090", "cid", {"BASE_SHA": "abc"})["container"]
        self.assertEqual(c["image"], a.w.IMAGE)
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/earn/tuning/salad/nvtest.sh", c["command"][2])
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
        self.assertIn('[ "$("$CUDA/bin/nvcc" --version 2> /dev/null | grep -oE \'V[0-9.]+\' | tail -1)" != "$NVCC_VER" ]',
                      src)
        self.assertIn("export GC=${GC:-$(card_gc)} FOLD=${FOLD:-pearl_tile_fold_tall}", src)

    def nvtest_lib(self, script, env=None, driver=None):
        """Source nvtest.sh as a library, then run `script`. nvidia-smi answers `driver` (none: it prints nothing)."""
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "nvidia-smi"), "w") as f:
            f.write("#!/bin/bash\n" + (f"echo '{driver}'\n" if driver else ""))
        os.chmod(os.path.join(d, "nvidia-smi"), 0o755)
        return subprocess.run(["bash", "-c", 'NVTEST_STAGE=lib . "$0"; ' + script, os.path.join(HERE, "nvtest.sh")],
                              capture_output=True, text=True,
                              env={"PATH": d + os.pathsep + os.environ["PATH"], **(env or {})})

    def test_nvtest_sh_cuda_13_3_is_an_option(self):
        show = 'echo "$CUDA_VER $NVCC_VER"; printf "%s\\n" "$CUDA_PARTS"'
        # The default is 12.8.1, as before.
        out = self.nvtest_lib(show).stdout.splitlines()
        self.assertEqual(out[0], "12.8 V12.8.93")
        self.assertEqual([l.split("|")[0].rsplit("/", 1)[1] for l in out[1:]],
                         ["cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz", "cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz",
                          "cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz",
                          "cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz"])
        # 13.3.1: nvcc, and the crt headers and nvvm (cicc) 13.x moved out of it, from redistrib_13.3.1.json.
        out = self.nvtest_lib(show, {"CUDA_VER": "13.3"}).stdout.splitlines()
        self.assertEqual(out[0], "13.3 V13.3.73")
        self.assertEqual(out[1:], [
            "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.73-archive.tar.xz|"
            "2ff9f9954060794a1c5134a933ccb45bec723d866b2629dadfe4a1a313f21068",
            "cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.3.73-archive.tar.xz|"
            "1251aa9d668c607a103489cd2250773701e83a313e355578044622cf36713a9d",
            "libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.3.73-archive.tar.xz|"
            "206b1ab4979c09b5c32f8bf907c42bc9e16cd7454cf6036f524c45a58d060f93",
            "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.3.29-archive.tar.xz|"
            "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d",
            "cccl/linux-x86_64/cccl-linux-x86_64-13.3.3.4.1-archive.tar.xz|"
            "26957cede74f9341174ecaf0372f3f886e7c46ceccb98d6dc775fe2b68d19268",
            "cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-13.3.73-archive.tar.xz|"
            "2f3cd496b9183ca2de9e17dbcd7f992eac3cd4475052b151605d9e8445736b94"])
        # Any other version leaves nothing to install, and the box stops on it before setup.
        self.assertEqual(self.nvtest_lib(show, {"CUDA_VER": "12.9"}).stdout.splitlines(), ["12.9 ", ""])
        src = read(os.path.join(HERE, "nvtest.sh"))
        self.assertIn('[ -n "$NVCC_VER" ] || fail "CUDA_VER must be 12.8 or 13.3, not $CUDA_VER"\n'
                      'driver_ok || fail "CUDA_VER=$CUDA_VER needs driver 580 or newer, and this card\'s is older"\n'
                      "setup\ncuda\n", src)
        # compilers.sh checks nvcc against the toolkit asked for, and logs the clocks every 5 s by default.
        self.assertIn('run compilers env SRC="$R" OUT="$OUT" CUDA_HOME="$CUDA" NVCC_VER="$NVCC_VER" SMI_EVERY="${SMI_EVERY:-5}"',
                      src)

    def test_nvtest_sh_needs_driver_580_for_cuda_13(self):
        ok = 'driver_ok && echo ok || echo old'
        for ver, driver, want in (("13.3", "580.82.07", "ok"), ("13.3", "590.44.01", "ok"), ("13.3", "575.64.05", "old"),
                                  ("13.3", None, "old"), ("13.3", "[N/A]", "old"), ("12.8", "535.183.01", "ok"),
                                  ("12.8", None, "ok")):
            self.assertEqual(self.nvtest_lib(ok, {"CUDA_VER": ver}, driver).stdout.strip(), want, (ver, driver))


    def test_mine_runs_the_cli_through_mine_cli(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        body = src[src.index("mine_cli() {"):src.index("\n}\n", src.index("mine_cli() {"))]
        self.assertIn("PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$A/build/pearl_core_hip.node NODE_PATH=$NM/node_modules", body)
        self.assertIn('--address "$WALLET" --mode mining --pool "$POOL" -w "$WORKER"', body)
        self.assertIn("(mine_cli) > /root/mine.log 2>&1 &", src)

    # RDNA 4 mines k 4096 (round 4 item 5, step 2): the tasks that check its shares outside our own code.
    def lib(self, script, root):
        """Source amdtest.sh as a library with AMDTEST_ROOT=root, then run `script`; the path goes in as $0."""
        sh = os.path.join(HERE, "amdtest.sh")
        return subprocess.run(["bash", "-c", 'AMDTEST_STAGE=lib . "$0"; ' + script, sh], capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], "AMDTEST_ROOT": root.replace("\\", "/")})

    def test_verifypool_and_refverify_are_tasks(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        self.assertIn("|mine|verifypool|srb|duel|cpuchecks|refverify|run) ;;", src)
        # The spec the reference verifier is built from is pinned, and the probe is copied into its zk-pow.
        self.assertIn("PEARL_SPEC_SHA=${PEARL_SPEC_SHA:-d7c517d267f10f5bcbbdc366f3e603942f9a5ab5}", src)
        self.assertIn("cp $R/earn/native/probes/refverify.rs $spec/zk-pow/src/bin/refverify.rs", src)
        self.assertIn("cargo build --release --bin refverify", src)
        # verify writes its shares where refverify looks, and cpuchecks writes the CPU core's.
        self.assertIn("PEARL_SHARE_DUMP=/root/shares/verify.jsonl", src)
        self.assertIn("SHARE_DIR=/root/shares/cpu ./test/run-cpu-checks.sh", src)

    def test_verifypool_refuses_an_unknown_mode_and_needs_a_build(self):
        d = tempfile.mkdtemp()
        r = self.lib("t_verifypool mixed; echo rc=$?", d)
        self.assertIn("verifypool FAIL no build/pearl_core_hip.node (run build first)", r.stdout)
        self.assertIn("rc=1", r.stdout)
        os.makedirs(os.path.join(d, "earn", "native", "amd", "build"))
        open(os.path.join(d, "earn", "native", "amd", "build", "pearl_core_hip.node"), "w").close()
        r = self.lib("t_verifypool bogus; echo rc=$?", d)
        self.assertIn("verifypool FAIL the mode is mixed, cli or cliplain, not bogus", r.stdout)
        self.assertIn("rc=1", r.stdout)

    def test_tap_mines_through_the_relay_under_the_pools_own_name(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        start = src[src.index("tap_start() {"):src.index("\n}\n", src.index("tap_start() {"))]
        # The relay listens on the pool's own port, and the CLI's lookup of the pool's name answers 127.0.0.1, so
        # the login is the one the pool's name picks (Kryptex: wallet/worker).
        self.assertIn('sharetap.py tap "$port" "$host" "$port" /root/tap.jsonl', start)
        self.assertIn('export TAP_HOST=$host NODE_OPTIONS="--require /root/tap-dns.js"', start)
        self.assertIn("if (host !== process.env.TAP_HOST) return lookup.apply(this, arguments);", start)
        end = src[src.index("tap_end() {"):src.index("\n}\n", src.index("tap_end() {"))]
        for sub in ("decode", "jobs", "shares"):
            self.assertIn(f"sharetap.py {sub} /root/tap.jsonl --since \"$1\"", end)
        mine = src[src.index("t_mine() {"):src.index("\n}\n", src.index("t_mine() {"))]
        self.assertIn('if [ -n "${TAP:-}" ]; then tap_start || return 1; fi', mine)
        self.assertEqual(mine.count('[ -n "${TAP:-}" ] && tap_end "$t0"'), 2)

    def test_refverify_builds_the_probe_into_the_spec_then_checks_every_share_file(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        body = src[src.index("t_refverify() {"):src.index("\n}\n", src.index("t_refverify() {"))]
        order = ["curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal",
                 'git fetch -q --depth 1 origin "$PEARL_SPEC_SHA"',
                 "cp $R/earn/native/probes/refverify.rs $spec/zk-pow/src/bin/refverify.rs",
                 "cargo build --release --bin refverify",
                 "mapfile -t files < <(find /root/shares -name '*.jsonl' 2>/dev/null | sort)",
                 '$spec/zk-pow/target/release/refverify < "$f" > /root/refverify.log 2>&1 || rc=1']
        at = [body.index(x) for x in order]
        self.assertEqual(at, sorted(at))
        self.assertIn('say "refverify FAIL no shares in /root/shares"; return 1;', body)


class TestCompilersMemoryReadings(unittest.TestCase):
    """compilers.sh (earn/native/probes/turing), which nvtest.sh runs: med.py's and sum.py's memory columns."""
    SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SRC = read(os.path.join(HERE, "..", "..", "native", "probes", "turing", "compilers.sh"))

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


class TestCompilersRace(unittest.TestCase):
    """compilers.sh's race stage: every build as a library, raced in one process by probes/race.cu."""
    SRC = RACE = None

    @classmethod
    def setUpClass(cls):
        native = os.path.join(HERE, "..", "..", "native")
        cls.SRC = read(os.path.join(native, "probes", "turing", "compilers.sh"))
        cls.RACE = read(os.path.join(native, "probes", "race.cu"))

    def test_race_is_not_a_default_stage(self):
        self.assertIn('STAGES=" ${STAGES:-setup build bench hashrate verify complete probe} "', self.SRC)
        self.assertIn("RACE_SECS=${RACE_SECS:-600}; RACE_LEG=${RACE_LEG:-4}; RACE_WARM=${RACE_WARM:-1}; RACE_AA=${RACE_AA:-1}",
                      self.SRC)

    def test_every_build_is_also_a_library_that_keeps_its_own_symbols(self):
        self.assertIn('nvcc -shared -cudart static -Xlinker -Bsymbolic -o "$K/libpearl.so" "$K/cuda-build/pearl_kernel.o"',
                      self.SRC)
        # A build without one still races in the other stages: a warning, not a failure.
        self.assertIn('say "build $name WARN: libpearl.so:', self.SRC)

    def test_the_race_takes_base_then_its_copy_then_the_rest(self):
        start = self.SRC.index("if has race && ")
        block = self.SRC[start:self.SRC.index("\nfi\n", start) + 4]
        out = tempfile.mkdtemp()
        for name in ("base", "flip", "noso"):
            os.makedirs(os.path.join(out, "v", name))
        for name in ("base", "flip"):
            open(os.path.join(out, "v", name, "libpearl.so"), "w").close()
        # Stubs: nvcc writes a race that prints its arguments, and timeout runs what it is given.
        stubs = ('say() { echo "$*"; }; has() { [ "$1" = race ]; }; '
                 'nvcc() { printf "#!/bin/sh\\necho \\"ARGS \\$*\\"\\n" > "$OUT/race"; chmod +x "$OUT/race"; }; '
                 'timeout() { shift 3; "$@"; }; ')
        # BUILT is the list of builds that built, as the script sets it before this stage. Each build's k (kof,
        # kset) and the clock lines (smi_start, off) as the script defines them; every build here is at k 2048.
        defs = ""
        for first, last in (("declare -A KOF=()", "kset() {"), ("smi_start() {", "smi_stop() {")):
            i = self.SRC.rindex("\n", 0, self.SRC.index(first)) + 1
            defs += self.SRC[i:self.SRC.index("\n", self.SRC.index(last, i)) + 1]
        r = subprocess.run(["bash", "-c", stubs + 'VARIANTS=("base|" "flip|-DX" "noso|"); BUILT=(base flip noso)\n'
                            + defs + block],
                           env={**os.environ, "OUT": out.replace("\\", "/"), "NATIVE": "/n", "RACE_SECS": "120",
                                "RACE_LEG": "4", "RACE_WARM": "1", "RACE_AA": "1", "SMI_EVERY": "0"},
                           capture_output=True, text=True)
        self.assertEqual(r.stderr, "")
        lines = r.stdout.splitlines()
        self.assertIn("race: noso has no libpearl.so, left out", lines)
        o = out.replace("\\", "/")
        self.assertIn(f"race ARGS 120 4 1 base={o}/v/base/libpearl.so aa={o}/v/base/libpearl-aa.so "
                      f"flip={o}/v/flip/libpearl.so", lines)
        self.assertTrue(os.path.exists(os.path.join(out, "v", "base", "libpearl-aa.so")))
        with open(os.path.join(out, "race.log")) as f:
            self.assertIn("ARGS 120 4 1 base=", f.read())

    def test_race_cu_prints_the_lines_it_documents(self):
        for line in ("START builds=", "AUDIT %s fold=[%s]", "LEG round=%d %s th=%.3f", "RACE %s vs %s rounds=%zu d=%+.3f%%",
                     "RACE %s reference rounds=", "FAIL %s"):
            self.assertIn(line, self.RACE)
        # The same path twice would load once, and race the build against itself unseen.
        self.assertIn("the same library as", self.RACE)
        # Each build's k, at the end of its AUDIT and RACE lines (name=lib.so:K; 2048 without one).
        self.assertIn("race <seconds> <leg seconds> <warm seconds> base=libA.so name=libB.so[:K]", self.RACE)
        self.assertIn('"AUDIT %s fold=[%s] %s k=%u%s\\n"', self.RACE)
        self.assertEqual(self.RACE.count(' k=%u\\n"'), 2)


class TestCompilersK(unittest.TestCase):
    """compilers.sh with a build past k 2048 (-DPEARL_FOLD_K=K): each stage runs it at its own k, and a k 2048 build's
    commands stay as they were. Also the SM clock and power lines (SMI_EVERY) and the nvcc check (NVCC_VER)."""
    SRC = None

    @classmethod
    def setUpClass(cls):
        native = os.path.join(HERE, "..", "..", "native")
        cls.SRC = read(os.path.join(native, "probes", "turing", "compilers.sh"))
        cls.SWEEP = read(os.path.join(native, "amd", "test", "sweep.sh"))

    def part(self, start, end):
        """The script's text from the line holding `start` to the end of the line holding `end`."""
        i = self.SRC.rindex("\n", 0, self.SRC.index(start)) + 1
        return self.SRC[i:self.SRC.index("\n", self.SRC.index(end, i)) + 1]

    def stage(self, head):
        start = self.SRC.index(head)
        return self.SRC[start:self.SRC.index("\nfi\n", start) + 4]

    def kof(self):
        return self.part("declare -A KOF=()", "kset() {")

    def smi(self):
        return self.part("smi_start() {", "smi_stop() {")

    def tools(self):
        """A directory with python3 (this Python) and a node that logs its k and arguments to $OUT/node.log and prints
        a result line."""
        d = tempfile.mkdtemp()
        py = sys.executable.replace("\\", "/")
        with open(os.path.join(d, "python3"), "w", newline="\n") as f:
            f.write(f'#!/bin/sh\nexec "{py}" "$@"\n')
        with open(os.path.join(d, "node"), "w", newline="\n") as f:
            f.write('#!/bin/sh\necho "${PEARL_PROFILE_K:--}|$*" >> "$OUT/node.log"\n'
                    'echo \'{"th":100.5,"sd":0.1,"mhz":1500,"watts":300,"tempC":60,"hits":400,"verified":400,'
                    '"failed":0,"salts":3,"PASS":true}\'\n')
        for n in ("python3", "node"):
            os.chmod(os.path.join(d, n), 0o755)
        return d

    def bash(self, script, **env):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              env={**os.environ, "SMI_EVERY": "0", **env})

    def test_a_builds_k_comes_from_its_flags_as_sweep_sh_reads_it(self):
        rule = '[[ " $flags " =~ [[:space:]]-DPEARL_FOLD_K=([0-9]+)u?[[:space:]] ]]'
        self.assertIn(rule, self.SRC)
        self.assertIn(rule, self.SWEEP)
        r = self.bash('VARIANTS=("base|" "k4|-DPEARL_FOLD_K=4096" "k8|-DPEARL_TALL_PREGUARD=0 -DPEARL_FOLD_K=8192u -DX=1" '
                      '"k2|-DPEARL_FOLD_K=2048u" "kx|-DPEARL_FOLD_KX=4096" "tail|-DA -DPEARL_FOLD_K=4096|abc123")\n'
                      + self.kof() + 'for n in base k4 k8 k2 kx tail none; do echo "$n $(kof $n) [$(kset $n)]"; done')
        self.assertEqual(r.stdout.splitlines(), ["base 2048 []", "k4 4096 [4096]", "k8 8192 [8192]", "k2 2048 []",
                                                 "kx 2048 []", "tail 4096 [4096]", "none 2048 []"])
        # After ONLY, so it reads the builds that run, and before any stage.
        self.assertLess(self.SRC.index('VARIANTS=("${keep[@]}")'), self.SRC.index("declare -A KOF=()"))
        self.assertLess(self.SRC.index("declare -A KOF=()"), self.SRC.index("if has setup; then"))

    def test_a_build_whose_k_is_not_the_one_read_from_its_flags_fails(self):
        """bench.cu is built with the k kof read as PEARL_BENCH_K, and does not compile if the build mines another k:
        -D PEARL_FOLD_K=4096 (read as 2048) or a second -DPEARL_FOLD_K. nvcc 12.6 refused both (1 November)."""
        bench = read(os.path.join(HERE, "..", "..", "native", "probes", "bench.cu"))
        self.assertIn("#ifdef PEARL_BENCH_K\nstatic_assert(PEARL_FOLD_K == PEARL_BENCH_K,", bench)
        cmd = self.part('nvcc -O3 -std=c++17 -cudart static $GC $flags -DPEARL_BENCH_K=', 'say "build $name FAIL: bench:')
        out = tempfile.mkdtemp().replace("\\", "/")
        r = self.bash('say() { echo "$*"; }; nvcc() { echo "$*" >> "$OUT/nvcc.log"; }; GC=-gc; NATIVE=/n\n'
                      'VARIANTS=("base|" "k4|-DPEARL_FOLD_K=4096" "sp|-D PEARL_FOLD_K=4096")\n' + self.kof()
                      + 'for v in "${VARIANTS[@]}"; do\nIFS="|" read -r name flags _ <<< "$v"; K=$OUT\n' + cmd + 'done',
                      OUT=out)
        self.assertEqual((r.stdout, r.stderr), ("", ""))
        with open(os.path.join(out, "nvcc.log")) as f:
            calls = [l.split(" -I")[0] for l in f.read().replace(out, "O").splitlines()]
        self.assertEqual(calls, ["-O3 -std=c++17 -cudart static -gc -DPEARL_BENCH_K=2048",
                                 "-O3 -std=c++17 -cudart static -gc -DPEARL_FOLD_K=4096 -DPEARL_BENCH_K=4096",
                                 "-O3 -std=c++17 -cudart static -gc -D PEARL_FOLD_K=4096 -DPEARL_BENCH_K=2048"])

    def race_args(self, variants, built):
        out = tempfile.mkdtemp()
        for name in built:
            os.makedirs(os.path.join(out, "v", name))
            open(os.path.join(out, "v", name, "libpearl.so"), "w").close()
        stubs = ('say() { echo "$*"; }; has() { [ "$1" = race ]; }; '
                 'nvcc() { printf "#!/bin/sh\\necho \\"ARGS \\$*\\"\\n" > "$OUT/race"; chmod +x "$OUT/race"; }; '
                 'timeout() { shift 3; "$@"; }; ')
        r = self.bash(stubs + f"VARIANTS=({variants}); BUILT=({' '.join(built)})\n" + self.kof() + self.smi()
                      + self.stage("if has race && "),
                      OUT=out.replace("\\", "/"), NATIVE="/n", RACE_SECS="120", RACE_LEG="8", RACE_WARM="1", RACE_AA="1")
        self.assertEqual(r.stderr, "")
        o = out.replace("\\", "/")
        return [l.replace(o, "O") for l in r.stdout.splitlines() if l.startswith("race ARGS")]

    def test_the_race_names_a_build_past_k_2048_with_its_k(self):
        self.assertEqual(self.race_args('"base|" "k4|-DPEARL_FOLD_K=4096 -DPEARL_TALL_PREGUARD=0" "k2|-DPEARL_FOLD_K=2048u"',
                                        ["base", "k4", "k2"]),
                         ["race ARGS 120 8 1 base=O/v/base/libpearl.so aa=O/v/base/libpearl-aa.so "
                          "k4=O/v/k4/libpearl.so:4096 k2=O/v/k2/libpearl.so"])
        # A base past k 2048 takes its copy with it.
        self.assertEqual(self.race_args('"base|-DPEARL_FOLD_K=4096" "x|"', ["base", "x"]),
                         ["race ARGS 120 8 1 base=O/v/base/libpearl.so:4096 aa=O/v/base/libpearl-aa.so:4096 "
                          "x=O/v/x/libpearl.so"])

    def test_bench_counts_256_x_k_macs_a_region_and_tags_rows_past_k_2048(self):
        th = self.part('th=$(python3 -c "print(round($att * 256 * $(kof "$name")', "th=$(")
        row = self.part('kcol="" k=$(kset "$name")', 'printf \'%s\\t%s\\t')
        out = tempfile.mkdtemp().replace("\\", "/")
        r = self.bash('VARIANTS=("base|" "k4|-DPEARL_FOLD_K=4096")\n' + self.kof()
                      + 'att=1000000000 secs=2 r=1 mhz=1500 w=300 c=60 pg=0 mean=1500 mc=20 mu=9000 mclk=10501 cyc=\n'
                      + 'for name in base k4; do\n' + th + 'echo "$name $th"\n' + row + 'done',
                      PATH=self.tools() + os.pathsep + os.environ["PATH"], OUT=out)
        # The rate as before at k 2048 (2048 written into the formula), and twice the MACs a region at k 4096.
        self.assertEqual(r.stdout.splitlines(), ["base 262.144", "k4 524.288"], r.stderr)
        self.assertEqual(r.stderr, "")
        with open(os.path.join(out, "bench.tsv")) as f:
            rows = [l.rstrip("\n").split("\t") for l in f]
        self.assertEqual([len(x) for x in rows], [12, 13])
        self.assertEqual(rows[0][:3] + rows[0][11:], ["1", "base", "262.144", "-"])
        self.assertEqual(rows[1][:3] + rows[1][11:], ["1", "k4", "524.288", "-", "4096"])

    def test_sum_py_scales_the_share_of_the_peak_by_k(self):
        start = self.SRC.index("cat > \"$OUT/sum.py\" <<'EOF'\n") + len("cat > \"$OUT/sum.py\" <<'EOF'\n")
        d = tempfile.mkdtemp()
        s, t = os.path.join(d, "sum.py"), os.path.join(d, "b.tsv")
        with open(s, "w") as f:
            f.write(self.SRC[start:self.SRC.index("\nEOF\n", start)] + "\n")
        with open(t, "w") as f:   # 48 x k cycles a tile at the peak: 98304 at k 2048, 196608 at k 4096
            f.write("1\tbase\t124.0\t1620\t350\t70\t76.5\t1620\t22\t1366\t9501\t102658\n"
                    "1\tk4\t125.0\t1620\t350\t70\t77.2\t1620\t22\t2400\t9501\t205316\t4096\n")
        out = subprocess.run([sys.executable, "-I", s, t, "bench-sum", os.path.join(d, "best")], capture_output=True,
                             text=True, check=True).stdout.splitlines()
        self.assertIn("cyc 102658 (95.76%)", out[0])
        self.assertIn("cyc 205316 (95.76%) vs base", out[1])

    def test_hashrate_and_verify_run_a_build_past_k_2048_at_its_k(self):
        out = tempfile.mkdtemp().replace("\\", "/")
        src = tempfile.mkdtemp().replace("\\", "/")
        os.makedirs(os.path.join(src, "earn"))
        for name in ("base", "k4"):
            os.makedirs(os.path.join(out, "v", name))
        open(os.path.join(out, "v", "base", "ok"), "w").close()
        start = self.SRC.index("cat > \"$OUT/sum.py\" <<'EOF'\n") + len("cat > \"$OUT/sum.py\" <<'EOF'\n")
        with open(os.path.join(out, "sum.py"), "w") as f:
            f.write(self.SRC[start:self.SRC.index("\nEOF\n", start)] + "\n")
        stubs = ('say() { echo "$*"; }; has() { [ "$1" = hashrate ] || [ "$1" = verify ]; }; '
                 'timeout() { shift 3; "$@"; }; ')
        r = self.bash(stubs + 'VARIANTS=("base|" "k4|-DPEARL_FOLD_K=4096"); BUILT=(base k4)\n' + self.kof() + self.smi()
                      + self.stage("if has hashrate && ") + self.stage("if has verify; then"),
                      PATH=self.tools() + os.pathsep + os.environ["PATH"], OUT=out, SRC=src, HR_BEST="k4",
                      HR_ROUNDS="1", HR_SECS="1", VERIFY_SECS="1")
        with open(os.path.join(out, "node.log")) as f:
            calls = [l.strip().replace(out, "O") for l in f]
        # Base as before (no PEARL_PROFILE_K: the core opens with cardK); k4 pinned to k 4096.
        self.assertEqual(calls, ["-|native/probes/hashrate.js O/v/base/build/Release/pearl_core.node 1 10",
                                 "4096|native/probes/hashrate.js O/v/k4/build/Release/pearl_core.node 1 10",
                                 "4096|native/probes/verify-hits.js O/v/k4/build/Release/pearl_core.node 1"], r.stderr)
        lines = r.stdout.splitlines()
        self.assertTrue(any(l.startswith("hashrate r1 base 100.5 TH/s") and l.endswith(" MHz") for l in lines), lines)
        self.assertTrue(any(l.startswith("hashrate r1 k4 100.5 TH/s") and l.endswith(" MHz k 4096") for l in lines), lines)
        self.assertTrue(any(l.startswith("verify k4 k 4096 ") and '"PASS":true' in l for l in lines), lines)
        self.assertEqual(r.stderr, "")

    def test_complete_compares_hit_lists_only_within_a_k(self):
        out = tempfile.mkdtemp()
        hl = {"base": "aaa bbb", "same": "aaa bbb", "bad": "aaa zzz", "k4": "ccc ddd", "k4same": "ccc ddd",
              "k4diff": "ccc eee"}
        for name, h in hl.items():
            os.makedirs(os.path.join(out, "v", name))
            with open(os.path.join(out, "v", name, "bench"), "w", newline="\n") as f:
                f.write("#!/bin/sh\n" + "".join(f"echo 'HL {i} {x}'\n" for i, x in enumerate(h.split()))
                        + "echo 'hitlist bits=8 regions=64 batches=2 hits=2 full=0 salts=1 digest=0123'\n")
            os.chmod(os.path.join(out, "v", name, "bench"), 0o755)
        open(os.path.join(out, "v", "base", "ok"), "w").close()
        stubs = 'say() { echo "$*"; }; has() { [ "$1" = complete ]; }; timeout() { shift 3; "$@"; }; '
        r = self.bash(stubs + 'VARIANTS=("base|" "same|-DY" "k4|-DPEARL_FOLD_K=4096" "k4same|-DPEARL_FOLD_K=4096 -DZ" '
                      '"k4diff|-DPEARL_FOLD_K=4096u -DW" "bad|-DV"); BUILT=(base same k4 k4same k4diff bad)\n'
                      + self.kof() + self.stage("if has complete && "),
                      OUT=out.replace("\\", "/"), COMPLETE_SHAPES="8192 16 2", COMPLETE_SECS="5")
        s = "bits=8 regions=64 batches=2 hits=2 full=0 salts=1 digest=0123"
        self.assertEqual(r.stdout.splitlines(), [
            f"complete m8192-cb16 base {s}",
            f"complete m8192-cb16 same PASS {s}",
            f"complete m8192-cb16 k4 {s} (the reference for k 4096)",
            f"complete m8192-cb16 k4same PASS against k4 {s}",
            f"complete m8192-cb16 k4diff FAIL against k4 {s}, first difference: < HL 1 ddd > HL 1 eee ",
            f"complete m8192-cb16 bad FAIL {s}, first difference: < HL 1 bbb > HL 1 zzz "], r.stderr)
        self.assertEqual(r.stderr, "")

    def test_complete_leaves_out_the_timing_only_builds(self):
        # A stamped build refuses the hit list's target, as verify's skip says; here its bench would pass, so only
        # the skip keeps it out.
        out = tempfile.mkdtemp()
        for name in ("base", "real", "stamped"):
            os.makedirs(os.path.join(out, "v", name))
            with open(os.path.join(out, "v", name, "bench"), "w", newline="\n") as f:
                f.write("#!/bin/sh\necho 'HL 0 aaa'\n"
                        "echo 'hitlist bits=8 regions=64 batches=1 hits=1 full=0 salts=1 digest=0123'\n")
            os.chmod(os.path.join(out, "v", name, "bench"), 0o755)
        open(os.path.join(out, "v", "base", "ok"), "w").close()
        stubs = 'say() { echo "$*"; }; has() { [ "$1" = complete ]; }; timeout() { shift 3; "$@"; }; '
        r = self.bash(stubs + 'VARIANTS=("base|" "real|-DPEARL_TALL_CYC -DPEARL_TALL_HASH_FMA=1" '
                      '"stamped|-DPEARL_TALL_CYC -DPEARL_TALL_HASHCYC=1"); BUILT=(base real stamped)\n'
                      + self.kof() + self.part("declare -A TIMING=()", "done") + self.stage("if has complete && "),
                      OUT=out.replace("\\", "/"), COMPLETE_SHAPES="8192 16 1", COMPLETE_SECS="5")
        s = "bits=8 regions=64 batches=1 hits=1 full=0 salts=1 digest=0123"
        self.assertEqual(r.stdout.splitlines(), [
            "complete stamped skipped: a timing-only build refuses any target but zero",
            f"complete m8192-cb16 base {s}",
            f"complete m8192-cb16 real PASS {s}"], r.stderr)
        self.assertEqual(r.stderr, "")

    def test_smi_lines_come_every_smi_every_seconds_while_a_stage_runs(self):
        d = tempfile.mkdtemp()
        log = os.path.join(d, "args").replace("\\", "/")
        with open(os.path.join(d, "nvidia-smi"), "w", newline="\n") as f:
            f.write(f'#!/bin/sh\necho "$*" >> "{log}"\necho "1845, 349.12, 350.00"\n')
        os.chmod(os.path.join(d, "nvidia-smi"), 0o755)
        say = self.part("say() {", "say() {")
        script = say + self.smi() + "smi_start race; sleep 1; smi_stop; echo stopped; sleep 0.6; echo end"
        env = {"PATH": d + os.pathsep + os.environ["PATH"], "OUT": d.replace("\\", "/")}
        lines = self.bash(script, SMI_EVERY="0.25", **env).stdout.splitlines()
        cut = lines.index("stopped")
        self.assertGreaterEqual(cut, 2, lines)
        self.assertEqual(set(lines[:cut]), {"[compilers] smi race 1845 MHz 349.12/350.00 W"})
        self.assertEqual(lines[cut:], ["stopped", "end"])
        with open(log) as f:
            self.assertIn("-i 0 --query-gpu=clocks.sm,power.draw,power.limit --format=csv,noheader,nounits", f.read())
        # Off at 0, as compilers.sh is unless nvtest.sh sets it.
        self.assertEqual(self.bash(script, SMI_EVERY="0", **env).stdout.splitlines(), ["stopped", "end"])
        # A query that hangs, and ignores TERM as a stuck driver call can, does not hold up the stop.
        with open(os.path.join(d, "nvidia-smi"), "w", newline="\n") as f:
            f.write("#!/bin/sh\ntrap '' TERM\nsleep 8\n")
        t = time.time()
        r = self.bash(say + self.smi() + "smi_start race; sleep 0.5; smi_stop; echo stopped", SMI_EVERY="0.25", **env)
        self.assertLess(time.time() - t, 4)
        self.assertEqual((r.stdout, r.stderr), ("stopped\n", ""))
        self.assertIn("SMI_EVERY=${SMI_EVERY:-0}", self.SRC)
        # Around the race and the hashrate rounds.
        race, hr = self.stage("if has race && "), self.stage("if has hashrate && ")
        self.assertLess(race.index("smi_start race"), race.index('timeout -k 10 $((RACE_SECS + 600)) "$OUT/race"'))
        self.assertLess(race.index("tee \"$OUT/race.log\""), race.index("smi_stop"))
        self.assertLess(hr.index("smi_start hashrate"), hr.index('for r in $(seq 1 "$HR_ROUNDS"); do'))
        self.assertLess(hr.index("smi_stop"), hr.index('python3 "$OUT/sum.py" "$OUT/hashrate.tsv"'))

    def test_the_nvcc_check_follows_the_toolkit_asked_for(self):
        start = self.SRC.index('if [ "$NVCC_VER" = V12.8.93 ]; then')
        check = self.SRC[start:self.SRC.index("\nfi\n", start) + 4]
        self.assertIn("NVCC_VER=${NVCC_VER:-V12.8.93}", self.SRC)
        for want, nv, line in (("V12.8.93", "V12.8.93", ""),
                               ("V12.8.93", "V12.6.85",
                                "WARN: nvcc V12.6.85 is not CI's 12.8.93; the expected fold hashes will not match"),
                               ("V13.3.73", "V13.3.73", ""),
                               ("V13.3.73", "V12.8.93", "WARN: nvcc V12.8.93 is not the V13.3.73 asked for (NVCC_VER)")):
            r = self.bash('say() { echo "$*"; }\n' + check, NVCC_VER=want, nv=nv)
            self.assertEqual(r.stdout.strip(), line, (want, nv, r.stderr))


def git_has(rev):
    try:
        return subprocess.run(["git", "-C", a.REPO, "rev-parse", "--verify", "-q", rev + "^{commit}"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


class TestToolsMoved(unittest.TestCase):
    """The tools moved from tuning/salad to earn/tuning/salad on 7 November. Boxes and commits from either side
    of the move still have to work together, or fail before they sit on a billed PC."""
    OLD_BOOT = a.BOOT.replace("/earn/tuning/salad/", "/tuning/salad/")

    def boot(self, script, fail_new):
        # BOOT with a fake curl that fails the new path (as GitHub does for a commit from before the move) and
        # serves `script` from the old one, and /root swapped for a temp folder.
        d = tempfile.mkdtemp().replace("\\", "/")
        os.makedirs(os.path.join(d, "bin"))
        with open(os.path.join(d, "bin", "curl"), "w", newline="\n") as f:
            f.write('#!/bin/sh\nfor x; do case "$x" in http*) u="$x";; esac; done\n'
                    'out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
                    f'case "$u" in */earn/tuning/salad/*) [ "{int(fail_new)}" = 1 ] && exit 22;; esac\n'
                    f'echo "echo ran {script} from $u" > "$out"\n')
        os.chmod(os.path.join(d, "bin", "curl"), 0o755)
        boot = (a.BOOT.replace("/root/", d + "/").replace("exec sleep infinity", "exit 0")
                .replace("sleep 20", ":"))
        return subprocess.run(["bash", "-c", boot], capture_output=True, text=True,
                              env={"PATH": d + "/bin:" + os.environ["PATH"], "BASE_SHA": "abc"}).stdout

    def test_boot_fetches_from_the_new_path(self):
        out = self.boot("new", fail_new=False)
        self.assertIn("ran new from https://raw.githubusercontent.com/super3/llmjob/abc/earn/tuning/salad/amdtest.sh",
                      out)

    def test_boot_falls_back_to_the_old_path_for_an_older_commit(self):
        out = self.boot("old", fail_new=True)
        self.assertIn("ran old from https://raw.githubusercontent.com/super3/llmjob/abc/tuning/salad/amdtest.sh", out)
        self.assertNotIn("FAIL fetch", out)

    def test_nv_boot_falls_back_too(self):
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/earn/tuning/salad/nvtest.sh", a.NV_BOOT)
        self.assertIn("${SCRIPT_SHA:-$BASE_SHA}/tuning/salad/nvtest.sh", a.NV_BOOT)

    def test_old_boot_tells_a_box_from_before_the_move(self):
        self.assertTrue(a.old_boot({"container": {"command": ["/bin/sh", "-c", self.OLD_BOOT]}}))
        self.assertFalse(a.old_boot({"container": {"command": ["/bin/sh", "-c", a.BOOT]}}))
        self.assertFalse(a.old_boot({"container": {"command": ["/bin/sh", "-c", a.NV_BOOT]}}))
        self.assertFalse(a.old_boot(None))

    def set_env(self, name, command, cur, sha=None, unset=()):
        calls = []
        g = {"container": {"command": ["/bin/sh", "-c", command], "environment_variables": cur}}
        args = type("A", (), {"name": name, "env": [], "sha": sha, "tasks": None, "unset": list(unset),
                              "patch_from": None, "no_patch": False, "restart": False})()
        old = (a.group, a.moved, a.w.http)
        a.group = lambda n: g
        a.moved = lambda s, repo=None: {"new": True, "old": False}.get(s)
        a.w.http = lambda *x, **k: calls.append(x) or (200, {"container": {"environment_variables": {}}})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a.cmd_set_env(args)
        finally:
            a.group, a.moved, a.w.http = old
        return calls

    def test_set_env_refuses_to_point_an_old_box_at_a_new_commit(self):
        for cur, sha in (({"BASE_SHA": "old"}, "new"), ({"BASE_SHA": "old", "SCRIPT_SHA": "new"}, None),
                         ({"BASE_SHA": "new", "SCRIPT_SHA": "old"}, None)):
            with self.assertRaises(SystemExit) as e:
                self.set_env("amdtest-rdna4-2", self.OLD_BOOT, cur, sha=sha)
            self.assertIn("Delete it and create a new box", str(e.exception))

    def test_set_env_still_runs_an_old_box_on_old_commits_and_any_new_box(self):
        self.assertEqual(len(self.set_env("amdtest-rdna4-2", self.OLD_BOOT, {"BASE_SHA": "x"}, sha="old")), 1)
        self.assertEqual(len(self.set_env("amdtest-rdna4-2", a.BOOT, {"BASE_SHA": "old"}, sha="new")), 1)
        # A commit this checkout doesn't know isn't refused: it may be newer than the checkout.
        self.assertEqual(len(self.set_env("amdtest-rdna4-2", self.OLD_BOOT, {"BASE_SHA": "x"}, sha="unknown")), 1)
        # An unset SCRIPT_SHA no longer counts.
        self.assertEqual(len(self.set_env("amdtest-rdna4-2", self.OLD_BOOT, {"BASE_SHA": "old", "SCRIPT_SHA": "new"},
                                          unset=["SCRIPT_SHA"])), 1)

    def test_an_old_nvtest_box_may_run_a_new_base_with_an_old_script(self):
        # nvtest.sh never hands over to the checkout's copy, so only the script's own commit matters.
        old_nv = a.NV_BOOT.replace("/earn/tuning/salad/", "/tuning/salad/").replace(
            ' || curl -fsSL "$o" -o /root/nvtest.sh', "")
        self.assertEqual(len(self.set_env("nvtest-rtx3090", old_nv, {"BASE_SHA": "new", "SCRIPT_SHA": "old"})), 1)
        with self.assertRaises(SystemExit):
            self.set_env("nvtest-rtx3090", old_nv, {"BASE_SHA": "new"})

    @unittest.skipUnless(git_has("HEAD"), "needs this checkout's git history")
    def test_moved_reads_the_commit(self):
        self.assertTrue(a.moved("HEAD"))
        self.assertIsNone(a.moved("0" * 40))
        if git_has("v0.6.13"):          # the last release before the move
            self.assertFalse(a.moved("v0.6.13"))

    def test_script_finds_its_folder_in_either_layout(self):
        sh = os.path.join(HERE, "amdtest.sh").replace("\\", "/")
        for sub in ("earn/tuning/salad", "tuning/salad"):
            root = tempfile.mkdtemp().replace("\\", "/")
            os.makedirs(os.path.join(root, *sub.split("/")))
            r = subprocess.run(["bash", "-c", f"AMDTEST_STAGE=lib . {sh}; tdir"], capture_output=True, text=True,
                               env={"PATH": os.environ["PATH"], "BASE_SHA": "abc", "AMDTEST_ROOT": root})
            self.assertEqual(r.stdout.strip(), f"{root}/{sub}")
        # Every use goes through tdir: no line but its own names either folder.
        rest = [l for l in read(os.path.join(HERE, "amdtest.sh")).splitlines() if not l.startswith("tdir() {")]
        self.assertEqual([l for l in rest if "$R/earn/tuning/salad" in l or "$R/tuning/salad" in l], [])


if __name__ == "__main__":
    unittest.main()
