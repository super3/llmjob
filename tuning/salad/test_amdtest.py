# Tests for amdtest.py's pure parts: names, classes, the create and update bodies, time parsing, log lines and
# the patch it sends, with no Salad calls. Also checks amdtest.sh's syntax and the task line parsing.
# Run: python3 -m unittest discover -s tuning/salad
import base64, contextlib, gzip, importlib.util, io, os, shutil, subprocess, tempfile, unittest

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

    def test_mine_runs_the_cli_through_mine_cli(self):
        src = read(os.path.join(HERE, "amdtest.sh"))
        body = src[src.index("mine_cli() {"):src.index("\n}\n", src.index("mine_cli() {"))]
        self.assertIn("PEARL_CORE_VARIANT=amd PEARL_CORE_PATH=$A/build/pearl_core_hip.node NODE_PATH=$NM/node_modules", body)
        self.assertIn('--address "$WALLET" --mode mining --pool "$POOL" -w "$WORKER"', body)
        self.assertIn("(mine_cli) > /root/mine.log 2>&1 &", src)


if __name__ == "__main__":
    unittest.main()
