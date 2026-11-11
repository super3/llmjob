// rot-path-check.cc: does the host find the rotation fold's cubin beside the addon, on Linux, however the
// addon was loaded? It tests src/pearl_rot_path.h, the lookup pearl_host_rot.inc uses, with a stand-in addon:
// this same file built as a shared object named pearl_core.node. No GPU and no CUDA.
//
//   g++ -std=c++17 -fPIC -shared -DROT_PATH_ADDON -I src probes/rot-path-check.cc -o <dir>/pearl_core.node
//   g++ -std=c++17 -I src probes/rot-path-check.cc -o <dir>/rot-path-check -ldl
//   <dir>/rot-path-check <dir>
//
// CI runs it in the Linux 12.8 job (native-core.yml, "Check where the rotation fold's cubin is looked for").
// Exit 0 when every case passes; each case prints one line.

#include "pearl_rot_path.h"

#ifdef ROT_PATH_ADDON

// The stand-in addon: where does the cubin for the module holding this function live?
static void here() {}
extern "C" int rot_path_probe(char *buf, size_t len) {
  return rot_cubin_path_for(reinterpret_cast<const void *>(&here), buf, len) ? 1 : 0;
}

#else

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

static int failures = 0;

static void expect(const char *what, bool ok, const std::string &got) {
  printf("%s %s: %s\n", ok ? "ok  " : "FAIL", what, got.c_str());
  if (!ok) failures++;
}

// Load `load_path`, ask the addon where its cubin is, unload. `after` runs between the load and the question.
static bool probe(const std::string &load_path, std::string *out, void (*after)() = nullptr) {
  void *h = dlopen(load_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    *out = std::string("dlopen failed: ") + dlerror();
    return false;
  }
  auto fn = reinterpret_cast<int (*)(char *, size_t)>(dlsym(h, "rot_path_probe"));
  char buf[1024] = {0};
  if (after) after();
  const bool found = fn && fn(buf, sizeof buf) == 1;
  *out = buf;
  dlclose(h);
  return found;
}

static void touch(const fs::path &p) { std::ofstream(p) << "cubin"; }
static bool same(const std::string &a, const fs::path &b) {
  std::error_code ec;
  return !a.empty() && fs::exists(a, ec) && fs::equivalent(a, b, ec);
}

int main(int argc, char **argv) {
  // rot-path-check hash <cubin>: the hash the host holds a cubin to, for comparing with asm89.py's.
  if (argc == 3 && strcmp(argv[1], "hash") == 0) {
    std::ifstream in(argv[2], std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    char hex[65] = {0};
    if (!in.good() && bytes.empty()) {
      fprintf(stderr, "cannot read %s\n", argv[2]);
      return 1;
    }
    if (!rot_fold_hash(reinterpret_cast<uint8_t *>(&bytes[0]), bytes.size(), hex)) {
      fprintf(stderr, "%s is not a whole ELF file\n", argv[2]);
      return 1;
    }
    printf("%s\n", hex);
    return 0;
  }
  if (argc != 2) {
    fprintf(stderr, "usage: rot-path-check <dir holding pearl_core.node> | rot-path-check hash <cubin>\n");
    return 2;
  }
  const fs::path dir = fs::absolute(argv[1]);
  const fs::path addon = dir / "pearl_core.node";
  unsetenv("PEARL_ROT_CUBIN");
  std::string got;

  // Each case has its own copy of the addon: the loader keeps a loaded file's first name.
  auto stage = [&](const char *name) {
    const fs::path d = dir / name;
    fs::remove_all(d);
    fs::create_directories(d);
    fs::copy_file(addon, d / "pearl_core.node");
    return d;
  };

  // 1. A full path, the way every package loads it: the CLI and HiveOS beside the binary, the AppImage on
  //    its mount. A folder with a space in its name, as an AppImage's mount or a user's folder may have.
  {
    const fs::path d = stage("full path");
    bool found = probe((d / "pearl_core.node").string(), &got);
    expect("full path, no cubin: not found", !found, got);
    expect("full path, no cubin: names the place it looked", got == (d / PEARL_ROT_FILE).string(), got);
    touch(d / PEARL_ROT_FILE);
    found = probe((d / "pearl_core.node").string(), &got);
    expect("full path: found beside the addon", found && same(got, d / PEARL_ROT_FILE), got);
  }
  // 2. A relative path, and the process has moved to another folder since.
  {
    const fs::path d = stage("relative");
    touch(d / PEARL_ROT_FILE);
    fs::current_path(dir);
    const bool found = probe("./relative/pearl_core.node", &got, [] { if (chdir("/") != 0) perror("chdir"); });
    expect("relative path, folder changed: found", found && same(got, d / PEARL_ROT_FILE), got);
    fs::current_path(dir);
  }
  // 3. The addon is a symbolic link: the cubin beside the link wins, else the one beside the real file.
  {
    const fs::path real = stage("real"), link = dir / "link";
    fs::remove_all(link);
    fs::create_directories(link);
    fs::create_symlink(real / "pearl_core.node", link / "pearl_core.node");
    touch(real / PEARL_ROT_FILE);
    bool found = probe((link / "pearl_core.node").string(), &got);
    expect("symbolic link, cubin beside the real file: found", found && same(got, real / PEARL_ROT_FILE), got);
    touch(link / PEARL_ROT_FILE);
    found = probe((link / "pearl_core.node").string(), &got);
    expect("symbolic link, cubin beside the link: that one", found && got == (link / PEARL_ROT_FILE).string(), got);
  }
  // 4. PEARL_ROT_CUBIN names the file, and nothing else is tried.
  {
    const fs::path d = stage("named");
    touch(d / PEARL_ROT_FILE);
    const fs::path other = dir / "other.cubin";
    touch(other);
    setenv("PEARL_ROT_CUBIN", other.c_str(), 1);
    bool found = probe((d / "pearl_core.node").string(), &got);
    expect("PEARL_ROT_CUBIN: the named file", found && got == other.string(), got);
    setenv("PEARL_ROT_CUBIN", (dir / "missing.cubin").c_str(), 1);
    found = probe((d / "pearl_core.node").string(), &got);
    expect("PEARL_ROT_CUBIN names no file: not found, not the one beside the addon", !found, got);
    unsetenv("PEARL_ROT_CUBIN");
  }
  printf("%s\n", failures ? "rot-path-check: FAILED" : "rot-path-check: every case passed");
  return failures ? 1 : 0;
}

#endif
