'use strict';

// The Microsoft Visual C++ runtime DLLs llama.cpp's Windows build links
// against. llama.cpp release zips don't include them, and machines without the
// VC++ redistributable can't load llama-server at all — it exits instantly with
// STATUS_DLL_NOT_FOUND (0xC0000135). Worse, a machine can have the DLLs only in
// some directory on the *user* PATH (this is how it usually goes unnoticed):
// normal launches work, but the app relaunched by the elevated updater inherits
// the machine-only PATH and the LLM silently dies on every start. So the app
// ships the three DLLs (build.extraResources → <resources>/llm-runtime) and
// copies them next to llama-server.exe, where the Windows loader always finds
// them first — no PATH involved.
const VC_RUNTIME_DLLS = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'];

// The bundled runtime directory inside a packaged app, or null in a dev run
// (no resourcesPath) — callers then skip the copy; dev machines launch from a
// shell whose PATH already resolves the runtime.
function bundledRuntimeDir(resourcesPath, joinFn) {
  if (!resourcesPath) return null;
  return joinFn(resourcesPath, 'llm-runtime');
}

// Which of the runtime DLLs are missing next to the server binary. Windows-only
// concern — everywhere else the answer is "none".
function missingRuntimeDlls(platform, dir, existsFn, joinFn) {
  if (platform !== 'win32') return [];
  return VC_RUNTIME_DLLS.filter((name) => !existsFn(joinFn(dir, name)));
}

// The copy plan: [{ from, to }] for each missing DLL that the bundle actually
// has. Pure — the caller does the IO — so every branch is unit-testable.
function runtimeCopyPlan({ platform, binDir, resourcesPath, existsFn, joinFn }) {
  const srcDir = bundledRuntimeDir(resourcesPath, joinFn);
  if (!srcDir) return [];
  return missingRuntimeDlls(platform, binDir, existsFn, joinFn)
    .map((name) => ({ from: joinFn(srcDir, name), to: joinFn(binDir, name) }))
    .filter((c) => existsFn(c.from));
}

module.exports = { VC_RUNTIME_DLLS, bundledRuntimeDir, missingRuntimeDlls, runtimeCopyPlan };
