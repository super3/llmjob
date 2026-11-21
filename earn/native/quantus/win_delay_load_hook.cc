// Windows only, linked into quantus_sycl.node by build-sycl-win.sh. The addon
// calls N-API through node.lib, which names node.exe, and the link delay-loads
// node.exe (/DELAYLOAD:node.exe). This hook answers that load with the running
// program itself, so the addon works in node.exe, in the Electron app and in a
// Node with another name. node-gyp links the same hook into every addon it
// builds (its src/win_delay_load_hook.cc).
#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <delayimp.h>
#include <string.h>

static FARPROC WINAPI quantus_load_exe_hook(unsigned int event, DelayLoadInfo *info) {
  if (event != dliNotePreLoadLibrary) return NULL;
  if (_stricmp(info->szDll, "node.exe") != 0) return NULL;
  // A Node built as a DLL keeps N-API in libnode.dll; otherwise it is the program.
  HMODULE m = GetModuleHandleA("libnode.dll");
  if (m == NULL) m = GetModuleHandleA(NULL);
  return (FARPROC)m;
}

decltype(__pfnDliNotifyHook2) __pfnDliNotifyHook2 = quantus_load_exe_hook;
#endif
