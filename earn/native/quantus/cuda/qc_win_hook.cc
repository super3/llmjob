// Windows only. The addon links N-API through node.lib, which names node.exe,
// and the link delay-loads it (/DELAYLOAD:node.exe). This hook answers that
// load with the running program itself, so the addon works in Node and in the
// Electron app, whatever the executable is called. node-gyp links the same kind
// of hook into every addon it builds.
#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <delayimp.h>
#include <string.h>

static FARPROC WINAPI qc_load_exe_hook(unsigned int event, DelayLoadInfo *info) {
  if (event != dliNotePreLoadLibrary) return NULL;
  if (_stricmp(info->szDll, "node.exe") != 0) return NULL;
  return (FARPROC)GetModuleHandle(NULL);
}

decltype(__pfnDliNotifyHook2) __pfnDliNotifyHook2 = qc_load_exe_hook;
#endif
