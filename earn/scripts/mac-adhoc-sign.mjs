'use strict';

// electron-builder `afterPack` hook: ad-hoc code-sign the macOS app bundle.
//
// Why this exists at all. Apple silicon will not exec a Mach-O binary that
// carries no code signature — the kernel kills it outright, before any Gatekeeper
// prompt, and there is nothing the user can do about it (`xattr -d
// com.apple.quarantine` clears quarantine, not the missing signature). Electron's
// own published binaries ARE signed, but repackaging invalidates that: renaming
// the executable, rewriting Info.plist and adding app.asar all change bytes the
// signature covers. So a Mac build that is not re-signed is a Mac build that
// cannot launch on the majority of Macs.
//
// electron-builder will not do it for us. Its MacPackager looks up a real
// identity in the keychain, and when it finds none it logs "skipped macOS
// application code signing" and moves on (app-builder-lib/out/macPackager.js) —
// there is no ad-hoc fallback. With `mac.identity: null` we tell it so
// explicitly, and sign here instead, in the window after the bundle is fully
// assembled (app.asar + extraResources are in place) and before the DMG is built
// around it.
//
// What this does NOT buy: an ad-hoc signature is not an Apple Developer ID, so
// the app is still unnotarized. Gatekeeper will refuse the first launch and the
// user has to allow it once (System Settings › Privacy & Security › "Open
// Anyway", or `xattr -dr com.apple.quarantine`). That is documented in
// earn/README.md. It is also why the app disables its in-app updater on macOS:
// Squirrel.Mac requires the update's signature to match the running app's, which
// ad-hoc signatures cannot satisfy (see src/shared/platform.js).

import { execFileSync } from 'node:child_process';
import path from 'node:path';

export default async function afterPack(context) {
  // The hook is global — it also fires for the Windows and Linux jobs, where
  // there is nothing to sign and no `codesign` to do it with.
  if (context.electronPlatformName !== 'darwin') return;

  const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);

  // --deep signs nested code (the helper apps, the frameworks, every dylib)
  // inside-out, which is what a bundle needs; --force replaces the signature
  // Electron shipped and our repack invalidated. `-` is the ad-hoc identity.
  // Deliberately not wrapped in a try/catch: a build that reaches the DMG stage
  // unsigned produces an installer nobody can open, and discovering that from a
  // user beats discovering it from a red CI job by several days.
  execFileSync('codesign', ['--force', '--deep', '--sign', '-', app], { stdio: 'inherit' });
  // Assert it took, rather than trusting the exit code above — the same reason
  // the release workflow re-reads the release it just wrote. --deep here too:
  // an unsigned helper app is as fatal as an unsigned main binary on Apple
  // silicon (the renderer process is killed and the window comes up blank), and
  // the top-level signature alone would not notice.
  execFileSync('codesign', ['--verify', '--deep', '--strict', app], { stdio: 'inherit' });

  // appOutDir already names the arch it built (mac-arm64 / mac-x64-…), so log
  // that rather than decoding electron-builder's numeric Arch enum.
  console.log(`  • ad-hoc signed ${path.basename(app)} in ${context.appOutDir}`);
}
