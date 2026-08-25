# vendor/native

Where the packaged installer picks up `pearl_core.node`, our CUDA mining core.

CI stages it here from the `native-core` workflow's artifact (see
`.github/workflows/miner-build.yml`), and electron-builder copies it to
`resources/native/`, which is the first place `src/main/pearlCore.js` looks.

Nothing to check in: the file is a compiled addon and is built per platform. A
local dev build is found automatically at `earn/native/build/Release/`, so this
directory stays empty outside CI.

An installer built without it produces an app that starts, reports that the
Pearl core is not built, and mines nothing — there is no second engine to fall
back to.
