'use strict';

// The Windows AMD core's targets (build-amd.sh WINDOWS_ARCHS). AMD's Windows driver lists a Ryzen's
// built-in graphics as a HIP card, often device 0, and when HIP meets a card the build has no code
// for it loads none for the cards after it. So the build carries code for the built-in graphics
// too, or a Radeon in a Ryzen PC can't mine (an RX 9070 XT beside a Ryzen 9000, 13 November).

const fs = require('fs');
const path = require('path');

const BUILD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'build-amd.sh'), 'utf8')
  .replace(/\r\n/g, '\n');
const FOLD = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_amd_fold.h'), 'utf8');
const HOST = fs.readFileSync(path.join(__dirname, '..', 'native', 'amd', 'src', 'pearl_host_hip.cpp'), 'utf8')
  .replace(/\r\n/g, '\n');

const windowsArchs = () => /^WINDOWS_ARCHS="(.*)"$/m.exec(BUILD)[1].split(' ');

describe('Windows AMD targets', () => {
  it('include the Ryzen built-in graphics AMD\'s driver lists beside a Radeon', () => {
    // gfx1035: Ryzen 6000 laptops. gfx1036: Ryzen 7000 and 9000 desktops. gfx1103: Ryzen
    // 7040/8040 laptops and 8000G. gfx1150/gfx1151: Ryzen AI 300 and Max.
    expect(windowsArchs()).toEqual(expect.arrayContaining(['gfx1035', 'gfx1036', 'gfx1103', 'gfx1150', 'gfx1151']));
  });

  it('count GPU indices over the discrete cards on Windows, skipping built-in graphics', () => {
    // planAmdGpus plans GPU 0..n-1 from the OS's n discrete Radeons; HIP also lists the Ryzen's
    // graphics, often as device 0. The Windows host maps an index to the n-th non-integrated HIP
    // device, and its own pick (no index) skips integrated devices too.
    const select = HOST.slice(HOST.indexOf('extern "C" int pearl_host_select_device'),
      HOST.indexOf('static bool k8192_by_default'));
    const win = select.slice(select.indexOf('#ifdef PEARL_WIN_HOST'), select.indexOf('#endif'));
    expect(win).toMatch(/prop\.integrated\) continue;/);
    expect(win).toMatch(/if \(discrete == requested\) mapped = d;/);
    expect(win).toMatch(/requested = mapped;/);
    expect(select).toMatch(/#ifdef PEARL_WIN_HOST\s+if \(prop\.integrated\) continue;/);
  });

  it('give the RDNA 2 built-in graphics a dot4 body, as the fold check needs', () => {
    // build-amd.sh check fails a target whose fold (dot4 for gfx103x) has no body.
    for (const a of ['gfx1035', 'gfx1036']) expect(FOLD).toContain(`defined(__${a}__)`);
  });
});
