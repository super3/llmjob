'use strict';

// The AMD core's C as the preprocessor leaves it for given values of its timing switches, e.g.
// { PEARL_AMD_ABLATE: 8, PEARL_AMD_STAMP: 1 }. No HIP compiler runs in these tests, so they read the C
// as text. Every #if chain whose condition names only those switches is resolved; every other line,
// directives too, is kept. An #elif in such a chain that names none of them, reached with nothing
// taken, becomes the #if it would be without the switches. A condition that names a switch and
// anything else is kept as it is, with its chain.
function preprocess(text, values) {
  const names = Object.keys(values);
  const nameRe = () => new RegExp('\\b(' + names.join('|') + ')\\b', 'g');
  const mentions = (c) => nameRe().test(c);
  const substituted = (c) => c.replace(nameRe(), (n) => String(values[n]));
  // Only the switches, numbers and operators: nothing else for the preprocessor to look up.
  const onlySwitches = (c) => !/[A-Za-z_]/.test(substituted(c).replace(/\b(0x[0-9a-fA-F]+|\d+)u?\b/g, ''));
  const test = (c) => Number(new Function('return (' + substituted(c).replace(/\b(0x[0-9a-fA-F]+|\d+)u\b/g, '$1')
    + ');')()) !== 0;
  const out = [];
  const stack = [];   // per open #if: { ours, taken, on }
  const live = () => stack.every((f) => !f.ours || f.on);
  for (const l of text.split('\n')) {
    let m;
    if ((m = /^#if (.*)$/.exec(l)) && mentions(m[1]) && onlySwitches(m[1])) {
      const on = test(m[1]);
      stack.push({ ours: true, taken: on, on });
    } else if (/^#if(n?def)?\b/.test(l)) {
      stack.push({ ours: false });
      if (live()) out.push(l);
    } else if ((m = /^#elif (.*)$/.exec(l))) {
      const f = stack[stack.length - 1];
      if (!f.ours) {
        if (live()) out.push(l);
      } else if (mentions(m[1])) {
        if (!onlySwitches(m[1])) throw new Error('an #elif mixes a switch with other names: ' + l);
        f.on = !f.taken && test(m[1]);
        f.taken = f.taken || f.on;
      } else if (!f.taken) {
        Object.assign(f, { ours: false });
        if (live()) out.push('#if ' + m[1]);
      } else {
        f.on = false;
      }
    } else if (/^#else\b/.test(l)) {
      const f = stack[stack.length - 1];
      if (!f.ours) {
        if (live()) out.push(l);
      } else {
        f.on = !f.taken;
        f.taken = true;
      }
    } else if (/^#endif\b/.test(l)) {
      if (!stack.pop().ours && live()) out.push(l);
    } else if (live()) {
      out.push(l);
    }
  }
  if (stack.length) throw new Error('unbalanced #if');
  return out.join('\n');
}

module.exports = { preprocess };
