"""Run CuAssembler (github.com/cloudcores/CuAssembler) from an explicit path.

usage:
  python3 -I cuasm_drv.py <CuAssembler dir> <nvdisasm> d <in.cubin> <out.cuasm>
  python3 -I cuasm_drv.py <CuAssembler dir> <nvdisasm> a <in.cuasm> <out.cubin> [learn.sass ...]

d disassembles a cubin to CuAssembler's text; a assembles it back. In a mode
the default sm_75 instruction repository is first extended with every
instruction in the learn.sass files (cuobjdump -sass of nvcc-built cubins), so
any instruction form ptxas emitted can be assembled.

Three changes to CuAssembler's behaviour. The first two make an nvcc cubin
come back as ptxas made it (checked on the shipped fold's sm_75 cubin: every
kernel's code and attributes byte-identical, only the order of two
debug-frame relocations differs); the third makes a replaced kernel's
attributes follow its new code:
  - It takes a kernel's barrier count as 4 bits and refuses 16, which ptxas
    writes when a barrier is named by a register (the fold's named barriers).
    The field is wider: 16 is 0x1000000 in the section flags.
  - It adds EIATTR_WMMA_USED to any kernel with an IMMA or HMMA. ptxas 12.8
    does not write it for mma.sync kernels.
  - It keeps a kernel's old EIATTR_EXIT_INSTR_OFFSETS when the EXIT moves.
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
from CuAsm.config import Config  # noqa: E402

Config.NVDISASM_PATH = sys.argv[2]
from CuAsm.CubinFile import CubinFile  # noqa: E402
from CuAsm.CuAsmParser import CuAsmParser, CuAsmSection  # noqa: E402
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos  # noqa: E402
from CuAsm.CuInsFeeder import CuInsFeeder  # noqa: E402
from CuAsm.CuKernelAssembler import CuKernelAssembler  # noqa: E402

_orig_update = CuAsmSection.updateResourceInfo


def _update(self):
    n = 0
    for f in self.flags:
        m = re.match(r'@"SHF_BARRIERS=(\d+)"', f)
        if m:
            n = int(m.group(1))
    if n != 16:
        return _orig_update(self)
    self.flags = [f for f in self.flags if not f.startswith('@"SHF_BARRIERS=')]
    _orig_update(self)
    self.header['flags'] = (self.header['flags'] & 0xfe0fffff) | (16 << 20)
    self.extra['barnum'] = 16


CuAsmSection.updateResourceInfo = _update

_orig_kinit = CuKernelAssembler.__init__


def _kinit(self, *a, **k):
    _orig_kinit(self, *a, **k)
    for op in ('DMMA', 'HMMA', 'IMMA', 'BMMA'):
        self.AutoAttrOpcodeCallback.pop(op, None)


CuKernelAssembler.__init__ = _kinit

# CuAssembler regenerates EIATTR_EXIT_INSTR_OFFSETS from the EXITs it
# assembles, then writes the old value back (updateNVInfoFromDict appends the
# attribute's previous value, not the new one), so a kernel whose code moved
# its EXIT kept ptxas's offset. Write the new one.
from CuAsm.CuNVInfo import CuNVInfo  # noqa: E402

_orig_upd = CuNVInfo.updateNVInfoFromDict


def _upd(self, nvinfo_dict):
    _orig_upd(self, nvinfo_dict)
    self.m_AttrList = [(a, nvinfo_dict[a]) if a in nvinfo_dict and a.endswith('_INSTR_OFFSETS')
                       else (a, v) for a, v in self.m_AttrList]


CuNVInfo.updateNVInfoFromDict = _upd

mode, src, dst = sys.argv[3], sys.argv[4], sys.argv[5]
if mode == 'd':
    CubinFile(src).saveAsCuAsm(dst)
else:
    learn = sys.argv[6:]
    if learn:
        repos = CuInsAssemblerRepos.getDefaultRepos('sm_75')
        for f in learn:
            repos.update(CuInsFeeder(f, archfilter='sm_75'))
        fd, merged = tempfile.mkstemp(suffix='.sm_75.txt')
        os.close(fd)
        repos.save2file(merged)
        Config.getDefaultInsAsmReposFile = staticmethod(lambda v: merged)
    p = CuAsmParser()
    p.parse(src)
    p.saveAsCubin(dst)
