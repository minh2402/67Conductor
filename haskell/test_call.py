import sys
import os
import ctypes
from ctypes import c_int, c_char_p

HERE = os.path.dirname(__file__)
LIB_NAME = "HSLib"

if sys.platform == "win32":
    libfile = os.path.join(HERE, LIB_NAME + ".dll")
else:
    libfile = os.path.join(HERE, "lib" + LIB_NAME + ".so")

if not os.path.exists(libfile):
    raise FileNotFoundError(f"Shared library not found: {libfile}\nBuild it first (see README.md)")

lib = ctypes.CDLL(libfile)

# Signatures
lib.hs_add.argtypes = (c_int, c_int)
lib.hs_add.restype = c_int

lib.hs_reverse.argtypes = (c_char_p,)
lib.hs_reverse.restype = c_char_p

lib.hs_free.argtypes = (c_char_p,)
lib.hs_free.restype = None

def add(a, b):
    return lib.hs_add(a, b)

def reverse_str(s: str) -> str:
    b = s.encode('utf-8')
    res = lib.hs_reverse(b)
    if not res:
        return ''
    # copy bytes into Python-managed memory
    py = ctypes.string_at(res).decode('utf-8')
    # free Haskell-allocated string
    lib.hs_free(res)
    return py

if __name__ == '__main__':
    print('Testing Haskell FFI')
    print('1 + 2 =', add(1, 2))
    s = 'Hello 67Conductor!'
    print('reverse("%s") = "%s"' % (s, reverse_str(s)))
