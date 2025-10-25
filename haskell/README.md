# Haskell <-> Python FFI example

This small example shows how to export Haskell functions as C-callable symbols and call them from Python using `ctypes`.

Files:

- `HSLib.hs` — Haskell source that exports `hs_add`, `hs_reverse`, and `hs_free` via `foreign export ccall`.
- `test_call.py` — Python script that loads the shared library and calls the functions.

Requirements
- GHC (the Glasgow Haskell Compiler) installed.
- Python 3.

Build

Windows (PowerShell):

1. Open a PowerShell where `ghc` is on PATH.
2. From this directory run:

```powershell
cd ffi_hs_py
ghc -shared -o HSLib.dll HSLib.hs
```

Notes: GHC on Windows will produce `HSLib.dll` and possibly other dependent DLLs. Make sure the folder containing those DLLs is on `PATH` or place them alongside `HSLib.dll`.

Linux/macOS (bash):

```bash
cd ffi_hs_py
ghc -dynamic -shared -fPIC -o libHSLib.so HSLib.hs
```

Run the Python test

After building the shared library run:

```powershell
python .\ffi_hs_py\test_call.py
```

or on Unix:

```bash
python3 ./ffi_hs_py/test_call.py
```

Memory management
- `hs_reverse` returns a newly allocated C string (via `newCString`). The Python side copies the bytes with `ctypes.string_at` and then calls `hs_free` to release the memory.
- Always make sure you free Haskell-allocated memory with the appropriate function exported from Haskell (or ensure both sides agree on allocation/free functions).

Troubleshooting
- If Python cannot load the DLL/so, make sure the library is named correctly and all dependent runtime DLLs are on PATH (Windows) or `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` (Linux/macOS).
- If you run into GHC runtime errors, ensure GHC's runtime DLLs are available to the Python process.

Next steps / extensions
- Export more complex data (structs) — require C structs and marshalling.
- Use `cffi` instead of `ctypes` for a richer API.
- Create a Cabal/Stack packaging and build a stable shared library with proper flags.
