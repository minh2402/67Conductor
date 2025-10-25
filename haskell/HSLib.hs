{-# LANGUAGE ForeignFunctionInterface #-}
module HSLib where

import Foreign.C.Types
import Foreign.C.String (CString, peekCString, newCString)
import Foreign.Marshal.Alloc (free)

-- Exported example: add two integers
foreign export ccall hs_add :: CInt -> CInt -> IO CInt

-- Exported example: reverse a UTF-8 C string and return a newly allocated C string
foreign export ccall hs_reverse :: CString -> IO CString

-- Free a CString allocated by Haskell (newCString uses C's malloc under the hood)
foreign export ccall hs_free :: CString -> IO ()

hs_add :: CInt -> CInt -> IO CInt
hs_add a b = return (a + b)

hs_reverse :: CString -> IO CString
hs_reverse cstr = do
  s <- peekCString cstr
  newCString (reverse s)

hs_free :: CString -> IO ()
hs_free ptr = free ptr
