{-# LANGUAGE OverloadedStrings #-}
-- udp_gesture.hs
-- Minimal UDP receiver that expects JSON frames from Python and notifies back when pattern matched.

import qualified Data.ByteString.Char8 as B
import qualified Data.ByteString.Lazy.Char8 as BL
import Network.Socket
import Network.BSD
import Control.Monad (forever, when)
import Data.Aeson
import Data.Time.Clock.POSIX (getPOSIXTime)
import Control.Concurrent (forkIO, threadDelay)
import Data.Text (Text, unpack)
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromMaybe)
import Text.Regex.TDFA ((=~))

-- change these to match your env or use CLI/env vars
pythonHost :: String
pythonHost = "127.0.0.1"

pythonNotifyPort :: Int
pythonNotifyPort = 5006

listenPort :: Int
listenPort = 5005

-- decode JSON frames like: {"ts":..., "landmarks": {"11":{"x":0.1,"y":0.2,"z":0.0}, ... } }
data LandmarkPoint = LandmarkPoint { lx :: Double, ly :: Double, lz :: Double } deriving Show

instance FromJSON LandmarkPoint where
  parseJSON = withObject "landmark" $ \o ->
    LandmarkPoint <$> o .: "x" <*> o .: "y" <*> o .: "z"

-- Very small helper: extract a landmark index from the JSON object
lookupIdx :: Object -> String -> Maybe LandmarkPoint
lookupIdx obj key = case HM.lookup (fromString key) obj of
  Just val -> case fromJSON val of
    Success p -> Just p
    _ -> Nothing
  _ -> Nothing

fromString :: String -> Data.Text.Text
fromString = Data.Text.pack

main :: IO ()
main = withSocketsDo $ do
  addrinfos <- getAddrInfo (Just (defaultHints { addrFlags = [AI_PASSIVE] })) Nothing (show listenPort)
  let serveraddr = head addrinfos
  sock <- socket (addrFamily serveraddr) Datagram defaultProtocol
  bind sock (addrAddress serveraddr)
  putStrLn $ "Listening for frames on UDP port " ++ show listenPort
  seqVar <- return "" -- simple string buffer for 'A'/'B' sequence
  loop sock seqVar
  where
    loop sock seqStr = forever $ do
      (msg, _addr) <- recvFrom sock 4096
      -- parse JSON
      let mobj = decode (BL.fromStrict msg) :: Maybe ObjectWrapper
      case mobj of
        Nothing -> putStrLn "bad json"
        Just obj -> do
          let lms = landmarks obj
          -- compute A/B using shoulder/wrist y (same indices as Python): 11,12, 15,16
          let ma = classifyAB lms
          case ma of
            Nothing -> do
              -- you may want to reset sequence or manage stable-frame detection here
              return ()
            Just code -> do
              let newSeq = seqStr ++ [code]
              putStrLn $ \"Got code: \" ++ [code] ++ \" seq=\" ++ newSeq
              -- run regex
              let regex = \"^(?:AB){2,}(?:A)?$|^(?:BA){2,}(?:B)?$\" :: String
              when (newSeq =~ regex) $ do
                putStrLn $ \"Pattern matched: \" ++ newSeq
                -- send notify to Python
                sendNotify pythonHost pythonNotifyPort (\"PATTERN:\" ++ newSeq)
              -- choose how to manage seqStr (e.g., keep last N or reset). This is simplified.
              return ()
-- Note: You must implement JSON wrapper types and functions here; the above is pseudo-Haskell
-- to show the architecture and where to run the regex. Use aeson to parse JSON and network to send UDP.