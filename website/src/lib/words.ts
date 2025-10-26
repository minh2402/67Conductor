export default function name123(params: string[] = ["word1", "word2"]) {
  return JSON.stringify({
      words: params,
    })
}