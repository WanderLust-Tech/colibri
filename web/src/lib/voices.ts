/* The saved voices of the cloning speech models (docs/qwen3-tts.md): the name rule the router applies, and a file as the base64 it takes. */

const NAME = /^[a-z0-9][a-z0-9_-]{0,63}$/

/* null when the name is acceptable, "empty" while nothing is typed, "invalid" when the router would refuse it. */
export function voiceNameProblem(name: string): "empty" | "invalid" | null {
  const trimmed = name.trim().toLowerCase()
  if (!trimmed) return "empty"
  return NAME.test(trimmed) ? null : "invalid"
}

/* The base64 of a file's bytes (no data: prefix), built in pieces so a few megabytes do not overflow the call stack. */
export async function fileToBase64(file: Blob): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer())
  let binary = ""
  for (let at = 0; at < bytes.length; at += 0x8000) binary += String.fromCharCode(...bytes.subarray(at, at + 0x8000))
  return btoa(binary)
}
