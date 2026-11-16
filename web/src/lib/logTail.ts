/* The router's recent output (GET /v1/logs): its own lines and those of every model it started (a model loading, a failed
   start, the GPU giving way to the CPU). The page asks for what came after the last line it saw and keeps the newest. */

export interface LogLine {
  seq: number
  time: number        /* seconds since 1970 */
  source: string      /* "router", or the id of the model that printed it */
  text: string
}

export interface LogRead {
  /* false when the server has no such route (a plain `coli serve`). */
  available: boolean
  lines: LogLine[]
  next: number
  first: number
}

export const KEEP_LINES = 1000

/* The lines already held plus those just read, oldest first, the newest `keep` of them. Lines are numbered, so one the
   page already has (a read that overlapped the last) is not added twice. */
export function mergeLines(held: LogLine[], incoming: LogLine[], keep = KEEP_LINES): LogLine[] {
  if (!incoming.length) return held
  const last = held.length ? held[held.length - 1].seq : 0
  const fresh = incoming.filter((line) => line.seq > last)
  if (!fresh.length) return held
  const all = held.concat(fresh)
  return all.length > keep ? all.slice(all.length - keep) : all
}

/* True when the server dropped lines between two reads: the oldest it still holds is later than the one after the last
   the page saw. The page then says some lines are missing instead of showing a seamless log. */
export function lostLines(read: Pick<LogRead, "first">, since: number): boolean {
  return since > 0 && read.first > since + 1
}

/* "14:03:22" in the viewer's own zone. */
export function clockTime(seconds: number): string {
  const date = new Date(seconds * 1000)
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString()
}

/* The distinct sources in the lines, "router" first, then the models in the order they appeared. */
export function sourcesOf(lines: LogLine[]): string[] {
  const seen: string[] = []
  for (const line of lines) if (!seen.includes(line.source)) seen.push(line.source)
  return seen.sort((a, b) => (a === "router" ? -1 : b === "router" ? 1 : 0))
}
