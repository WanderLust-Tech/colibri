import type { HealthResponse } from "./api"

export function activeRequests(health: HealthResponse | null): number {
  return Number(health?.scheduler?.active || 0)
}

export function supportsCacheSlots(health: HealthResponse | null): boolean {
  return typeof health?.kv_slots === "number" && health.kv_slots > 0
}

/* Whether the server has a feature. A server that does not say (no `features` block, or a name this list does not know) is
   asked: only an explicit false hides something. */
export function serverHas(health: HealthResponse | null, name: string): boolean {
  const value = health?.features?.[name]
  return value === undefined || value !== false
}

/* The settings pages to offer. Requests, Presets and the Log are what a router has: against `coli serve` they would only say so.
   (A router started without a request log or a presets file keeps its pages: they explain how to turn them on.) */
export function settingsPages(health: HealthResponse | null): string[] {
  const router = serverHas(health, "router")
  return ["general", "model", ...(router ? ["presets", "voices", "requests"] : []), ...(health?.features?.library === true ? ["library"] : []), ...(serverHas(health, "server_log") ? ["log"] : []), "connection", "system"]
}

/* Only an explicit true: a server that predates the field answers a trailing
   assistant turn fresh, so a Continue button there would mislabel a re-answer. */
export function supportsContinuation(health: HealthResponse | null): boolean {
  return health?.continue_assistant === true
}
