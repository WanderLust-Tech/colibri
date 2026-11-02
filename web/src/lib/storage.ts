export interface StringStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export function stored(storage: Pick<StringStorage, "getItem">, key: string, fallback: string) {
  try { return storage.getItem(key) || fallback } catch { return fallback }
}

/* A link such as http://host:8000/#key=... carries the server's API key to the dashboard. The
   fragment is never sent to the server; it is read once and removed from the address bar, and the
   key stays in memory like one typed by hand. */
export function takeKeyFromLocation(
  location: Pick<Location, "hash" | "pathname" | "search">,
  history: Pick<History, "replaceState">,
): string {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""))
  const key = params.get("key") ?? ""
  if (!params.has("key")) return ""
  params.delete("key")
  const rest = params.toString()
  try { history.replaceState(null, "", location.pathname + location.search + (rest ? `#${rest}` : "")) } catch { /* no history API */ }
  return key
}

export function persistPublicSettings(storage: StringStorage, baseUrl: string, model: string) {
  try {
    storage.setItem("colibri.baseUrl", baseUrl)
    storage.setItem("colibri.model", model)
    // API credentials intentionally remain memory-only. Remove values left by
    // older web releases whenever public settings are persisted.
    storage.removeItem("colibri.apiKey")
  } catch { /* restricted storage mode */ }
}
