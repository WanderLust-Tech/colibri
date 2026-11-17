/* The labels the dashboard puts on what it asks for (X-Colibri-Tags, docs/api.md "Client tags"): which app, and which conversation. */

/* A short name for one conversation: "c-" and six hex digits. Made when a chat starts; not a secret, only a way to find its requests. */
export function newChatTag(random: () => number = Math.random): string {
  let hex = ""
  for (let i = 0; i < 6; i++) hex += Math.floor(random() * 16).toString(16)
  return `c-${hex}`
}

export function dashboardTags(chat: string): Record<string, string> {
  return { app: "dashboard", chat }
}
