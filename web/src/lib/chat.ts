import type { ChatMessage } from "./api"

export interface ResendTurn {
  text: string
  previous: ChatMessage[]
  pictures: string[]
}

/* Rebuild the last user turn so regenerate can send it again. The pictures
   live on that message, not in the composer: a follow-up without them would
   get an answer about nothing, and an image-only turn would not send at all. */
export function resendFrom(messages: ChatMessage[], assistantIndex: number): ResendTurn | null {
  let userIndex = -1
  for (let i = 0; i < assistantIndex; i++) {
    if (messages[i].role === "user") userIndex = i
  }
  if (userIndex < 0) return null
  const user = messages[userIndex]
  return {
    text: user.content,
    previous: messages.slice(0, userIndex),
    pictures: user.images ? [...user.images] : [],
  }
}

/* A turn of plain text chat. Image and music turns are not text the model can be asked to rewrite, edit or branch. */
export function isPlainTurn(message: ChatMessage): boolean {
  return !message.mode && !message.generated && !message.song
}

export interface EditTurn {
  text: string
  pictures: string[]
  /* Everything before the message, which stays. The message and what followed it are cut. */
  previous: ChatMessage[]
}

/* Take a user message back to the composer. The conversation is cut at that message, so the model reads everything
   after it again on the next send: the start of the prompt up to the edit is unchanged and stays cached, the rest
   is not. null for a message that is not a plain user turn. */
export function editFrom(messages: ChatMessage[], userIndex: number): EditTurn | null {
  const user = messages[userIndex]
  if (!user || user.role !== "user" || !isPlainTurn(user)) return null
  return { text: user.content, pictures: user.images ? [...user.images] : [], previous: messages.slice(0, userIndex) }
}

/* A new conversation that keeps this reply and everything before it. null unless it is a finished plain assistant
   reply (a half-written one has nothing to branch from). */
export function branchAt(messages: ChatMessage[], index: number): ChatMessage[] | null {
  const reply = messages[index]
  if (!reply || reply.role !== "assistant" || !isPlainTurn(reply) || !reply.content.trim()) return null
  return messages.slice(0, index + 1)
}

/* One-click follow-ups on the last reply. They are ordinary messages added to the end, so the conversation before
   them is unchanged and stays cached. The text goes to the model, so it is the dictionary's own wording. */
export const REWRITES = {
  shorter: "rewrite.shorterPrompt",
  simpler: "rewrite.simplerPrompt",
} as const
export type Rewrite = keyof typeof REWRITES
