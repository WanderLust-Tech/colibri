/* The context windows the model page offers for a model (docs/model-settings.md): the usual steps up to what its family supports, the model's
   own setting even when it is not one of them, and 0 for the default. */

const STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

export function contextChoices(current: number | null, max: number): number[] {
  const steps = STEPS.filter((tokens) => tokens <= max)
  if (current && !steps.includes(current)) steps.push(current)
  return [0, ...steps.sort((a, b) => a - b)]
}

/* 32768 as "32k", 1536 as "1.5k", an odd number as itself. */
export function contextLabel(tokens: number): string {
  if (tokens % 1024 === 0) return `${tokens / 1024}k`
  if (tokens % 512 === 0) return `${tokens / 1024}k`
  return String(tokens)
}
