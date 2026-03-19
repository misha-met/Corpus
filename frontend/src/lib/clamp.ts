/**
 * Clamp a numeric value to [min, max], returning fallback if not finite.
 */
export function clamp(
  value: number,
  min: number,
  max: number,
  fallback: number,
): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}
