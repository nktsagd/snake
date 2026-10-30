import type { Direction } from "./game";
export const LABELS = ["up", "right", "left", "down", "unknown", "background"] as const;
export function probabilities(logits: Float32Array): number[] {
  const max = Math.max(...logits), exp = Array.from(logits, n => Math.exp(n - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(n => n / sum);
}
export class Detector {
  private candidate = ""; private count = 0; private last = ""; private quietSince: number | null = null; private emitted = -Infinity;
  constructor(public threshold = 0.919165, public confirmations = 2) {}
  update(label: string, confidence: number, active: boolean, now: number): Direction | null {
    if (!active) {
      this.candidate = ""; this.count = 0; this.quietSince ??= now;
      if (now - this.quietSince >= 300) this.last = "";
      return null;
    }
    this.quietSince = null;
    if (label === this.last || now - this.emitted < 350 || !LABELS.slice(0, 4).includes(label as Direction) || confidence < this.threshold) {
      this.candidate = ""; this.count = 0; return null;
    }
    this.count = this.candidate === label ? this.count + 1 : 1; this.candidate = label;
    if (this.count < this.confirmations) return null;
    this.last = label; this.emitted = now; this.count = 0;
    return label as Direction;
  }
}
