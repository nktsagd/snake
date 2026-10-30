// Integrate each input sample over output intervals: works at 44.1/48 kHz as well as 16 kHz.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.remaining = sampleRate / 16000; this.sum = 0; this.block = new Float32Array(1600); this.offset = 0; }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    const ratio = sampleRate / 16000;
    for (const sample of input) {
      let left = 1;
      while (left > 1e-9) {
        const take = Math.min(left, this.remaining);
        this.sum += sample * take; this.remaining -= take; left -= take;
        if (this.remaining < 1e-9) {
          this.block[this.offset++] = this.sum / ratio;
          this.sum = 0; this.remaining = ratio;
          if (this.offset === this.block.length) {
            this.port.postMessage(this.block, [this.block.buffer]);
            this.block = new Float32Array(1600); this.offset = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("capture-16k", CaptureProcessor);
