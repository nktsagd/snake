import * as ort from "onnxruntime-web/wasm";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";
let session: ort.InferenceSession | null = null;
self.onmessage = async (event: MessageEvent) => {
  try {
    if (event.data.type === "load") {
      session = await ort.InferenceSession.create("/voicesnake.onnx", { executionProviders: ["wasm"] });
      self.postMessage({ type: "ready" });
    } else if (event.data.type === "infer" && session) {
      const start = performance.now();
      const result = await session.run({ waveform: new ort.Tensor("float32", event.data.samples, [1, 16000]) });
      self.postMessage({ type: "result", logits: Array.from(result.logits.data as Float32Array), ms: performance.now() - start });
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
