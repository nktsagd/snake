import { describe, expect, test } from "bun:test";
import { initialGame, tick, turn, SIZE } from "../lib/game";
import { Detector, probabilities } from "../lib/detector";
import * as ort from "onnxruntime-web/wasm";
import { runInNewContext } from "node:vm";

describe("game", () => {
  test("turns only once per tick and rejects reversal", () => {
    let game = initialGame();
    expect(turn(game, "left")).toEqual(game);
    game = turn(game, "up");
    expect(turn(game, "left").pending).toBe("up");
    expect(tick(game).snake[0]).toEqual({ x: 8, y: 8 });
  });
  test("wraps around the board and grows on food", () => {
    const game = { ...initialGame(), snake: [{ x: SIZE - 1, y: 9 }, { x: SIZE - 2, y: 9 }], food: { x: 0, y: 9 } };
    const result = tick(game, () => 0);
    expect(result.snake[0]).toEqual({ x: 0, y: 9 }); expect(result.score).toBe(1); expect(result.snake).toHaveLength(3);
  });
});
describe("voice decisions", () => {
  test("requires confidence and repeated direction, rejects unknown and rearms after silence", () => {
    const d = new Detector(.9, 2);
    expect(d.update("up", .89, true, 1000)).toBeNull();
    expect(d.update("unknown", .99, true, 1100)).toBeNull();
    expect(d.update("up", .95, true, 1200)).toBeNull();
    expect(d.update("up", .95, true, 1300)).toBe("up");
    expect(d.update("up", .95, true, 1800)).toBeNull();
    d.update("background", 0, false, 1900); d.update("background", 0, false, 2201);
    expect(d.update("up", .95, true, 2300)).toBeNull();
    expect(d.update("up", .95, true, 2400)).toBe("up");
  });
  test("softmax remains stable for large logits", () => {
    const p = probabilities(new Float32Array([1000, 1001, 999, 998, 997, 996]));
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p.indexOf(Math.max(...p))).toBe(1);
  });
  test("uses the configured cooldown between different commands", () => {
    const d = new Detector(.5, 1, 800);
    expect(d.update("up", .9, true, 1000)).toBe("up");
    expect(d.update("left", .9, true, 1500)).toBeNull();
    expect(d.update("left", .9, true, 1800)).toBe("left");
  });
});
test("ONNX WASM loads the shipped model and produces six finite logits", async () => {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = `${process.cwd()}/node_modules/onnxruntime-web/dist/`;
  const session = await ort.InferenceSession.create(await Bun.file("public/voicesnake.onnx").arrayBuffer(), { executionProviders: ["wasm"] });
  const output = await session.run({ waveform: new ort.Tensor("float32", new Float32Array(16000), [1, 16000]) });
  expect(output.logits.dims).toEqual([1, 6]);
  expect(Array.from(output.logits.data as Float32Array).every(Number.isFinite)).toBe(true);
});
test("audio worklet resamples continuous input into 16 kHz blocks", async () => {
  const source = await Bun.file("public/audio-worklet.js").text();
  for (const rate of [16000, 44100, 48000]) {
    const blocks: Float32Array[] = []; let Processor: any;
    class Base { port = { postMessage: (data: Float32Array) => blocks.push(data) }; }
    runInNewContext(source, { AudioWorkletProcessor: Base, sampleRate: rate, Float32Array, registerProcessor: (_: string, c: any) => { Processor = c; } });
    const processor = new Processor();
    for (let start = 0; start < rate; start += 128) processor.process([[new Float32Array(Math.min(128, rate - start)).fill(.25)]]);
    expect(blocks).toHaveLength(10);
    expect(blocks.every(block => block.length === 1600 && Math.abs(block[0] - .25) < 1e-6)).toBe(true);
  }
});
