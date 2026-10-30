import { mkdir, copyFile } from "node:fs/promises";
await mkdir("public/ort", { recursive: true });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(`node_modules/onnxruntime-web/dist/${file}`, `public/ort/${file}`);
}
