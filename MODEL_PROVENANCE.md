# VoiceSnake GRU4 — bundled model provenance

This desktop release uses the GRU trained from scratch in `VoiceSnake_Train_GRU_4_Commands.ipynb`.

- Classes, in order: `up`, `right`, `left`, `down`.
- Model: log-spectrogram frontend, bidirectional GRU (hidden 64 per direction), mean/max pooling, 4-class head; 80,452 parameters.
- Input: float32 mono waveform, 16 kHz, 16,000 samples. Frontend is included in TorchScript.
- Training: TensorFlow mini Speech Commands, four direction classes only, seed 42, speaker-disjoint SHA-1 split. Best validation-loss checkpoint epoch 18; early stopping epoch 26.
- Export runtime: PyTorch 2.8.0 CPU.
- Saved test result: 599/637 correct (94.03%), macro-F1 0.940698. These are recorded training-run results, not a new cross-machine evaluation.
- Model bytes and labels are checked against `voicesnake/assets/manifest.json` before loading.

## Bundled SHA-256

- `voicesnake_gru4.pt`: `33f8a28ce9699e9fc7e51b7232988f343b36a2fc6d3338c1d954471cd1c49eef`
- `labels.json`: `8d1141cff42a6f5b73375abdac75981aff8f1f0d8caec548b7c8c04f02030caf`

## Limitations

Four classes do not provide unknown-word rejection. In the notebook's whole-clip check, 82.17% of 690 outside-command clips passed confidence 0.7 / RMS 0.004. Live detector confirmation/cooldown were not included in that measurement. Microphone end-to-end latency and Windows/Linux hardware behavior have not been verified here.

Source data: https://www.tensorflow.org/tutorials/audio/simple_audio . Dataset/audio is not included in the desktop release; the notebook downloads it separately. Historical model alternatives are not required by this package.
