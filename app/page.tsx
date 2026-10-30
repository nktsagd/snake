"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { initialGame, SIZE, tick, turn, type Direction } from "../lib/game";
import { Detector, LABELS, probabilities } from "../lib/detector";

type VoiceStatus = "loading" | "ready" | "listening" | "error";
const directions: Direction[] = ["up", "left", "down", "right"];
const arrows: Record<Direction, string> = { up: "↑", right: "→", down: "↓", left: "←" };
const defaults = { speed: 3, threshold: .70, energy: .004, confirmations: 1, cooldown: 350 };

export default function Home() {
  const [game, setGame] = useState(initialGame);
  const [playing, setPlaying] = useState(false);
  const [voice, setVoice] = useState<VoiceStatus>("loading");
  const [detail, setDetail] = useState("Đang tải model…");
  const [last, setLast] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [latency, setLatency] = useState<number | null>(null);
  const [level, setLevel] = useState(0);
  const [best, setBest] = useState(0);
  const [settings, setSettings] = useState(defaults);
  const [decision, setDecision] = useState("Chưa có dự đoán");
  const settingsRef = useRef(settings);
  const worker = useRef<Worker | null>(null);
  const detector = useRef(new Detector());
  const audio = useRef<{ context: AudioContext; stream: MediaStream; source: MediaStreamAudioSourceNode; worklet: AudioWorkletNode } | null>(null);
  const buffer = useRef(new Float32Array(16000));
  const filled = useRef(0); const busy = useRef(false); const enabled = useRef(false);
  const generation = useRef(0);
  const playingRef = useRef(false); const active = useRef(false); const lastVoice = useRef(0);

  const steer = useCallback((direction: Direction) => setGame(previous => turn(previous, direction)), []);
  useEffect(() => {
    playingRef.current = playing;
    detector.current = new Detector(settingsRef.current.threshold, settingsRef.current.confirmations, settingsRef.current.cooldown);
  }, [playing]);
  useEffect(() => { settingsRef.current = settings; }, [settings]);
  useEffect(() => {
    detector.current = new Detector(settings.threshold, settings.confirmations, settings.cooldown);
  }, [settings.threshold, settings.confirmations, settings.cooldown]);
  useEffect(() => {
    const saved = Number(localStorage.getItem("voicesnake-best")) || 0; setBest(saved);
    const w = new Worker(new URL("../lib/inference.worker.ts", import.meta.url)); worker.current = w;
    w.onmessage = e => {
      if (e.data.type === "ready") { setVoice("ready"); setDetail("Model sẵn sàng."); }
      if (e.data.type === "error") {
        if (e.data.id !== undefined && e.data.id !== generation.current) return;
        busy.current = false; enabled.current = false; audio.current?.stream.getTracks().forEach(t => t.stop());
        if (audio.current) void audio.current.context.close(); audio.current = null;
        setVoice("error"); setDetail(`Không thể chạy model: ${e.data.message}`);
      }
      if (e.data.type === "result") {
        if (e.data.id !== generation.current) return;
        busy.current = false;
        if (!enabled.current) return;
        const probs = probabilities(new Float32Array(e.data.logits));
        const index = probs.indexOf(Math.max(...probs)); const label = LABELS[index];
        setLast(label); setConfidence(probs[index]); setLatency(e.data.ms);
        const command = playingRef.current ? detector.current.update(label, probs[index], active.current, performance.now()) : null;
        setDecision(!playingRef.current ? "Đang tạm dừng" : !active.current ? "Không có tiếng nói" : index >= 4 ? `Bỏ qua: ${label}` : probs[index] < settingsRef.current.threshold ? "Dưới ngưỡng confidence" : command ? `Đã nhận: ${command.toUpperCase()}` : "Chờ xác nhận");
        if (command && playingRef.current) steer(command);
      }
    };
    w.onerror = () => { enabled.current = false; setVoice("error"); setDetail("Không tải được model. Hãy tải lại trang."); };
    w.postMessage({ type: "load" });
    return () => { enabled.current = false; w.terminate(); audio.current?.stream.getTracks().forEach(t => t.stop()); void audio.current?.context.close(); };
  }, [steer]);
  useEffect(() => {
    if (!playing || game.over) return;
    const timer = setInterval(() => setGame(previous => tick(previous)), 1000 / settings.speed);
    return () => clearInterval(timer);
  }, [playing, game.over, settings.speed]);
  useEffect(() => {
    if (game.score > best) { setBest(game.score); localStorage.setItem("voicesnake-best", String(game.score)); }
  }, [game.score, best]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement && (event.target.matches("input, select, textarea") || event.target.isContentEditable || (event.target.matches("button") && event.code === "Space"))) return;
      const map: Record<string, Direction> = { ArrowUp: "up", w: "up", ArrowRight: "right", d: "right", ArrowDown: "down", s: "down", ArrowLeft: "left", a: "left" };
      const direction = map[event.key] ?? map[event.key.toLowerCase()];
      if (direction) { event.preventDefault(); steer(direction); }
      if (event.code === "Space" && !event.repeat) { event.preventDefault(); if (!game.over) setPlaying(p => !p); }
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [game.over, steer]);

  const stopVoice = useCallback(() => {
    enabled.current = false; active.current = false; generation.current++; busy.current = false;
    detector.current = new Detector(settingsRef.current.threshold, settingsRef.current.confirmations, settingsRef.current.cooldown);
    const current = audio.current; audio.current = null;
    if (current) { current.worklet.disconnect(); current.source.disconnect(); current.stream.getTracks().forEach(t => t.stop()); void current.context.close(); }
    filled.current = 0; buffer.current.fill(0); setLevel(0); setVoice("ready"); setDetail("Micro đã tắt.");
  }, []);

  const startVoice = async () => {
    if (voice === "listening") { stopVoice(); return; }
    if (voice !== "ready") return;
    let stream: MediaStream | null = null; let context: AudioContext | null = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone cần HTTPS hoặc localhost và trình duyệt hỗ trợ.");
      setVoice("loading");
      setDetail("Đang chờ quyền microphone…");
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      context = new AudioContext(); await context.audioWorklet.addModule("/audio-worklet.js");
      const source = context.createMediaStreamSource(stream); const worklet = new AudioWorkletNode(context, "capture-16k");
      const mute = context.createGain(); mute.gain.value = 0;
      source.connect(worklet); worklet.connect(mute); mute.connect(context.destination);
      audio.current = { context, stream, source, worklet }; enabled.current = true;
      generation.current++; busy.current = false;
      detector.current = new Detector(settingsRef.current.threshold, settingsRef.current.confirmations, settingsRef.current.cooldown);
      lastVoice.current = 0;
      worklet.port.onmessage = event => {
        if (!enabled.current) return;
        const samples = event.data as Float32Array; buffer.current.copyWithin(0, samples.length); buffer.current.set(samples, 16000 - samples.length);
        filled.current = Math.min(16000, filled.current + samples.length);
        let sum = 0; for (const x of samples) sum += x * x;
        const rms = Math.sqrt(sum / samples.length); setLevel(Math.min(1, rms * 12));
        const now = performance.now(); if (rms >= settingsRef.current.energy) lastVoice.current = now;
        active.current = now - lastVoice.current < 240;
        if (!active.current) { detector.current.update("background", 0, false, now); return; }
        if (filled.current === 16000 && !busy.current && worker.current) {
          busy.current = true; const copy = buffer.current.slice(); worker.current.postMessage({ type: "infer", id: generation.current, samples: copy }, [copy.buffer]);
        }
      };
      await context.resume(); setVoice("listening"); setDetail("Đang nghe.");
    } catch (error) {
      stream?.getTracks().forEach(t => t.stop()); if (context) void context.close(); enabled.current = false;
      setVoice("ready"); setDetail(`Không mở được micro: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  const restart = () => { setGame(initialGame()); setPlaying(true); };
  const statusText = voice === "listening" ? "ĐANG LẮNG NGHE" : voice === "loading" ? "ĐANG TẢI MODEL" : voice === "error" ? "CÓ LỖI" : "MIC ĐANG TẮT";
  return <main className="page">
    <header className="topbar"><a href="#" className="brand"><span className="brand-mark">✳</span> voice<span>snake</span></a><div className="mic-top"><div className="mic-info"><div className="status"><span className={`status-dot ${voice === "listening" ? "active" : ""}`}/>{statusText}<div className="meter"><i style={{ width: `${level * 100}%` }}/></div></div><div className="voice-detail" aria-live="polite">{detail}</div></div><button className={`mic-button ${voice === "listening" ? "on" : ""}`} onClick={startVoice} disabled={voice === "loading" || voice === "error"}>{voice === "listening" ? "Tắt micro" : "Bật micro"}</button></div></header>
    <section className="workspace"><div className="board-card"><div className="board-header"><div><h2>Bàn chơi</h2></div><div className="scoreboard"><div><small>ĐIỂM</small><b>{String(game.score).padStart(2, "0")}</b></div><div><small>KỶ LỤC</small><b>{String(best).padStart(2, "0")}</b></div></div></div>
      <div className="board-wrap"><div className="board" role="group" aria-label={`Bàn chơi, điểm ${game.score}${game.over ? ", kết thúc" : ""}`}>
        {game.snake.map((part, i) => <div key={`${part.x}-${part.y}`} className={`snake ${i === 0 ? "head " + game.direction : ""}`} style={{ left: `${part.x / SIZE * 100}%`, top: `${part.y / SIZE * 100}%`, width: `${100 / SIZE}%`, height: `${100 / SIZE}%` }}>{i === 0 && <><i/><i/></>}</div>)}
        <div className="food" style={{ left: `${(game.food.x + .5) / SIZE * 100}%`, top: `${(game.food.y + .5) / SIZE * 100}%` }} />
        {!playing && !game.over && <div className="board-overlay"><button className="primary" onClick={() => setPlaying(true)}>Bắt đầu</button></div>}
        {game.over && <div className="board-overlay"><b>{game.won ? "Chiến thắng" : "Kết thúc"}</b><p>Điểm: {game.score}</p><button className="primary" onClick={restart}>Chơi lại</button></div>}
      </div></div><div className="board-footer"><span>Rắn đi xuyên tường.</span><button className="text-button" onClick={() => game.over ? restart() : setPlaying(p => !p)}>{game.over ? "↻ Chơi lại" : playing ? "Ⅱ Tạm dừng" : "▶ Tiếp tục"}</button></div></div>
      <aside className="side"><div className="commands-card diagnostics-card"><div className="diagnostic"><span>MODEL NGHE ĐƯỢC</span><b>{last.toUpperCase()} {last !== "—" && `${Math.round(confidence * 100)}%`}</b></div><div className="diagnostic"><span>TRẠNG THÁI</span><b>{decision}</b></div>{latency !== null && <div className="diagnostic"><span>INFERENCE</span><b>{latency.toFixed(0)} ms</b></div>}</div>
      <div className="settings-card"><div className="card-top"><span className="section-label">CÀI ĐẶT</span><button className="text-button" onClick={() => setSettings(defaults)}>Đặt lại</button></div>
        <label>Tốc độ rắn <output>{settings.speed} ô/giây</output><input type="range" min="2" max="10" step="1" value={settings.speed} onChange={e => setSettings(s => ({ ...s, speed: Number(e.target.value) }))}/></label>
        <label>Confidence tối thiểu <output>{Math.round(settings.threshold * 100)}%</output><input type="range" min="0.1" max="0.99" step="0.01" value={settings.threshold} onChange={e => setSettings(s => ({ ...s, threshold: Number(e.target.value) }))}/></label>
        <label>Ngưỡng âm lượng RMS <output>{settings.energy.toFixed(3)}</output><input type="range" min="0.001" max="0.03" step="0.001" value={settings.energy} onChange={e => setSettings(s => ({ ...s, energy: Number(e.target.value) }))}/></label>
        <label>Số lần xác nhận <output>{settings.confirmations}</output><input type="range" min="1" max="4" step="1" value={settings.confirmations} onChange={e => setSettings(s => ({ ...s, confirmations: Number(e.target.value) }))}/></label>
        <label>Khoảng nghỉ giữa lệnh <output>{settings.cooldown} ms</output><input type="range" min="0" max="1000" step="50" value={settings.cooldown} onChange={e => setSettings(s => ({ ...s, cooldown: Number(e.target.value) }))}/></label>
        <p>Giảm ngưỡng giúp dễ nhận lệnh hơn nhưng có thể tăng nhận nhầm.</p>
      </div>
      <div className="touch-controls" aria-label="Điều khiển cảm ứng">{directions.map(d => <button key={d} aria-label={d} onClick={() => steer(d)}>{arrows[d]}</button>)}</div></aside></section>
    <section className="commands-card instructions"><div className="card-top"><span className="section-label">LỆNH GIỌNG NÓI</span></div><div className="command-grid">{directions.map(d => <div className="command" key={d}><span>{arrows[d]}</span><div><b>{d.toUpperCase()}</b><small>{({ up: "LÊN", left: "TRÁI", down: "XUỐNG", right: "PHẢI" })[d]}</small></div></div>)}</div></section>
    <footer>Âm thanh được xử lý trên thiết bị.</footer>
  </main>;
}
