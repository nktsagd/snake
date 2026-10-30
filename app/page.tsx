"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { initialGame, SIZE, tick, turn, type Direction } from "../lib/game";
import { Detector, LABELS, probabilities } from "../lib/detector";

type VoiceStatus = "loading" | "ready" | "listening" | "error";
const directions: Direction[] = ["up", "left", "down", "right"];
const arrows: Record<Direction, string> = { up: "↑", right: "→", down: "↓", left: "←" };

export default function Home() {
  const [game, setGame] = useState(initialGame);
  const [playing, setPlaying] = useState(false);
  const [voice, setVoice] = useState<VoiceStatus>("loading");
  const [detail, setDetail] = useState("Đang tải model nhận giọng nói…");
  const [last, setLast] = useState("—");
  const [confidence, setConfidence] = useState(0);
  const [latency, setLatency] = useState<number | null>(null);
  const [level, setLevel] = useState(0);
  const [best, setBest] = useState(0);
  const worker = useRef<Worker | null>(null);
  const detector = useRef(new Detector());
  const audio = useRef<{ context: AudioContext; stream: MediaStream; source: MediaStreamAudioSourceNode; worklet: AudioWorkletNode } | null>(null);
  const buffer = useRef(new Float32Array(16000));
  const filled = useRef(0); const busy = useRef(false); const enabled = useRef(false);
  const playingRef = useRef(false); const active = useRef(false); const lastVoice = useRef(0);

  const steer = useCallback((direction: Direction) => setGame(previous => turn(previous, direction)), []);
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => {
    const saved = Number(localStorage.getItem("voicesnake-best")) || 0; setBest(saved);
    const w = new Worker(new URL("../lib/inference.worker.ts", import.meta.url)); worker.current = w;
    w.onmessage = e => {
      if (e.data.type === "ready") { setVoice("ready"); setDetail("Model sẵn sàng trên thiết bị của bạn."); }
      if (e.data.type === "error") { busy.current = false; enabled.current = false; setVoice("error"); setDetail(`Không thể chạy model: ${e.data.message}`); }
      if (e.data.type === "result") {
        busy.current = false;
        if (!enabled.current) return;
        const probs = probabilities(new Float32Array(e.data.logits));
        const index = probs.indexOf(Math.max(...probs)); const label = LABELS[index];
        setLast(label); setConfidence(probs[index]); setLatency(e.data.ms);
        const command = detector.current.update(label, probs[index], active.current, performance.now());
        if (command && playingRef.current) steer(command);
      }
    };
    w.onerror = () => { enabled.current = false; setVoice("error"); setDetail("Không tải được bộ xử lý model. Hãy tải lại trang."); };
    w.postMessage({ type: "load" });
    return () => { enabled.current = false; w.terminate(); audio.current?.stream.getTracks().forEach(t => t.stop()); void audio.current?.context.close(); };
  }, [steer]);
  useEffect(() => {
    if (!playing || game.over) return;
    const timer = setInterval(() => setGame(previous => tick(previous)), 170);
    return () => clearInterval(timer);
  }, [playing, game.over]);
  useEffect(() => {
    if (game.score > best) { setBest(game.score); localStorage.setItem("voicesnake-best", String(game.score)); }
  }, [game.score, best]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const map: Record<string, Direction> = { ArrowUp: "up", w: "up", ArrowRight: "right", d: "right", ArrowDown: "down", s: "down", ArrowLeft: "left", a: "left" };
      const direction = map[event.key] ?? map[event.key.toLowerCase()];
      if (direction) { event.preventDefault(); steer(direction); }
      if (event.code === "Space" && !event.repeat) { event.preventDefault(); if (!game.over) setPlaying(p => !p); }
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [game.over, steer]);

  const stopVoice = useCallback(() => {
    enabled.current = false; active.current = false; detector.current = new Detector();
    const current = audio.current; audio.current = null;
    if (current) { current.worklet.disconnect(); current.source.disconnect(); current.stream.getTracks().forEach(t => t.stop()); void current.context.close(); }
    filled.current = 0; buffer.current.fill(0); setLevel(0); setVoice("ready"); setDetail("Micro đã tắt. Bật lại khi bạn sẵn sàng.");
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
      audio.current = { context, stream, source, worklet }; enabled.current = true; detector.current = new Detector();
      lastVoice.current = 0;
      worklet.port.onmessage = event => {
        if (!enabled.current) return;
        const samples = event.data as Float32Array; buffer.current.copyWithin(0, samples.length); buffer.current.set(samples, 16000 - samples.length);
        filled.current = Math.min(16000, filled.current + samples.length);
        let sum = 0; for (const x of samples) sum += x * x;
        const rms = Math.sqrt(sum / samples.length); setLevel(Math.min(1, rms * 12));
        const now = performance.now(); if (rms >= .004) lastVoice.current = now;
        active.current = now - lastVoice.current < 240;
        if (!active.current) { detector.current.update("background", 0, false, now); return; }
        if (filled.current === 16000 && !busy.current && worker.current) {
          busy.current = true; const copy = buffer.current.slice(); worker.current.postMessage({ type: "infer", samples: copy }, [copy.buffer]);
        }
      };
      await context.resume(); setVoice("listening"); setDetail("Đang lắng nghe. Nói một hướng bằng tiếng Anh.");
    } catch (error) {
      stream?.getTracks().forEach(t => t.stop()); if (context) void context.close(); enabled.current = false;
      setVoice("ready"); setDetail(`Không mở được micro: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  const restart = () => { setGame(initialGame()); setPlaying(true); };
  const statusText = voice === "listening" ? "ĐANG LẮNG NGHE" : voice === "loading" ? "ĐANG TẢI MODEL" : voice === "error" ? "CÓ LỖI" : "MIC ĐANG TẮT";
  return <main className="page">
    <header className="topbar"><a href="#" className="brand"><span className="brand-mark">✳</span> voice<span>snake</span><i> / playground</i></a><div className="top-right"><span className="live-dot"/> CHƠI NGAY TRÊN TRÌNH DUYỆT <span className="top-index">001 / 001</span></div></header>
    <section className="intro"><div><div className="eyebrow"><span className="mini-dot"/> MỘT TRÒ CHƠI NHỎ, MỘT GIỌNG NÓI LỚN</div><h1>Nói một lời.<br/><em>Rẽ một hướng.</em></h1><p>Điều khiển chú rắn bằng giọng nói của bạn.<br/>Bốn từ đơn giản, vô vàn đường đi.</p></div><div className="intro-note"><span>HOW TO PLAY</span><b>01 — 04</b><p>Ăn những chấm cam.<br/>Đừng tự cắn đuôi mình.</p><span className="scribble">let’s play ↗</span></div></section>
    <section className="workspace"><div className="board-card"><div className="board-header"><div><span className="section-label">THE PLAYGROUND</span><h2>Voice Snake <span>✳</span></h2></div><div className="scoreboard"><div><small>ĐIỂM</small><b>{String(game.score).padStart(2, "0")}</b></div><div><small>KỶ LỤC</small><b>{String(best).padStart(2, "0")}</b></div></div></div>
      <div className="board-wrap"><div className="board" role="group" aria-label={`Bàn chơi, điểm ${game.score}${game.over ? ", kết thúc" : ""}`}>
        {game.snake.map((part, i) => <div key={`${part.x}-${part.y}`} className={`snake ${i === 0 ? "head " + game.direction : ""}`} style={{ left: `${part.x / SIZE * 100}%`, top: `${part.y / SIZE * 100}%`, width: `${100 / SIZE}%`, height: `${100 / SIZE}%` }}>{i === 0 && <><i/><i/></>}</div>)}
        <div className="food" style={{ left: `${(game.food.x + .5) / SIZE * 100}%`, top: `${(game.food.y + .5) / SIZE * 100}%` }} />
        {!playing && !game.over && <div className="board-overlay"><span className="overlay-icon">✳</span><b>Sẵn sàng chơi?</b><p>Bắt đầu bằng một cú nhấn.</p><button className="primary" onClick={() => setPlaying(true)}>Bắt đầu chơi <span>↗</span></button></div>}
        {game.over && <div className="board-overlay"><span className="overlay-icon">✳</span><b>{game.won ? "Bạn đã chiến thắng!" : "Ôi, va vào đuôi rồi!"}</b><p>Điểm của bạn: {game.score}</p><button className="primary" onClick={restart}>Chơi lại <span>↗</span></button></div>}
      </div></div><div className="board-footer"><span><b>✳</b> Băng qua mép để xuất hiện ở phía bên kia.</span><button className="text-button" onClick={() => game.over ? restart() : setPlaying(p => !p)}>{game.over ? "↻ Chơi lại" : playing ? "Ⅱ Tạm dừng" : "▶ Tiếp tục"}</button></div></div>
      <aside className="side"><div className="voice-card"><div className="card-top"><span className="section-label">VOICE CONTROL</span><span className="round-icon">◉</span></div><h3>Giọng nói<br/><em>của bạn.</em></h3><p>Bật micro, rồi nói một trong bốn hướng để điều khiển.</p><button className={`mic-button ${voice === "listening" ? "on" : ""}`} onClick={startVoice} disabled={voice === "loading" || voice === "error"}><span>◉</span>{voice === "listening" ? "Tắt microphone" : "Bật microphone"}<b>{voice === "listening" ? "×" : "↗"}</b></button><div className="status"><span className={`status-dot ${voice === "listening" ? "active" : ""}`}/>{statusText}<div className="meter"><i style={{ width: `${level * 100}%` }}/></div></div><div className="voice-detail" aria-live="polite">{detail}</div></div>
      <div className="commands-card"><div className="card-top"><span className="section-label">BỐN TỪ KỲ DIỆU</span><span className="muted">01 — 04</span></div><div className="command-grid">{directions.map(d => <div className="command" key={d}><span>{arrows[d]}</span><div><b>{d.toUpperCase()}</b><small>{({ up: "LÊN", left: "TRÁI", down: "XUỐNG", right: "PHẢI" })[d]}</small></div></div>)}</div><div className="diagnostic"><span>MODEL NGHE ĐƯỢC</span><b>{last.toUpperCase()} {last !== "—" && `${Math.round(confidence * 100)}%`}</b></div>{latency !== null && <div className="diagnostic"><span>INFERENCE</span><b>{latency.toFixed(0)} ms</b></div>}</div>
      <div className="keyboard-card"><span>⌨</span><div><b>Thích dùng bàn phím?</b><p>Dùng phím mũi tên hoặc W A S D.</p></div></div><div className="touch-controls" aria-label="Điều khiển cảm ứng">{directions.map(d => <button key={d} aria-label={d} onClick={() => steer(d)}>{arrows[d]}</button>)}</div></aside></section>
    <footer><span>✳ VOICESNAKE <i>— made for curious voices.</i></span><span>ÂM THANH ĐƯỢC XỬ LÝ TRÊN THIẾT BỊ CỦA BẠN.</span></footer>
  </main>;
}
