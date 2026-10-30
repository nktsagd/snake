import type { Metadata } from "next";
import "./style.css";

export const metadata: Metadata = {
  title: "VoiceSnake — Say it. Play it.",
  description: "A little snake game controlled by your voice, running in your browser.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
