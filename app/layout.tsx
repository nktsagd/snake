import type { Metadata } from "next";
import "./style.css";

export const metadata: Metadata = {
  title: "VoiceSnake",
  description: "Trò chơi rắn điều khiển bằng giọng nói.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
