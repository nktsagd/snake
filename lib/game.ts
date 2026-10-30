export type Direction = "up" | "right" | "left" | "down";
type Point = { x: number; y: number };
export type Game = { snake: Point[]; direction: Direction; pending: Direction; food: Point; score: number; over: boolean; won: boolean };
export const SIZE = 18;
const vectors: Record<Direction, Point> = { up: { x: 0, y: -1 }, right: { x: 1, y: 0 }, down: { x: 0, y: 1 }, left: { x: -1, y: 0 } };
const opposite: Record<Direction, Direction> = { up: "down", down: "up", left: "right", right: "left" };

export function initialGame(): Game {
  return { snake: [{ x: 8, y: 9 }, { x: 7, y: 9 }, { x: 6, y: 9 }], direction: "right", pending: "right", food: { x: 13, y: 9 }, score: 0, over: false, won: false };
}
export function turn(game: Game, direction: Direction): Game {
  if (game.over || game.pending !== game.direction || opposite[game.direction] === direction) return game;
  return { ...game, pending: direction };
}
export function tick(game: Game, random = Math.random): Game {
  if (game.over) return game;
  const d = vectors[game.pending];
  const head = { x: (game.snake[0].x + d.x + SIZE) % SIZE, y: (game.snake[0].y + d.y + SIZE) % SIZE };
  const eats = head.x === game.food.x && head.y === game.food.y;
  const occupied = eats ? game.snake : game.snake.slice(0, -1);
  if (occupied.some(p => p.x === head.x && p.y === head.y)) return { ...game, over: true };
  const snake = [head, ...game.snake.slice(0, eats ? undefined : -1)];
  if (!eats) return { ...game, snake, direction: game.pending };
  const empty: Point[] = [];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) if (!snake.some(p => p.x === x && p.y === y)) empty.push({ x, y });
  if (!empty.length) return { ...game, snake, score: game.score + 1, over: true, won: true };
  return { ...game, snake, food: empty[Math.min(empty.length - 1, Math.floor(random() * empty.length))], score: game.score + 1, direction: game.pending };
}
