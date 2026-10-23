import random

VECTORS = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}


class Snake:
    def __init__(self, size=24):
        self.size = size
        self.reset()

    def reset(self):
        n = self.size // 2
        self.body = [(n, n), (n-1, n), (n-2, n)]
        self.direction = self.pending_direction = 'right'
        self.score = 0
        self.over = self.paused = False
        self.food = self.spawn_food()

    def spawn_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.body]
        return random.choice(free) if free else None

    def turn(self, direction):
        if self.over or self.paused or direction not in VECTORS:
            return False
        a, b = VECTORS[self.direction], VECTORS[direction]
        if a == (-b[0], -b[1]):
            return False
        self.pending_direction = direction
        return True

    def tick(self):
        if self.over or self.paused:
            return False
        changed = self.direction != self.pending_direction
        self.direction = self.pending_direction
        dx, dy = VECTORS[self.direction]
        x, y = self.body[0]
        head = ((x+dx) % self.size, (y+dy) % self.size)
        eating = head == self.food
        occupied = self.body if eating else self.body[:-1]
        if head in occupied:
            self.over = True
            return changed
        self.body.insert(0, head)
        if eating:
            self.score += 1
            self.food = self.spawn_food()
            self.over = self.food is None
        else:
            self.body.pop()
        return changed
