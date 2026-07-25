# Hex Truchet — itch.io page description

*Paste-ready. itch.io's editor accepts headings, bold, and lists.*

---

## Hex Truchet

A two-player tile game about building curves you don't fully control.

Every tile is three arcs across a hexagon. Place one next to another and the
arcs connect. Keep going and eventually a chain of arcs closes into a loop —
and whoever completes it scores the area it encloses.

The catch: the loops belong to the board, not to you. You spend the whole
game laying groundwork that either player might cash in. Placing the tile
that finally seals a big loop is worth a lot. Handing your opponent the tile
placement that lets *them* seal it is worth considerably less.

### How to play

- You hold **3 tiles**. On your turn, pick one, rotate it, and place it.
- Tiles must be placed **next to a tile already on the board** (unless you
  switch on Free Placement in the menu).
- **Click a tile in your hand** to select it, **press R** to rotate it, then
  **click a highlighted cell** to place it. A ghost preview shows exactly
  where the arcs will land before you commit.
- Whenever your placement closes one or more loops, you score the number of
  cells each loop encloses.
- The board is 37 cells and the game runs exactly 37 turns. Highest score
  wins.

### Two things worth knowing

**Small loops are cheap.** Three tight turns meeting at a corner close very
easily, and they enclose almost nothing. They're free points, and they're
also a trap — you can spend the whole game collecting them and still lose.
The scoring rewards the rarer long loops that wrap real territory.

**Watch what you're setting up.** An open chain of arcs is a loop waiting to
happen, and it's waiting for *either* of you. If you leave one nearly closed
on your turn, look at what your opponent is holding first.

### Modes

- **Hot-seat** — two players, one screen.
- **Vs Bot** — four difficulty levels.
- **Replay** — step through a finished game move by move.

### About the bots

The opponents are pure search — no machine learning anywhere. They rank their
options by immediate gain, then simulate the rest of the game forward to see
which one actually pays off.

- **Random** — places legally, otherwise thinks about nothing.
- **Easy** — always takes the best immediate score. Surprisingly hard to beat
  if you're being careless.
- **Medium** — searches ahead, but deliberately plays the merely-greedy move
  about a third of the time. It plays coherently and then periodically walks
  into something it should have seen.
- **Hard** — the same search, looking further ahead, with no handicap at all.
  It is not trying to be fair to you.

Start on Medium. Easy is not as easy as it sounds.

### Controls

| Action | Input |
|---|---|
| Select tile | Click a tile in your hand |
| Rotate | **R** |
| Place | Click a highlighted cell |
| Back to menu | **Esc** |
| Replay: step | **←** / **→** |
| Replay: play / pause | **Space** |

---

Runs in the browser. Built in Godot 4.
