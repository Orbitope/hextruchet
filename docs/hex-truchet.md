# Hex Truchet

*Designing a tile game, building it, and three different ways of failing to
beat a greedy heuristic.*

---

A Truchet tile is a square with a decoration that isn't symmetric under
rotation — two quarter-circle arcs, say, connecting midpoints of adjacent
edges. Tile a plane with them at random rotations and the arcs chain across
tile boundaries into long meandering curves. It's a hundred-year-old idea
that still shows up in generative art, and the appeal is that the pattern is
entirely emergent: no single tile knows anything about the curve it's part
of.

Hex Truchet is what happens if you put that on a hex grid and make two people
fight over it. You can [play it in your
browser](https://orbitope.itch.io/hex-truchet) if you'd rather see it than
read about it.

Each cell is a hexagon with six edges. A tile is three arcs that pair those
six edges up — a perfect matching. Place a tile next to an existing one and
the arcs connect across the shared edge. Keep going and eventually a chain of
arcs bites its own tail and closes into a loop. Closing a loop scores you the
area it encloses, and whoever has more points at the end wins.

That's the whole game. Two players, a hand of three tiles, place one and draw
one, thirty-seven cells on a radius-3 board, thirty-seven turns, done. The
interesting part is that the thing you're competing over — loops — is a
property of the board as a whole, and neither of you fully controls it. You
spend the game building infrastructure that either of you might cash in.

## The tile vocabulary is small

There are 15 ways to pair up six edges. Quotient those by rotation and you
get **five distinct tiles**:

| Tile | Arc spans | Character | Distinct rotations |
|---|---|---|---|
| 0 | (1,1,1) | three tight turns | 2 |
| 1 | (1,2,2) | mixed | 6 |
| 2 | (1,1,3) | two tight turns + a straight-through | 3 |
| 3 | (2,2,3) | mixed | 3 |
| 4 | (3,3,3) | all three straight through | 1 |

Five shapes is a good number for a physical game. You can learn them in a
minute, and there's real texture between them — tile 0 curls everything back
on itself, tile 4 sends everything straight across.

The rotation counts in that last column are not trivia. Tile 4 looks
identical in all six rotations, so in the finished game, pressing R on a
straight-through tile does nothing, six times in a row. Which reads, to a
player, as a broken control. The UI has to skip aliased rotations. It's the
sort of thing you only discover by holding the thing in your hands.

## The deck is the design

The first real finding was that a uniform deck — equal numbers of all five
tiles — produces a boring game. On a 37-cell board it closes an average of
**0.67 loops per game**, and **51% of boards close no loops at all**. Half
your games end with nobody scoring. That's not a game, that's a screensaver.

The fix turned out to be blunt. Throw out three of the five tiles and use
only tile 0 and tile 2, in a **1:2 ratio** — twelve and twenty-five on a
37-cell board. That yields **4.28 loops per game** with a 0.6% chance of a
scoreless board. Tile 0 is the one that curls arcs back toward each other and
tile 2 has a straight-through to carry chains across distance, and the mix of
the two is what makes closure common without making it automatic. It also
happens to be a lovely property for a physical edition: two shapes, simple
ratio.

Then there's the finding that shaped scoring, and which I'd call the single
most important result of the early work:

> No matter how you reweight the deck, **roughly 65–80% of every closed loop
> is the minimal one** — a length-3 loop curling around a single vertex.

That's not a tuning problem. It's a property of hexagonal geometry. Three
tight turns meeting at a corner is by far the easiest way for arcs to close,
and it stays the dominant case under every deck I tried. I spent a while
trying to engineer around it — "spacer" tiles meant to push closures apart —
and the experiment came back not just negative but backwards: spacers
concentrated *more* probability onto the minimal case. That path is closed.

So the design accepts it. Scoring by **enclosed area** rather than by loop
count means minimal loops are cheap, fast, incidental points, and the real
decision weight sits on the rarer long loops that wrap actual territory.
Scoring purely by loop count, tested head-to-head, was consistently the
weakest separator of player skill — it rewards exactly the thing the geometry
hands out for free.

## Is any of this actually strategic?

Here's the question that ate most of the project.

There's an obvious way to play: at every turn, look at all your legal
placements, and make the one that scores the most right now. Call it greedy.
It's the first thing anyone would code and roughly what a new player does.

Greedy is *very* good. Against a random player it wins 100% of games in
eleven of twelve configurations I tested. Fine — beating random is a low bar.
The worrying part was what happened when I tried to beat greedy.

**Attempt one: a smarter heuristic.** I wrote a "denial" agent that weighs
blocking the opponent's future scoring against its own immediate gain — the
obvious strategic idea, the thing you'd tell a new player to start thinking
about. Against greedy it won between 41.7% and 58.3% of games. A coin flip.
Interestingly, denial-vs-denial games had consistently *higher* margins than
greedy-vs-greedy, so it genuinely plays differently. It just doesn't play
better.

**Attempt two: self-play reinforcement learning.** I built a proper batched
RL environment for this — the whole apparatus, a readable single-instance
reference implementation differentially tested bit-for-bit against a
vectorized tensor version, twelve invariants checked, the works. Then I
trained a policy against itself.

It learned to beat random. It never learned to beat greedy. Across six
checkpoints, its win rate against greedy was 0.000, 0.000, 0.000, 0.000, and
finally **0.004**, with score margins sitting around −25. The diagnostic was
brutal: the trained policy scored an average of 0.46 points per game against
greedy — the most it ever scored in any single game was 5 — while greedy
scored 28.6. The same policy scored 13.6 against random. It hadn't learned
nothing. It had learned something that evaporated on contact with a
competent opponent.

**Attempt three: train directly against greedy.** Cold-start didn't move off
0% and was *losing ground against random* while it did so. Warm-starting from
the self-play checkpoint worked better and produced the only real progress of
the whole RL effort:

| Updates | Win vs greedy | Margin |
|---|---|---|
| 20 | 0% | −26.6 |
| 60 | 0% | −20.4 |
| 120 | 0.8% | −14.2 |
| 150 | 5.5% | −12.8 |
| 225 | **8.6%** | −9.5 |
| 300 | 4–8%, oscillating | −9.5, flat |

Which is a plateau. It climbs to a noisy 4–9% band and stops, with the margin
pinned around −9.5 and refusing to move.

And the diagnostic on that plateaued checkpoint is the detail I keep coming
back to. Most of the margin improvement came from **suppressing greedy's
score** (28.6 → 12.8), not from the policy scoring more itself (0.46 → 3.83).
It had learned to be a nuisance. It had not learned to play.

At this point I had three independent methods all saying "greedy is
approximately unbeatable," and two very different explanations available:

- **H1:** The game's strategic ceiling really is that low. Greedy is near
  optimal and there's nothing to find.
- **H2:** All three methods failed for their own reasons and none of them
  constitutes evidence about the game.

H2 is uncomfortable because it's unfalsifiable-sounding — "my methods were
bad" is what you say when you don't want to accept a result. But it had
specific support. Self-play's two seats are the *same* evolving policy, so it
only ever faces an equally mediocre copy of itself; nothing forces it to
become as mechanically sharp as "always grab the best available score." This
is a well-known failure mode, and it's why serious self-play systems mix in
fixed scripted opponents and past-checkpoint exploiters rather than trusting
self-play alone.

The way to settle it was obviously not to train harder.

## The thing that worked had no learning in it

A rollout policy. At each turn: rank your legal moves by immediate score
(that ranking *is* the greedy heuristic), take the top **K**, and for each
one, simulate the rest of the game with both sides playing plain greedy. Keep
whichever candidate produced the best final margin.

That's it. It's one step of policy improvement over greedy, it's about thirty
lines, and there is no neural network anywhere in it.

At K=8, simulating every candidate all the way to the end of the game:

```
n=50   win 1.000   draw 0.000   scores 27.98 / 5.34   margin +22.64
```

Fifty out of fifty. At that sample size, a perfect record against a true win
probability below ~90% is a sub-1% event. This is not noise.

So: **H2.** Every earlier "greedy looks unbeatable" result was a failure of
the method, and each failed differently. The denial heuristic's one-ply
lookahead was too shallow to see it. Self-play stalled in mutual mediocrity
before ever reaching greedy-level competence. And direct RL-vs-greedy
plateaued because sparse terminal reward plus a flat, spatially-blind
observation made credit assignment too hard — the network sees a vector, not
a board, and has no representation of adjacency or loop topology to hang a
strategy on.

Two honest caveats on that +22.64, because a number that clean deserves
suspicion. First, that configuration uses full information during its
rollouts — it sees the opponent's exact hand. So it's an upper bound on
exploitability under perfect information, not proof that an
information-respecting agent gets there. Second, it costs about **29 seconds
per game** of pure CPU, which makes it a research instrument rather than
something a person can play against.

The genuinely useful result came from sweeping the parameters. A far cheaper
configuration — **K=3, rolling forward only 2 plies** — still wins **93%** of
games against greedy at **30 milliseconds per move**. That's roughly a
thousand times cheaper for most of the strength.

Three things fell out of that sweep that I didn't expect:

1. **More search is not reliably better.** The K=6/depth=6 configuration
   scored the *lowest* win rate of any search config while costing ten times
   what the cheap one does.
2. **Depth is not where the strength lives.** Raising K beats raising depth.
   Depth does buy margin — +12.4 at eight plies versus +10.2 at two — so it
   makes the bot win *harder*, not more often.
3. **Read that sweep for magnitudes, not rankings.** At 30 games per
   configuration the standard error on a 90% win rate is about 5.5 points, so
   everything in the 80–93% band is statistically indistinguishable. The one
   solid, enormous difference is greedy at 47% versus *any* search at 80–93%.

The practical upshot for the game: no distillation, no neural network, no ML
runtime in the shipped product. The strong opponent is thirty lines of search.

## Building the actual game

The research all lived in Python. The game is Godot 4 and GDScript, which
means the entire rules engine — legality, loop detection, scoring — had to be
ported to a second language, where it could quietly diverge.

So it didn't get to be trusted. A Python script dumps complete games as
"packs": every move, plus the points gained, the exact set of loops closed,
and both scores at every single step. A headless Godot test replays each pack
through the new engine and asserts it matches at every step. Only when that
passed did anything get built on top of it. This is the same discipline that
validated the reference and vectorized implementations of the RL environment
against each other, and it's the reason I could later change rendering and
UI freely without wondering whether I'd broken the rules underneath.

Loop detection was the one genuinely tricky port. A loop is a connected
component of arcs in which every arc has both of its ports matched to a
neighbor — no open ends — which is a union-find problem. Enclosed area is a
ray-cast crossing-parity count. The tempting optimization is to only examine
the neighborhood of the tile you just placed, and an early version of the
Python code did exactly that, and it was silently wrong: it under-reported
whenever one tile's arcs belonged to two independent loops at once. The
failing case was a single placement closing a loop of length 3 and a loop of
length 18 simultaneously. The board has at most 111 arcs on it. Recomputing
everything from scratch on every placement is instant. Don't be clever.

Two other things bit, both worth knowing if you're doing this:

**Arcs.** The obvious way to draw a circular arc is to hand a curve primitive
a start point, an end point, and a radius. Don't. SVG-style arc commands pick
the reflected circle center and cheerfully throw the arc *outside* the hex
cell it belongs to. The fix is to compute the tangent-continuous arc yourself
and sample it into explicit points — nineteen per arc, in this case — and
draw a polyline. Straight-through spans get drawn as an actual straight line
rather than an arc of infinite radius.

**The browser.** The game targets WebAssembly, and in WASM the main thread
must never block. A bot that thinks for a full second doesn't show a spinner;
it freezes the tab. No input, no rendering, nothing. This is the single
biggest constraint on the search budget, and it's why the bot yields between
candidates (`await get_tree().process_frame`) rather than running its search
to completion in one call. That constraint showed up again at deployment: the
Godot web export defaults to real thread support, which requires
cross-origin-isolation headers that itch.io's embed doesn't reliably serve.
Turning thread support off makes the build header-independent, and costs
nothing, because the search was never using threads — it was yielding across
frames.

A note on process, since it's the part people ask about. Most of this build
was done with an AI agent driving the tooling directly: not generating code
blind, but launching the Godot editor, running the project, taking
screenshots, reading the runtime logs, and serving the web build to a real
browser to click through. That distinction matters more than the code
generation does. The autoload bug — a development-only script that was
filtered out of the web export while still being registered at startup, which
broke *all input* on web with a single log line as the only symptom — is not
a bug you find by reading code. It's a bug you find by loading the page and
discovering that nothing clicks.

## Where it landed

It's playable: hot-seat for two people at one screen, a bot ladder, free
placement as a rule variant, replays, and it runs in a browser.

And it's genuinely hard. I lose to the medium bot — the 30-millisecond one —
regularly.

Getting there required solving a problem the sweep had already flagged: every
search configuration beats greedy around 90% of the time, and greedy is
roughly "a reasonable human." Turning search up or down barely moves that,
which means search strength is useless as a difficulty dial. The ladder would
have varied in *how badly* it beat you rather than *whether* it did.

So difficulty isn't a search parameter here. It's a **greedy-slip**: a fixed
probability that the bot deliberately plays the plain greedy move instead of
the one its search chose. Medium runs at 35% slip on top of a K=3, 2-ply
search. Hard runs the same search with no slip at all. It's a much better
handicap axis than search depth because it degrades the bot's *judgment*
rather than its *reach* — a slipping bot still plays coherently, it just
periodically fails to see the trap it's walking into, which is a recognizably
human way to lose.

What's still open is whether 35% is the right number. It was picked by
reasoning, not measured — I have exactly one playtester's worth of evidence
that Medium is fun to lose to, and that playtester is me. Tuning it properly
is a playtesting problem rather than a coding one, which makes it the most
interesting thing left and the thing least likely to get done by staring at
the code.

Which is a nice inversion of where this started. The first half of the
project was a long fight to build something that could beat the obvious
strategy. The second half was teaching it how to lose convincingly.

---

*Play it: [orbitope.itch.io/hex-truchet](https://orbitope.itch.io/hex-truchet)*
*Code: [github.com/Orbitope/hextruchet](https://github.com/Orbitope/hextruchet)*
