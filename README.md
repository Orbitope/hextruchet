# Hex Truchet

A hex-grid Truchet-tile placement game, developed from geometry research
through a trained bot to a playable Godot game.

Players alternate placing hexagonal tiles carrying curved paths onto a
shared board. Adjacent tile edges connect, and closing a loop scores points
for whoever completes it. The core design question driving this project was
whether a simple greedy strategy is already close to optimal play, or
whether there's real strategic depth to exploit — see
`hex_truchet/spec.md` for the full rules.

## Play it

The Godot game (`hextruchet-godot/`) is the playable deliverable, with
hot-seat, vs-bot, and replay modes.

To try the web build locally:

```bash
cd hextruchet-godot/build/web
python3 -m http.server 8000
```

Then open `http://localhost:8000/index.html`. (If `build/web/` doesn't
exist yet, export it first from the Godot editor, or headless via
`godot --headless --export-release "Web" build/web/index.html`.)

To run in the editor: open `hextruchet-godot/` in Godot 4.7+.

## Repo layout

- **`hex_truchet/`** — the validated Python game engine and RL environment
  (rules, scoring, loop detection), built on the `simulacrum` environment
  framework. This is the reference implementation the Godot rules engine
  was differentially tested against.
- **`hextruchet-godot/`** — the playable Godot 4 game: native GDScript rules
  engine, renderer, hot-seat/vs-bot/replay modes, and search-based bots.
- **`training/`** — bot development and evaluation: the tunable rollout-search
  bots (`bots.py`), sweeps against a greedy baseline, and RL training
  experiments.
- **`viz/`** — arc-rendering math and an HTML board viewer used to validate
  tile geometry before it was ported into Godot.
- **`hextruchet_first_pass/`** — early-stage geometry, scoring, and agent
  research that shaped the final ruleset.

## Status

Core gameplay (hot-seat, vs-bot, and free-placement modes; rule variants;
tile rendering) is implemented and has been verified working, including in
a browser-based web export. Difficulty balancing against human players is
still unverified — the bot presets are currently tuned for strength, not
for a fair fight against a casual player.
