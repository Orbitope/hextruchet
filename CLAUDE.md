# Hex Truchet — Project Instructions

## Project context

Hex Truchet is a tile-placement game research project. Stages 0–3 (geometry, scoring,
agent screening, RL environment) are complete and live in Python; see
`hextruchet_first_pass/HANDOFF.md` for the full history and findings.

**Stage 4 is a playable Godot game** — hot-seat, vs-bot, and replay modes.
The implementation plan is `viz/GODOT_GAME_PLAN.md` (rules engine, bots,
rendering, milestones). The guidelines below govern all Godot work.

Key facts to carry into the Godot build (details in the plan):
- The bots are **pure search, no ML** — one tunable rollout search with named
  difficulty presets (`training/bots.py::PRESETS` is the shared source of truth).
  Nothing needs exporting from Python for the game to have a strong opponent.
- Godot needs its **own native rules engine** (legality, loop detection, scoring),
  ported from `hex_truchet/spec.md` and differentially tested against the Python
  reference before any UI is built on top of it.
- Arc rendering math is already validated in `viz/build_viewer.py` — sample arcs into
  explicit points; never trust an SVG-style arc command (it picks the reflected centre
  and throws arcs outside the cell).

---

# Godot 4.5+ GDScript Web Project Guidelines

## 1. Project Scope & Architecture
- **Engine:** Godot 4.5 / 4.6
- **Language:** 100% GDScript. **DO NOT** generate, suggest, or scaffold any C# (`.cs`) files or `.csproj` environments. This project strictly targets WebAssembly (HTML5) and any C# injection will break the build.
- **Architecture:** Favor composition (Custom Resources for data, Node components for logic) over deep class inheritance. Keep logic decoupled from UI via signals.

## 2. MCP Server & Tool Directives
- **Zero Blind Guesses:** Do not hallucinate Node paths (e.g., `$Player/Sprite`). Before writing code that references nodes, use your MCP tools to parse the relevant `.tscn` file and verify the exact hierarchy.
- **Read the Traces:** If I instruct you to test a scene, or if an error occurs, proactively use your log-fetching tools (e.g., `get_errors`, `read_logs`) to pull the Godot runtime stack trace before proposing a fix.
- **Safe Scene Editing:** Remember that `.tscn` files are text-based serializations. If you modify them directly via file tools, ensure exact Godot resource syntax to prevent corruption. If using an engine-level MCP, use the live scene manipulation endpoints instead.
- **Visual Context:** If your MCP supports screen capture, ask me to trigger a screenshot if you need to understand the spatial layout of a UI or 2D/3D scene before writing positioning logic.

## 3. Strict Godot 4 Syntax (Anti-Hallucination)
Your training data is heavy with Godot 3. You must strictly enforce these Godot 4 rules:
- **Signals:** Use modern callables: `button.pressed.connect(_on_pressed)`. NEVER use string-based connections like `connect("pressed", self, "_on_pressed")`.
- **Tweens:** Use `create_tween()` bound to the node. NEVER use the deprecated `Tween` node or `interpolate_property()`.
- **Exports:** Use `@export var variable_name: type`. NEVER use the old `export` keyword.
- **Onready:** Use `@onready var`. NEVER use `onready var`.
- **Set/Get:** Use modern inline `set` and `get` properties. NEVER use `setget`.
- **Typing:** Enforce strict static typing universally (`var speed: float = 100.0`, `func move(delta: float) -> void:`). This is critical for catching errors before WASM compilation.

## 4. Web-Safe Constraints (HTML5/WASM)
- **File System:** Do not use OS-specific absolute paths. Rely entirely on `res://` for static assets and `user://` for saved data.
- **Performance:** Avoid heavy main-thread blocking `while` loops that will freeze the browser tab. Use `await get_tree().process_frame` if chunking operations over multiple frames is necessary.
