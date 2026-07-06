# Lorenz Attractor — Design Spec

**Date:** 2026-07-06
**Status:** Approved (design review)
**Stack:** Python, Taichi (CPU), Pygame

## 1. Overview

A visually appealing, screensaver-like physics simulation of the Lorenz Attractor from chaos theory. A single particle traces the butterfly-shaped attractor from one starting point, drawing itself over time on a pure-black canvas. Color encodes the particle's z-height via a perceptually-uniform viridis palette, so the two lobes of the attractor naturally read in different hues. The camera slowly auto-rotates around the completed shape for a cinematic, hypnotic effect. Minimal UI, no parameter tweaking — a pure aesthetic showcase.

## 2. Goals & Non-Goals

**Goals**
- Render a beautiful, smooth Lorenz attractor at 60fps on a CPU-only machine (no dedicated GPU).
- Single particle grows the trail over time; full butterfly fills in then freezes.
- Z-height → viridis color mapping (purple → green), making the two wings visually distinct.
- Auto-rotating cinematic camera; minimal interaction (pause/reduced-motion only).
- Clean thin antialiased lines on pure black.

**Non-Goals**
- Interactive parameter tweaking (σ, ρ, β sliders).
- Mouse-controlled orbit/zoom.
- Educational overlays, labels, or chaos-theory explanations.
- Multiple diverging trails / butterfly-effect visualization.
- Glow/bloom effects (clean lines only).
- GPU acceleration.

## 3. Architecture

The app splits into focused modules, each with one job:

### `simulation.py` — Physics core
- Owns the Lorenz ODE system (σ, ρ, β), the initial state, and the RK4 integrator.
- Exposes a Taichi kernel that advances the single particle via RK4 and appends its 3D position to a growing `ti.field`.
- Runs **once at startup** to pre-compute the full trail up to the point cap, then hands off the complete 3D point array.
- Integration is inherently serial (one particle's sequential trajectory), so the kernel is not parallelized across timesteps — but it runs only once, so it's effectively instantaneous.

### `render.py` — Per-frame projection
- A Taichi kernel takes the full 3D trail + the current camera angle, rotates around the Y axis, applies perspective projection, and outputs a 2D `(x_screen, y_screen, z_depth)` array to numpy.
- Also maps each point's z-height through the viridis colormap LUT to an RGB value.
- All points processed in one parallel Taichi kernel each frame.

### `camera.py` — Auto-rotating cinematic state
- Owns the yaw angle and optional vertical-bob phase.
- Each frame increments the view angle at a slow constant rate; also applies a gentle sinusoidal vertical tilt (the "bob") so the view orbits *around* the shape rather than flatly spinning past it. The bob is ON by default; the `R` key disables it via reduced-motion mode.
- No input handling — it just drifts.

### `draw.py` — Pygame rendering pass
- Takes the projected 2D coords + colors and draws clean antialiased line segments between consecutive points onto a pure-black surface.
- Sorts segments back-to-front by average depth so nearer parts correctly overlap farther parts as the camera rotates.

### `main.py` — Orchestrator
- Initializes Taichi (`ti.init(arch=ti.cpu)`), runs the startup integration, owns the Pygame window + main loop, and wires simulation → camera → render → draw each frame.
- Handles the minimal key controls.

## 4. Physics & Integration

The Lorenz system with canonical parameters:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

σ = 10, ρ = 28, β = 8/3
```

- **Initial state:** Single particle at `(x₀, y₀, z₀) = (1.0, 1.0, 1.0)` — slightly off-origin so it falls into the attractor naturally.
- **Transient:** Discard the first ~500 steps so the trajectory has settled onto the attractor before recording begins.
- **Integrator:** 4th-order Runge-Kutta (RK4). Euler would drift and corrupt the butterfly shape over a long run; RK4 preserves accuracy. Fixed timestep `dt = 0.01`.
- **Point cap:** **~12,000 points** (one constant at the top of `simulation.py`). Tuned to stay smooth at 60fps under Approach B's per-frame Pygame drawing of 12k antialiased segments. If target framerate can't be held on the host CPU, lower this single constant (e.g. 8k).
- **When full:** Stop appending. The 3D trail array is frozen; no further integration. The render camera keeps rotating. Per-frame cost becomes purely projection + draw.
- **Bottleneck note:** Since a single particle is traced sequentially, integration itself is fast and not the bottleneck. The real per-frame cost is projection (parallelized in Taichi) + drawing (Pygame line loop).

## 5. Rendering Approach (Approach B)

Taichi handles integration and 3D→2D projection in parallel; Pygame draws the resulting antialiased line segments. This trades maximum point count for nicer built-in line quality.

### 3D → 2D projection (Taichi kernel, per frame)
- Rotate the full trail around the Y axis by the current camera yaw angle θ. This is a simple yaw rotation that keeps the attractor's wingspread visible as it turns.
- Apply perspective projection with a fixed focal length: `x_screen = focal * x_rot / z_rot`, `y_screen = focal * y_rot / z_rot`, scaled and offset to window center.
- Output per point: `(sx, sy, depth)` where `depth = z_rot`, used for draw ordering.
- The entire point array is processed in one parallel Taichi kernel, copied to a numpy array for the draw pass.

### Viridis colormap (z-height → RGB)
- Precompute a 256-entry lookup table mapping normalized z-height (0→1 across the attractor's observed z-range) to viridis RGB (purple → magenta → cyan → green).
- In the projection kernel, each point's raw z maps to a LUT index → that point's color. Cheap and consistent frame-to-frame.
- The two lobes, sitting at different heights, naturally land in different hue ranges — wing separation reads instantly.

### Camera (auto-rotating cinematic)
- Yaw angle θ increases at a slow constant rate per frame (~0.2°/frame → full revolution every few minutes). Smooth and hypnotic.
- Optional gentle vertical bob: a small sinusoidal tilt (ON by default) so the view isn't a flat equatorial spin — gives a sense of orbiting around the shape. Disabled by reduced-motion mode (`R`).
- Fixed zoom/distance chosen so the full butterfly fills most of the window with a small margin.

### Drawing (Pygame)
- Sort segments back-to-front by average depth (numpy `argsort` on mean segment depth) so nearer parts overlap farther parts correctly during rotation.
- Draw consecutive point pairs as antialiased lines (`pygame.draw.aaline`) in each point's z-color on a pure-black surface.
- Clear to black each frame. Because the trail is frozen but the camera moves, the entire trail is re-projected and re-drawn every frame — this is how a rotating camera works with a frozen trail, and the 12k cap keeps it at 60fps.

## 6. Main Loop, Window & Controls

### Window
- Resolution: **1280×800**, pure black background. Cinematic widescreen without straining CPU projection.
- Pygame `DOUBLEBUF` + `vsync=1` for smooth, tear-free presentation.

### Startup sequence (runs once)
1. Init Taichi with `ti.init(arch=ti.cpu)`.
2. Init Pygame display + clock.
3. Run the RK4 integration kernel → produce the frozen 3D trail array (~12k points).
4. Precompute the viridis 256-entry LUT.

### Main loop (per frame)
1. Pump Pygame events (quit / window-close only).
2. `camera.update(dt_frame)` → new yaw angle θ (+ bob phase).
3. `render.project(trail_3d, θ)` → Taichi kernel → 2D `(sx, sy, depth)` + per-point color → numpy.
4. Sort segments by depth (numpy argsort on average depth).
5. Clear surface to black.
6. `draw.segments(...)` → `aaline` per segment in z-color, back-to-front.
7. `pygame.display.flip()`.
8. `clock.tick(60)` to cap at 60fps.

### Controls (minimal)
- `ESC` / window close → quit.
- `SPACE` → pause/resume camera rotation (hold a view you like).
- `R` → toggle reduced-motion (freeze the bob, halve the yaw rate) for a calmer look.
- No parameter tweaking, no mouse orbit — per the aesthetic-showcase decision.

### Framerate target
60fps. The point cap (`~12,000` in `simulation.py`) is the single knob to lower if projection+draw can't hold 60 on the host CPU.

## 7. Performance Strategy (CPU-only)

- Taichi compiled CPU kernels run parallel across cores for the per-frame projection (12k points) — this is the parallelizable part and stays fast.
- The Pygame per-segment `aaline` loop is the bottleneck (12k Python-level calls/frame). The 12k cap is chosen to keep this inside the 60fps budget. It is the one tunable constant.
- Integration runs once at startup and never repeats, so it imposes no per-frame cost.
- No pixel-buffer rasterizer is written in Taichi (that would be Approach A); we lean on Pygame's built-in antialiased line drawing for quality, accepting a lower point ceiling.

## 8. Dependencies

- **Python** (3.10+)
- **taichi** — CPU-parallel compute kernels for integration + projection.
- **pygame** (or `pygame-ce`) — window, event loop, antialiased line drawing, display flip.
- **numpy** — array handoff between Taichi kernels and Pygame, depth sorting.

## 9. File Layout

```
lorenz-attractor/
├── main.py            # Orchestrator: window, loop, controls, wiring
├── simulation.py      # Lorenz ODE + RK4 integrator (Taichi kernel, runs once)
├── render.py          # Per-frame 3D→2D projection + viridis colormap (Taichi kernel)
├── camera.py          # Auto-rotating cinematic camera state
├── draw.py            # Pygame antialiased segment drawing + depth sort
└── docs/superpowers/specs/2026-07-06-lorenz-attractor-design.md
```

## 10. Open Items / Tuning Knobs

- **Point cap** (`~12,000`): lower if 60fps can't be held; raise only if measured headroom exists.
- **Camera yaw rate** (`~0.2°/frame`) and **bob amplitude**: aesthetic tuning during implementation.
- **Focal length / zoom**: chosen during implementation so the butterfly fills the window with a small margin.
- **Viridis LUT exact stops**: pulled from a standard viridis definition at implementation time.
