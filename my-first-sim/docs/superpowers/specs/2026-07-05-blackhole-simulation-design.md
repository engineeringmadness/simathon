# Black Hole Simulation Design

## Goal
Build a real-time, pannable, Interstellar-like black hole visualization using Python and Taichi on the CPU backend.

## Success Criteria
- Runs at **≥30 fps** at **512×512** on a CPU-only machine.
- Dragging the mouse **orbits the camera around the black hole**.
- Shows the iconic black-hole silhouette with a thin, gravitationally-lensed accretion disk.
- No GPU required; works in Taichi CPU mode.

## Out of Scope (YAGNI)
- Doppler / relativistic beaming.
- Star-field gravitational lensing.
- Volumetric / thick disk.
- Accretion disk animation / time evolution.
- Scroll-to-zoom (optional; added only if trivial).

## Architecture

```
main.py
  ├── Taichi fields: pixels (512×512 RGB), camera params
  ├── render_kernel()
  │     └── For every pixel: trace ray, detect disk intersection, write color
  └── event loop (Pygame)
        ├── Poll mouse drag → update camera azimuth/elevation in Taichi fields
        ├── Call render_kernel()
        └── Blit pixel buffer to screen
```

### Why this structure
- Pure Taichi kernel means no Python per-pixel overhead. Critical for CPU 30 fps.
- Pygame gives a window and mouse events without requiring a GPU renderer.
- Camera state lives in Taichi fields so the kernel reads it directly.

## Rendering Approach

### Physics model
- Static Schwarzschild black hole.
- Ray-march photon paths backward from camera through Schwarzschild geodesics.
- Stop early when ray:
  1. Crosses the disk plane within disk inner/outer radius → shade.
  2. Crosses photon sphere / event horizon → black.
  3. Escapes far enough → background color (black).

### Disk
- Geometrically thin, planar disk in the equatorial plane (`z = 0`).
- Inner radius `r_in` (e.g. 2.6 M) and outer radius `r_out` (e.g. 7.0 M).
- Color radial gradient: hot white-yellow near inner edge, cooling to orange-red near outer edge.
- Both front and back sides visible due to lensing.

### Geodesic integration
- Use reduced photon equations in spherical coordinates.
- Fixed step size chosen for stability and speed at 512×512.
- Early termination keeps cost low.

## Interaction

| Input              | Action                                          |
|--------------------|-------------------------------------------------|
| Left mouse drag    | Orbit camera: horizontal = azimuth, vertical = elevation |
| Scroll wheel       | Optional zoom (adjust camera distance)          |

- Elevation clamped to avoid gimbal-lock and looking directly edge-on.
- Orbit updates two scalar Taichi fields: `cam_azimuth`, `cam_elevation`.

## Libraries
- `taichi` (CPU backend, `ti.init(arch=ti.cpu)`)
- `numpy` for buffer handling
- `pygame` for window and input (added to `requirements.txt`)

## File Layout
```
my-first-sim/
├── main.py              # entry point + event loop + render kernel
├── blackhole.py         # physics constants + kernel (optional split)
├── requirements.txt     # taichi, numpy, pygame
└── docs/superpowers/specs/2026-07-05-blackhole-simulation-design.md
```

Initial implementation keeps everything in `main.py` unless it grows.

## Testing / Validation
- Visual smoke test: launch, drag, verify smooth orbit.
- FPS counter displayed on screen.
- No automated tests needed; output is visual.

## Risks & Mitigations
| Risk                              | Mitigation                                   |
|-----------------------------------|----------------------------------------------|
| CPU too slow at 512×512           | Reduce resolution to 400×400 or step count.  |
| Lensing looks wrong / unstable    | Visual check; tune step size and radii.      |
| Pygame install issues             | Use `pygame-ce` fallback if needed.          |

## Future Upgrades (ceiling)
- Doppler beaming once base hits 30 fps.
- Star-field background with ray bending.
- Animated disk texture.
