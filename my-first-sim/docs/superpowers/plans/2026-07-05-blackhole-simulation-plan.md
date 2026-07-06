# Black Hole Simulation Implementation Plan

## Source
Design spec: `docs/superpowers/specs/2026-07-05-blackhole-simulation-design.md`

## Goal
Implement a real-time, pannable, Interstellar-like black hole visualization in Python + Taichi on CPU.

## Milestones

### Milestone 1: Project skeleton
- [ ] Add `pygame` to `requirements.txt`.
- [ ] Create `main.py` with `ti.init(arch=ti.cpu)` and Pygame window setup.
- [ ] Create 512×512 Taichi vector field `pixels`.
- [ ] Implement minimal event loop that calls a blank kernel and exits on close.
- [ ] Verify window opens and closes cleanly.

### Milestone 2: Static camera ray tracer
- [ ] Add camera fields: `cam_dist`, `cam_az`, `cam_el`.
- [ ] Implement a kernel that maps each pixel to a ray direction in world space based on camera angles.
- [ ] Add a simple sphere intersection test (radius = event horizon).
- [ ] Color pixels: black inside sphere, dark gray gradient outside for sanity check.
- [ ] Verify sphere looks like a round black disk from multiple angles.

### Milestone 3: Gravitational lensing
- [ ] Replace sphere test with Schwarzschild geodesic ray march.
- [ ] Photon state: position + velocity in Cartesian coordinates.
- [ ] Step using Schwarzschild acceleration term derived from the Euler-Lagrange equations.
- [ ] Terminate rays that cross event horizon (r ≤ 2M) or escape (r ≥ r_far).
- [ ] Visual check: background should show distorted edge of black hole.

### Milestone 4: Accretion disk
- [ ] Detect ray crossing the equatorial plane (`z = 0`) during integration.
- [ ] On crossing, check radius is within `[r_in, r_out]`.
- [ ] Assign color from radial gradient: hot inner → cooler outer.
- [ ] Allow multiple crossings so lensed back-side ring appears.
- [ ] Tune `r_in`, `r_out`, step size, and max steps for visual quality and speed.

### Milestone 5: Interaction
- [ ] Track left-mouse drag in Pygame event loop.
- [ ] Update `cam_az` / `cam_el` fields from drag deltas.
- [ ] Clamp elevation to ±80° to avoid degenerate views.
- [ ] Optional: scroll wheel adjusts `cam_dist`.
- [ ] Display FPS counter in window title or overlay.

### Milestone 6: Polish and validation
- [ ] Ensure ≥30 fps at 512×512 on CPU.
- [ ] If slow, reduce step count or resolution.
- [ ] Clean code, remove debug colors.
- [ ] Add README note about running on CPU.

## File Layout
```
my-first-sim/
├── main.py
├── requirements.txt
└── docs/superpowers/
    ├── specs/2026-07-05-blackhole-simulation-design.md
    └── plans/2026-07-05-blackhole-simulation-plan.md
```

## Key Implementation Details

### Taichi kernel outline
```python
@ti.kernel
def render_kernel(
    cam_dist: ti.f32,
    cam_az: ti.f32,
    cam_el: ti.f32,
):
    for i, j in pixels:
        # ray direction from camera through pixel
        ray = compute_ray(i, j, cam_az, cam_el)
        pos = camera_position(cam_dist, cam_az, cam_el)
        color = trace(pos, ray)
        pixels[i, j] = color
```

### Geodesic trace loop (scalar, inside kernel)
- Fixed step `dt`.
- Update velocity `v += a(pos, v) * dt`.
- Update position `pos += v * dt`.
- After each step:
  - if `r < 2.1 * M`: return black
  - if sign(z) flipped and `r_in < r < r_out`: return disk color
  - if `r > r_far`: return black (background)

### Disk color function
```python
def disk_color(r):
    t = (r - r_in) / (r_out - r_in)
    return mix(inner_color, outer_color, t)
```

## Risk Mitigation
| Risk                          | Action                                       |
|-------------------------------|----------------------------------------------|
| FPS too low                   | Reduce `max_steps`, increase `dt`, or drop to 400×400. |
| Kernel compile slow           | Keep fields small; avoid Python loops inside kernel. |
| Black hole not round          | Verify ray generation uses correct aspect ratio and FOV. |
| Disk appears only on one side | Allow crossings from both +z and -z; check z sign flip. |

## Definition of Done
- `main.py` launches, shows black hole silhouette and lensed disk.
- Mouse drag orbits smoothly.
- FPS stays at or above 30 at 512×512 on CPU.
- `requirements.txt` lists all dependencies.

## Next Step
Begin Milestone 1: project skeleton.
