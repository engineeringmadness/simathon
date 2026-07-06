import sys

import numpy as np
import pygame
import taichi as ti

# --- config ------------------------------------------------------------------
WIDTH, HEIGHT = 512, 512
FPS_LIMIT = 60

# constants
PI = 3.14159265359
TWO_PI = 6.28318530718

# black hole parameters (geometrized units, M = 1)
M = 1.0
R_S = 2.0 * M               # Schwarzschild radius
R_IN = 2.6 * M              # inner disk edge
R_OUT = 7.0 * M             # outer disk edge
R_FAR = 100.0 * M           # escaped to background

CAM_DIST = 18.0
FOV = 1.2                   # radians vertical

# integration parameters
DT = 0.4
MAX_STEPS = 40

# particle ring parameters
N_RINGS = 30000
PER_RING = 3000           # total particles = 800000
PER_RING_F = 2000.0
# spacing between particles in a ring
ANGULAR_SPACING = TWO_PI / PER_RING_F
# 3x spacing -> particles overlap so disk fills in, but clumps remain visible
ANGULAR_HALF_WIDTH = 5.0 * ANGULAR_SPACING  # particles overlap so disk is visible

# colors
INNER_COLOR = ti.Vector([1.0, 0.95, 0.7])   # white-yellow
OUTER_COLOR = ti.Vector([1.0, 0.2, 0.0])    # red-orange

# volumetric disk parameters
DISK_SCALE_HEIGHT = 0.12 * M   # vertical scale height of disk
DISK_DENSITY_FALLOFF = 0.6     # radial density profile exponent

# --- taichi init -------------------------------------------------------------
ti.init(arch=ti.cpu, advanced_optimization=True)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# camera state
azimuth = ti.field(dtype=ti.f32, shape=())
elevation = ti.field(dtype=ti.f32, shape=())
cam_dist = ti.field(dtype=ti.f32, shape=())

azimuth[None] = 0.0
elevation[None] = 1.47079632679  # edge-on: disk horizontal like a table edge
cam_dist[None] = CAM_DIST

# particle state
ring_radius = ti.field(dtype=ti.f32, shape=N_RINGS)
ring_base_phi = ti.field(dtype=ti.f32, shape=N_RINGS)
ring_omega = ti.field(dtype=ti.f32, shape=N_RINGS)
particle_offset = ti.field(dtype=ti.f32, shape=(N_RINGS, PER_RING))

# --- helpers -----------------------------------------------------------------
@ti.func
def length(v: ti.template()) -> ti.f32:
    return ti.sqrt(v.dot(v))


@ti.func
def normalize(v: ti.template()) -> ti.template():
    return v / length(v)


@ti.func
def sgn(x: ti.f32) -> ti.f32:
    r = 0.0
    if x > 0.0:
        r = 1.0
    elif x < 0.0:
        r = -1.0
    return r


@ti.func
def blackbody_color(t: ti.f32) -> ti.Vector:
    # approximate blackbody RGB from normalized temperature t in [0, 1]
    # t=1 hot (blue-white), t=0 cool (red)
    # simple polynomial fit, clamped
    r = 1.0
    g = t * (0.8 + 0.2 * t)
    b = t * t * (0.4 + 0.6 * t)
    return ti.Vector([r, g, b])


@ti.func
def disk_color(r: ti.f32) -> ti.Vector:
    t = (r - R_IN) / (R_OUT - R_IN)
    t = ti.max(0.0, ti.min(1.0, t))
    # invert: inner edge hot, outer edge cool
    return blackbody_color(1.0 - t)


@ti.func
def disk_brightness(r: ti.f32) -> ti.f32:
    # inner disk brighter, outer disk fades
    t = (r - R_IN) / (R_OUT - R_IN)
    t = ti.max(0.0, ti.min(1.0, t))
    return 1.0 + 3.0 * (1.0 - t) * (1.0 - t)


@ti.func
def hash2(k: ti.i32, i: ti.i32) -> ti.f32:
    x = ti.sin(ti.cast(k, ti.f32) * 127.1 + ti.cast(i, ti.f32) * 311.7) * 43758.5453
    return x - ti.floor(x)


@ti.kernel
def init_particles():
    dr = (R_OUT - R_IN) / ti.cast(N_RINGS - 1, ti.f32)
    for k in range(N_RINGS):
        r = R_IN + dr * ti.cast(k, ti.f32)
        ring_radius[k] = r
        ring_base_phi[k] = 0.0
        ring_omega[k] = ti.sqrt(M / (r * r * r))
        for i in range(PER_RING):
            # even spacing plus small random jitter
            jitter = (hash2(k, i + 1) - 0.5) * 0.3
            particle_offset[k, i] = (ti.cast(i, ti.f32) / PER_RING_F + jitter) * TWO_PI


@ti.kernel
def update_particles(dt: ti.f32):
    for k in range(N_RINGS):
        ring_base_phi[k] += ring_omega[k] * dt


# --- ray tracing kernel ------------------------------------------------------
@ti.kernel
def render():
    cam_pos = ti.Vector([
        cam_dist[None] * ti.sin(elevation[None]) * ti.cos(azimuth[None]),
        cam_dist[None] * ti.cos(elevation[None]),
        cam_dist[None] * ti.sin(elevation[None]) * ti.sin(azimuth[None]),
    ])

    forward = normalize(-cam_pos)
    # avoid gimbal lock when looking down disk normal
    ref = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(forward.dot(ref)) > 0.99:
        ref = ti.Vector([1.0, 0.0, 0.0])
    right = normalize(forward.cross(ref))
    up = right.cross(forward)

    dr = (R_OUT - R_IN) / ti.cast(N_RINGS - 1, ti.f32)

    for i, j in pixels:
        u = (ti.cast(i, ti.f32) + 0.5) / WIDTH - 0.5
        v = (ti.cast(j, ti.f32) + 0.5) / HEIGHT - 0.5
        u *= WIDTH / HEIGHT

        ray_dir = normalize(forward + u * right * ti.tan(FOV * 0.5) + v * up * ti.tan(FOV * 0.5))

        color = ti.Vector([0.0, 0.0, 0.0])

        pos = cam_pos
        vel = ray_dir
        r2 = pos.dot(pos)
        z_sign = sgn(pos[2])

        for _ in range(MAX_STEPS):
            r = ti.sqrt(r2)
            if r < R_S * 1.05:
                break
            if r > R_FAR:
                break

            dt = 0.2 * r
            if dt < 0.05:
                dt = 0.05
            if dt > 1.0:
                dt = 1.0

            rv = pos.dot(vel)
            v2 = vel.dot(vel)
            factor = -3.0 * M / (r2 * r2 * r)
            acc = factor * (pos * v2 - vel * rv)

            vel = vel + acc * dt
            pos = pos + vel * dt

            r2 = pos.dot(pos)
            new_z_sign = sgn(pos[2])

            # particle disk crossing check
            if z_sign * new_z_sign < 0.0:
                prev_z = pos[2] - vel[2] * dt
                alpha = ti.abs(pos[2]) / (ti.abs(pos[2]) + ti.abs(prev_z))
                cross_pos = pos - vel * dt * alpha
                cross_r = ti.sqrt(cross_pos[0] ** 2 + cross_pos[1] ** 2)
                cross_phi = ti.atan2(cross_pos[1], cross_pos[0])

                # nearest ring
                k_float = (cross_r - R_IN) / dr
                k = ti.cast(k_float + 0.5, ti.i32)
                if 0 <= k < N_RINGS:
                    r_ring = ring_radius[k]
                    # find nearest particle in this ring
                    phi_rel = cross_phi - ring_base_phi[k]
                    # wrap to [0, 2pi)
                    while phi_rel < 0.0:
                        phi_rel += TWO_PI
                    while phi_rel >= TWO_PI:
                        phi_rel -= TWO_PI
                    idx_float = phi_rel / TWO_PI * PER_RING_F
                    idx = ti.cast(idx_float + 0.5, ti.i32) % PER_RING
                    p_phi = particle_offset[k, idx]
                    # wrap difference
                    diff = phi_rel - p_phi
                    while diff < -PI:
                        diff += TWO_PI
                    while diff > PI:
                        diff -= TWO_PI
                    # particle size scales with radius so inner ones aren't huge
                    size = 2.0 * dr
                    radial_err = ti.abs(cross_r - r_ring) / size
                    angular_err = ti.abs(diff) / ANGULAR_HALF_WIDTH
                    if radial_err < 1.0 and angular_err < 1.0:
                        # soft particle core: bright center, darker edges
                        falloff = (1.0 - radial_err) * (1.0 - angular_err)
                        falloff = falloff * falloff  # sharpen
                        color = disk_color(r_ring) * falloff * 2.0
                        break

            # volumetric disk glow: accumulate brightness when close to midplane
            z_abs = ti.abs(pos[2])
            r_xy = ti.sqrt(pos[0] ** 2 + pos[1] ** 2)
            if z_abs < DISK_SCALE_HEIGHT * 3.0 and R_IN <= r_xy <= R_OUT:
                t = (r_xy - R_IN) / (R_OUT - R_IN)
                radial_falloff = ti.pow(1.0 - t, DISK_DENSITY_FALLOFF)
                height_falloff = ti.exp(-z_abs / DISK_SCALE_HEIGHT)
                contrib = disk_color(r_xy) * disk_brightness(r_xy) * radial_falloff * height_falloff * 0.08
                color += contrib
                # clamp early to avoid blowout
                color = ti.min(color, ti.Vector([2.0, 2.0, 2.0]))

            z_sign = new_z_sign

        pixels[i, j] = color


# --- pygame loop --------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Black Hole Simulation")
    clock = pygame.time.Clock()

    init_particles()

    dragging = False
    last_mouse = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                last_mouse = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                last_mouse = event.pos

                azimuth[None] += dx * 0.01
                elevation[None] += dy * 0.01
                # elevation 0 = face-on, pi/2 = edge-on
                if elevation[None] > 1.55:
                    elevation[None] = 1.55
                if elevation[None] < 0.05:
                    elevation[None] = 0.05
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist[None] *= 1.0 - event.y * 0.1
                cam_dist[None] = max(5.0, min(50.0, cam_dist[None]))

        update_particles(0.008)
        render()

        img = pixels.to_numpy()
        # reinhard-ish tone map so bright inner disk doesn't clip to white
        img = img / (1.0 + img)
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 0, 2))
        surface = pygame.surfarray.make_surface(img)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        ms = clock.tick(FPS_LIMIT)
        fps = 1000.0 / max(ms, 1)
        pygame.display.set_caption(f"Black Hole Simulation - FPS: {fps:.0f}")


if __name__ == "__main__":
    main()
