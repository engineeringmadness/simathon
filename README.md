# Simathon (Physics Simulation Workshop)

## Tech Stack
- Python
- Taichi Framework (for computation)
- PyGame (for actual animation because I don't have a dedicated GPU)

### Lorenz Attractor Simulator (Final submission)

The Lorenz attractor is a 3D visual shape that graphs the behavior of a chaotic, deterministic system. Originally created to model atmospheric air movements, 
it famously resembles a butterfly. It demonstrates how simple, fixed rules can produce complex, unpredictable patterns,

Edward Lorenz developed the system as a simplified mathematical model for atmospheric convection. He was attempting to model the way air moves when heated from below and cooled from above.

Functionality is quite straightforward. The simluation can be controlled using mouse actions.
- Left click and drag to pan the camera angle
- Right click to freeze the auto camera pan
- Scroll wheel to slow down / speed up the simulation.

<img width="1592" height="1017" alt="image" src="https://github.com/user-attachments/assets/d76f4001-c231-4be5-a174-237e8af3ec48" />


**Tooling**
- Command Code (similar to Claude Code but cheaper)
- GLM 5.2 for the initial build and DeepSeek V4 PRO for enhancements

**Setup**

> conda create -n simathon python=3.12

> pip install -r requirements.txt

> python lorenz-attractor/main.py
   
### My First Sim 
My first go at building a realistic black hole simulator, didn't go that well :).

**Tooling**
- Claude Code
- Kimi K2.7 Code model for the build
