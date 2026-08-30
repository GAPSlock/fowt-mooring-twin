# Unity Digital Twin — Technical Documentation

This document describes the real-time 3D visualization of the FOWT mooring digital twin built in Unity 6 (URP).

## Architecture

The Unity scene is driven by four scripts working together:

```
telemetry_feed.csv (OpenFAST 6-DOF data)
        │
        ▼
┌──────────────────┐     ┌─────────────────────┐
│ TelemetryReceiver│────▶│  MooringDigitalTwin  │
│  (reads CSV,     │     │  (ONNX inference,    │
│   drives motion) │     │   predicts tensions) │
└──────────────────┘     └──────┬──────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
     ┌────────────┐    ┌────────────┐     ┌────────────┐
     │FatigueMgr 1│    │FatigueMgr 2│     │FatigueMgr 3│
     │(Line 1)    │    │(Line 2)    │     │(Line 3)    │
     └────────────┘    └────────────┘     └────────────┘
              │                 │                  │
              ▼                 ▼                  ▼
        Rainflow cycle    Rainflow cycle     Rainflow cycle
        counting +        counting +         counting +
        DNV S-N fatigue   DNV S-N fatigue    DNV S-N fatigue
        → rust color      → rust color       → rust color
```

## Scripts

### 1. TelemetryReceiver.cs
- **Purpose:** Reads the `telemetry_feed.csv` file row by row and applies the 6-DOF platform kinematics (surge, sway, heave, roll, pitch, yaw) to the Cube's Rigidbody.
- **Update Rate:** 10 Hz (one row per 0.1s), matching real-time playback.
- **Motion Multiplier:** Position is scaled by a configurable multiplier (default 2x) for visual clarity. Rotation is always applied at true physical scale.
- **Looping:** The CSV loops infinitely when it reaches the end.

### 2. MooringDigitalTwin.cs
- **Purpose:** Runs the trained neural network (ONNX) in real time to predict mooring line tensions from the platform kinematics.
- **Input:** 18 features — 6 positions, 6 velocities, 6 accelerations — calculated via finite differences of the Rigidbody state.
- **Normalization:** Inputs are normalized using training-set statistics (`x_mean`, `x_std`). Outputs are un-normalized using `y_mean` and `y_std` to produce tensions in kN.
- **Inference Engine:** Unity Inference Engine (formerly Barracuda), running on GPU.

### 3. FatigueManager.cs
- **Purpose:** Performs real-time Miner's Rule fatigue damage accumulation using DNV GL B1 chain S-N curve parameters.
- **Rainflow Counting:** Simplified half-cycle extraction from the neural network tension output.
- **Chain Properties:** Nominal diameter 76.6 mm, S-N curve slope m = 3.0, log(a) = 11.566 (DNV-OS-E301).
- **Visual Feedback:** The mooring line color interpolates from clean steel grey (0% damage) to dark rust (100% damage) based on `totalDamage`.
- **Startup Shock Protection:** Ignores the first 10 frames to prevent the initial 0→1500 kN jump from registering as a massive fatigue cycle.

### 4. TurbineVisuals.cs
- **Purpose:** Procedurally generates a scaled OC4 DeepCwind semi-submersible geometry at runtime.
- **Components:** Central column, 3 outer columns (120° apart), connecting pontoons, tower, nacelle, hub, and 3 rotating blades.

### 5. OceanEnvironment.cs
- **Purpose:** Scene manager. Handles camera tracking, ocean transparency, mooring line spawning, and sunlight.
- **Camera:** Isometric drone camera that smoothly tracks the turbine.
- **Ocean:** Semi-transparent URP material (50% opacity) so underwater mooring lines are visible.

## Key Inspector Values

| Parameter | Value | Notes |
|---|---|---|
| Surge/Sway Multiplier | 2 | Horizontal visual scaling |
| Heave Multiplier | 15 | Vertical visual scaling (bobbing) |
| Update Rate | 0.1s | 10 Hz, real-time playback |
| Simulation Time Multiplier | 1000 | Fatigue accumulation speed |
| Nominal Diameter | 0.0766 m | R4 studless chain |
| S-N Slope (m) | 3.0 | DNV GL B1 curve |
| log(a) | 11.566 | DNV-OS-E301 Table 2-2 |

## Typical Output Tensions

| Line | Range (kN) | Physical Meaning |
|---|---|---|
| Line 1 | 1450 – 1750 | Windward, highest tension under surge |
| Line 2 | 1300 – 1650 | Leeward, lowest tension (goes slack) |
| Line 3 | 1450 – 1750 | Windward, symmetric with Line 1 |

Lines 1 and 3 are symmetric because they are the two windward mooring lines in the OC4 DeepCwind 120° spread. Line 2 is the single leeward line which relaxes under positive surge.

## GIF Recording Checklist

When recording the demonstration GIF for the README:

1. **Check Multipliers** (Inspector → Cube → TelemetryReceiver) are set to `Surge/Sway = 2` and `Heave = 15`.
2. **Let it run for ~30 seconds** before recording so the camera settles and startup transients pass
3. **Record for 15–20 seconds** to capture a few 10-second heave wave cycles
4. **Click on each MooringLine_Master in the Hierarchy** during recording to show the Inspector values changing in real time (tension, damage, RUL)

### Recommended GIF Caption for README:
> *Real-time digital twin running in Unity 6. The neural surrogate predicts mooring tensions at 50 Hz from platform kinematics. Mooring line color transitions from steel grey to rust as fatigue damage accumulates (Miner's Rule, DNV GL B1 S-N curve). Platform visual motion amplified for clarity.*

## Accelerated Playback — Frequency Explanation

The oscillation frequency you see in the GIF is the **true wave frequency** from the OpenFAST simulation. The playback runs at 1:1 real time (10 Hz CSV, 10 Hz Unity). The OC4 DeepCwind heave natural period is ~17.5 seconds, so you will see approximately one major heave cycle every 17.5 seconds of recording. Shorter wave-frequency oscillations (~6–12 second period) are superimposed from the sea state.

The `simulationTimeMultiplier = 1000` only affects how fast **fatigue damage** accumulates — it does NOT speed up the motion playback. This multiplier exists because real fatigue takes 20+ years to manifest; the 1000x multiplier compresses years of cumulative damage into minutes of simulation for visual demonstration purposes.
