# Research Paper Strategy: Real-Time FOWT Digital Twin

## 1. Target Venues
Based on the intersection of marine engineering and machine learning, this work is highly suitable for the following targets:
1. **Ocean Engineering (Elsevier):** High impact factor, heavily publishes on offshore structures, mooring dynamics, and increasingly, machine learning applications in marine technology.
2. **OMAE Conference (ASME):** The premier international conference for Ocean, Offshore, and Arctic Engineering. Perfect for immediate visibility and networking.
3. **Marine Structures (Elsevier):** Focuses strictly on structural integrity and fatigue, making the Rainflow counting / Miner's rule aspect of this project highly relevant.

## 2. Working Title Ideas
* *Real-Time Fatigue Monitoring of Floating Offshore Wind Turbine Mooring Systems via Physics-Informed Neural Networks and Edge Inference*
* *Bypassing Finite Element Solvers: Edge-Deployable Neural Networks for FOWT Mooring Fatigue Analysis*
* *A Digital Twin Architecture for Real-Time Structural Health Monitoring of Floating Wind Turbines using Neural Network Inference*

## 3. Paper Outline

### Abstract
* **Context:** Floating Offshore Wind Turbines (FOWTs) require continuous Structural Health Monitoring (SHM) of mooring lines to prevent catastrophic failure and reduce LCOE.
* **Problem:** Traditional coupled aero-hydro-servo-elastic solvers (like OpenFAST/MoorDyn) are too computationally intensive for real-time edge deployment on actual offshore platforms.
* **Method:** We propose a lightweight digital twin architecture. An 18-feature Deep Neural Network was trained on OC4-DeepCwind simulation data, mapping 6-DOF platform kinematics and their derivatives to non-linear mooring tensions. The model was exported via ONNX and deployed in a C# runtime engine.
* **Results:** The network achieves 98.3% accuracy (RMSE ~34kN) against the MoorDyn ground truth. Deployed on standard hardware, the digital twin executes real-time inference and Rainflow fatigue counting at 60 FPS, proving the viability of neural networks as replacements for heavy physics solvers in edge SHM applications.

### 1. Introduction
* The push for deep-water offshore wind and the reliance on semi-submersible platforms.
* The mechanics of mooring line fatigue and the cost of physical sensor failure in harsh marine environments.
* The computational bottleneck of existing solvers (MoorDyn, OrcaFlex).
* **Research Gap:** The need for a fast, accurate surrogate model that can run on standard IoT/edge hardware using only GPS/IMU telemetry (6-DOF data) as input.

### 2. Methodology
* **2.1 Data Generation:** Detail the setup of the NREL OpenFAST OC4 benchmark. Explain the metocean conditions used to generate the 750,000-row training dataset.
* **2.2 Feature Engineering:** Justify the expansion from 6 inputs (positions) to 18 inputs (positions, velocities, accelerations) to capture inertia and hydrodynamic drag, solving the early plateau in model accuracy.
* **2.3 Neural Network Architecture:** Detail the PyTorch model (layers, activation functions, Adam optimizer).
* **2.4 Digital Twin Deployment:** Explain the translation from Python to ONNX to C# (Unity InferenceEngine) to achieve real-time execution.
* **2.5 Fatigue Calculation:** Outline the implementation of the half-cycle Rainflow counting algorithm and Miner's Rule using DNVGL S-N curves (m=3.0, log_a=11.566).

### 3. Results and Validation
* **3.1 Model Accuracy:** Present the RMSE and loss curves. Include Plotly graphs comparing the PINN predictions to the OpenFAST ground truth across Surge and Pitch variations.
* **3.2 Computational Efficiency:** Compare the execution time of one step of MoorDyn vs. one inference pass of the ONNX model in C#. 
* **3.3 Fatigue Accumulation:** Demonstrate how accurately the Rainflow counting in the edge twin matches the theoretical fatigue of the raw dataset.

### 4. Discussion
* The limitations of the model (e.g., trained on specific sea states, generalizing to unseen wave spectrums).
* Future work: Integrating L-BFGS for sub-1% error margins, or expanding the model to include wind speed and wave elevation as direct inputs.

### 5. Conclusion
* Reiterate that the architecture successfully bridges the gap between high-fidelity simulation and real-time edge monitoring.

## 4. Required Figures to Generate
To make the paper publishable, we will need to generate the following specific plots from your data:
1. **Architecture Diagram:** A flowchart showing Data -> PyTorch -> ONNX -> Unity -> Rainflow.
2. **Time-Series Overlay:** A clean line graph showing 60 seconds of `True Tension` vs `Predicted Tension` (which you already have from `dashboard/app.py`).
3. **Scatter Plot (Predicted vs True):** A 45-degree parity plot proving the 98.3% accuracy.
4. **Fatigue Histogram:** A bar chart showing the distribution of stress ranges identified by the Rainflow counting algorithm.
