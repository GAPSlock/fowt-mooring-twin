# Unity Sentis Integration: FOWT Digital Twin

To run the PINN inside Unity, you will use **Unity Sentis** (Unity's official neural network inference library). 

## 1. Setup Instructions
1. Open Unity Hub and create a **New 3D Project** (URP or Standard).
2. Open the Package Manager (`Window > Package Manager`).
3. Click the `+` icon $\rightarrow$ "Add package from git URL..."
4. Enter `com.unity.sentis` and install it.
5. Drag and drop the `fowt_mooring_twin.onnx` file into your Unity `Assets` folder.

## 2. The Core Inference Script (`MooringDigitalTwin.cs`)
Attach this script to an empty GameObject in your scene called `DigitalTwinManager`. Assign your `.onnx` model to the `modelAsset` slot in the Unity Inspector.

```csharp
using UnityEngine;
using Unity.Sentis;

public class MooringDigitalTwin : MonoBehaviour
{
    [Header("Neural Network")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private IWorker worker;

    [Header("Turbine Rigidbody")]
    public Rigidbody platformRb;
    
    // To calculate Acceleration, we need the previous velocity
    private Vector3 previousVelocity;
    private Vector3 previousAngularVelocity;

    [Header("Output Tensions (kN)")]
    public float tensionLine1;
    public float tensionLine2;
    public float tensionLine3;

    void Start()
    {
        // Load the ONNX model and create a worker to run it on the GPU
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
        
        if (platformRb != null) {
            previousVelocity = platformRb.velocity;
            previousAngularVelocity = platformRb.angularVelocity;
        }
    }

    void FixedUpdate()
    {
        if (platformRb == null) return;

        // 1. Extract Positions (Surge, Sway, Heave) & Rotations (Roll, Pitch, Yaw)
        // Note: Unity is left-handed Y-up, OpenFAST is right-handed Z-up. 
        // We do a rough mapping here.
        float surge = platformRb.position.x;
        float sway = platformRb.position.z;
        float heave = platformRb.position.y;
        
        float roll = platformRb.rotation.eulerAngles.x;
        float pitch = platformRb.rotation.eulerAngles.z;
        float yaw = platformRb.rotation.eulerAngles.y;

        // 2. Extract Velocities
        Vector3 vel = platformRb.velocity;
        Vector3 angVel = platformRb.angularVelocity;

        // 3. Calculate Accelerations (dv/dt)
        float dt = Time.fixedDeltaTime;
        Vector3 acc = (vel - previousVelocity) / dt;
        Vector3 angAcc = (angVel - previousAngularVelocity) / dt;

        // 4. Build the 18-element Input Tensor
        // Order must match Python: [Surge, Sway, Heave, Roll, Pitch, Yaw, Vx, Vy, Vz, Wx, Wy, Wz, Ax, Ay, Az, Alpha_x, Alpha_y, Alpha_z]
        float[] inputs = new float[18] {
            surge, sway, heave, roll, pitch, yaw,
            vel.x, vel.z, vel.y, angVel.x, angVel.z, angVel.y,
            acc.x, acc.z, acc.y, angAcc.x, angAcc.z, angAcc.y
        };

        // Note: In a production twin, you MUST normalize this array using the X_mean and X_std values saved during training!
        // float[] normalizedInputs = Normalize(inputs);

        // 5. Run the Neural Network
        using TensorFloat inputTensor = new TensorFloat(new TensorShape(1, 18), inputs);
        worker.Execute(inputTensor);

        // 6. Extract the predicted Tensions
        TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
        float[] predictedTensions = outputTensor.ToReadOnlyArray();

        tensionLine1 = predictedTensions[0];
        tensionLine2 = predictedTensions[1];
        tensionLine3 = predictedTensions[2];

        // Save state for next acceleration calculation
        previousVelocity = vel;
        previousAngularVelocity = angVel;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
```

## 3. The Visualizer Script (`MooringVisualizer.cs`)
Attach this script to your 3D Mooring Line object (e.g., a cylinder or a chain mesh). Drag the `DigitalTwinManager` into the script's inspector slot, and specify which line (1, 2, or 3) this object represents.

```csharp
using UnityEngine;

public class MooringVisualizer : MonoBehaviour
{
    public MooringDigitalTwin digitalTwin;
    
    [Tooltip("1, 2, or 3")]
    public int lineNumber = 1; 
    
    [Tooltip("Max Expected Tension (kN) for Color Scaling")]
    public float maxTension = 2000f; 

    private Material material;

    void Start()
    {
        // Get the material of the mooring line so we can change its color
        material = GetComponent<Renderer>().material;
    }

    void Update()
    {
        if (digitalTwin == null) return;

        float currentTension = 0f;

        // Get the correct tension from the neural network output
        switch (lineNumber)
        {
            case 1: currentTension = digitalTwin.tensionLine1; break;
            case 2: currentTension = digitalTwin.tensionLine2; break;
            case 3: currentTension = digitalTwin.tensionLine3; break;
        }

        // Calculate how close the chain is to breaking (0.0 to 1.0)
        float stressRatio = Mathf.Clamp01(currentTension / maxTension);

        // Color Interpolation: 
        // 0% Stress = Gray
        // 50% Stress = Yellow
        // 100% Stress = Red
        Color safeColor = Color.gray;
        Color warningColor = Color.yellow;
        Color dangerColor = Color.red;

        Color finalColor;
        if (stressRatio < 0.5f)
        {
            // Lerp from Gray to Yellow
            finalColor = Color.Lerp(safeColor, warningColor, stressRatio * 2f);
        }
        else
        {
            // Lerp from Yellow to Red
            finalColor = Color.Lerp(warningColor, dangerColor, (stressRatio - 0.5f) * 2f);
        }

        // Apply the color to the 3D chain
        material.color = finalColor;
    }
}
```

## 4. Real-Time Fatigue & Rust Visualizer (`FatigueManager.cs`)
This script implements a real-time Peak-Valley cycle counter. As the PINN streams tension data, this script extracts the stress reversals, calculates the accumulated damage using the DNV-OS-E301 S-N curve for studless mooring chains, and visibly "rusts" the chain.

Because real mooring chains take 25 years to degrade, this script includes a `simulationTimeMultiplier` so you can visually demonstrate the chain degrading into rust over a 5-minute presentation.

```csharp
using UnityEngine;
using UnityEngine.UI; // If you want to link a UI Text element for RUL

public class FatigueManager : MonoBehaviour
{
    public MooringDigitalTwin digitalTwin;
    public int lineNumber = 1;

    [Header("Chain Properties (DNVGL)")]
    public float nominalDiameter = 0.0766f; // 76.6 mm chain
    public float m_curve = 3.0f;            // S-N curve slope parameter
    public float log_a = 11.566f;           // Intercept for studless chain
    private float K_curve;
    
    [Header("Fatigue State")]
    public float totalDamage = 0.0f;
    public float remainingUsefulLife = 1.0f; // 1.0 = 100% life
    
    [Header("Demonstration Settings")]
    [Tooltip("Multiplies the damage so you can visually see the rust accumulate in minutes instead of years.")]
    public float simulationTimeMultiplier = 1000000f; 

    [Header("Visual Degradation")]
    public Color cleanColor = new Color(0.3f, 0.3f, 0.3f); // Metallic Gray
    public Color rustColor = new Color(0.6f, 0.2f, 0.05f); // Rusty Orange/Brown
    private Material material;

    // Peak-Valley Extraction variables
    private float lastTension = 0f;
    private float lastTrend = 0f;
    private float lastExtremum = 0f;

    void Start()
    {
        material = GetComponent<Renderer>().material;
        // Convert log_a to linear scale K
        K_curve = Mathf.Pow(10, log_a);
    }

    void Update()
    {
        if (digitalTwin == null) return;

        // 1. Get PINN Tension (in kN)
        float currentTension = 0f;
        switch (lineNumber)
        {
            case 1: currentTension = digitalTwin.tensionLine1; break;
            case 2: currentTension = digitalTwin.tensionLine2; break;
            case 3: currentTension = digitalTwin.tensionLine3; break;
        }

        // 2. Real-Time Half-Cycle Extraction
        float trend = currentTension - lastTension;
        
        // If the trend changes sign (e.g. was going up, now going down), we hit a peak or valley
        if (trend * lastTrend < 0) 
        {
            // We completed a reversal! Calculate the stress range
            float tensionRange_kN = Mathf.Abs(currentTension - lastExtremum);
            
            if (tensionRange_kN > 10.0f) // Ignore tiny noise vibrations
            {
                CalculateFatigueDamage(tensionRange_kN);
            }

            lastExtremum = currentTension;
        }

        if (Mathf.Abs(trend) > 0.001f) 
        {
            lastTrend = trend;
        }
        
        lastTension = currentTension;

        // 3. Update Visual Rust
        UpdateRustVisuals();
    }

    void CalculateFatigueDamage(float tensionRange_kN)
    {
        // A. Convert Tension (kN) to Nominal Stress (MPa)
        // Area = 2 * (pi/4) * d^2 for a chain link
        float area_m2 = 2.0f * (Mathf.PI / 4.0f) * Mathf.Pow(nominalDiameter, 2);
        float stressRange_MPa = (tensionRange_kN / 1000.0f) / area_m2;

        // B. Miner's Rule & S-N Curve (N = K * S^-m)
        // N is the number of cycles to failure AT THIS specific stress range
        float cyclesToFailure = K_curve * Mathf.Pow(stressRange_MPa, -m_curve);

        // C. Add the fractional damage of this single half-cycle
        // Half cycle = 0.5 / N
        float fractionalDamage = 0.5f / cyclesToFailure;
        
        // Multiply by our demo scale factor
        totalDamage += (fractionalDamage * simulationTimeMultiplier);
        
        // Clamp RUL to 0
        remainingUsefulLife = Mathf.Clamp01(1.0f - totalDamage);
    }

    void UpdateRustVisuals()
    {
        // Lerp from clean metallic to rusty brown based on total damage
        material.color = Color.Lerp(cleanColor, rustColor, totalDamage);
        
        // Optional: If you use a custom shader, you can interpolate a Rust Texture!
        // material.SetFloat("_RustBlend", totalDamage);
    }
}
```
