using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class FatigueManager : MonoBehaviour
{
    public MooringDigitalTwin digitalTwin;
    public int lineNumber = 1;
    
    [Header("Mooring Line Anchors")]
    public Vector3 seabedAnchor = new Vector3(30f, -40f, 30f);
    public Vector3 fairleadOffset = new Vector3(5f, -2f, 5f); // Where it attaches to the floating base

    [Header("Chain Properties (DNVGL)")]
    public float nominalDiameter = 0.0766f; 
    public float m_curve = 3.0f;            
    public float log_a = 11.566f;           
    private float K_curve;
    
    [Header("Fatigue State")]
    public float totalDamage = 0.0f;
    public float remainingUsefulLife = 1.0f;
    public float simulationTimeMultiplier = 1000f; 

    [Header("Visual Degradation")]
    public Color cleanColor = new Color(0.3f, 0.3f, 0.3f); 
    public Color rustColor = new Color(0.6f, 0.2f, 0.05f); 
    
    private LineRenderer lineRenderer;
    private float lastTension = 1500f;
    private float lastTrend = 0f;
    private float lastExtremum = 1500f;
    private int frameCount = 0;

    void Start()
    {
        K_curve = Mathf.Pow(10, log_a);
        
        // Setup the physical rope visualizer
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = 2;
    }

    void Update()
    {
        if (digitalTwin == null) return;
        frameCount++;

        // 1. Draw the dynamic rope stretching from the seabed to the moving turbine
        lineRenderer.SetPosition(0, seabedAnchor);
        Vector3 attachmentPointOnTurbine = digitalTwin.transform.TransformPoint(fairleadOffset);
        lineRenderer.SetPosition(1, attachmentPointOnTurbine);

        // 2. Read the Neural Network prediction
        float currentTension = 0f;
        switch (lineNumber)
        {
            case 1: currentTension = digitalTwin.tensionLine1; break;
            case 2: currentTension = digitalTwin.tensionLine2; break;
            case 3: currentTension = digitalTwin.tensionLine3; break;
        }

        // --- CRITICAL FIX: IGNORE STARTUP SHOCK ---
        // The first few frames the NN jumps from 0 to 1500+ kN. This massive 1500kN "cycle"
        // instantly caused 1.2+ totalDamage (120% damage), turning the rope instantly red and RUL to 0.
        if (frameCount < 10) 
        {
            lastTension = currentTension;
            lastExtremum = currentTension;
            return;
        }

        // 3. Rainflow Half-Cycle extraction
        float trend = currentTension - lastTension;
        if (trend * lastTrend < 0) 
        {
            float tensionRange_kN = Mathf.Abs(currentTension - lastExtremum);
            if (tensionRange_kN > 5.0f) CalculateFatigueDamage(tensionRange_kN);
            lastExtremum = currentTension;
        }
        if (Mathf.Abs(trend) > 0.001f) lastTrend = trend;
        lastTension = currentTension;

        // 4. Apply rust to the rope!
        if (lineRenderer.material != null)
        {
            lineRenderer.material.color = Color.Lerp(cleanColor, rustColor, totalDamage);
        }
    }

    void CalculateFatigueDamage(float tensionRange_kN)
    {
        float area_m2 = 2.0f * (Mathf.PI / 4.0f) * Mathf.Pow(nominalDiameter, 2);
        float stressRange_MPa = (tensionRange_kN / 1000.0f) / area_m2;
        float cyclesToFailure = K_curve * Mathf.Pow(stressRange_MPa, -m_curve);
        float fractionalDamage = 0.5f / cyclesToFailure;
        
        totalDamage += (fractionalDamage * simulationTimeMultiplier);
        remainingUsefulLife = Mathf.Clamp01(1.0f - totalDamage);
    }
}
