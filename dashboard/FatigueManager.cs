using UnityEngine;

public class FatigueManager : MonoBehaviour
{
    public MooringDigitalTwin digitalTwin;
    public int lineNumber = 1;

    [Header("Chain Properties (DNVGL)")]
    public float nominalDiameter = 0.0766f; 
    public float m_curve = 3.0f;            
    public float log_a = 11.566f;           
    private float K_curve;
    
    [Header("Fatigue State")]
    public float totalDamage = 0.0f;
    public float remainingUsefulLife = 1.0f; // 1.0 = 100% life
    
    [Header("Presentation Settings")]
    [Tooltip("Accelerates time so you can watch years of rust accumulate in 5 minutes!")]
    public float simulationTimeMultiplier = 1000000f; 

    [Header("Visual Degradation")]
    public Color cleanColor = new Color(0.3f, 0.3f, 0.3f); // Metallic Gray
    public Color rustColor = new Color(0.6f, 0.2f, 0.05f); // Rusty Orange/Brown
    private Material material;

    private float lastTension = 0f;
    private float lastTrend = 0f;
    private float lastExtremum = 0f;

    void Start()
    {
        material = GetComponent<Renderer>().material;
        K_curve = Mathf.Pow(10, log_a);
    }

    void Update()
    {
        if (digitalTwin == null) return;

        float currentTension = 0f;
        switch (lineNumber)
        {
            case 1: currentTension = digitalTwin.tensionLine1; break;
            case 2: currentTension = digitalTwin.tensionLine2; break;
            case 3: currentTension = digitalTwin.tensionLine3; break;
        }

        // Half-Cycle Peak-Valley Extraction
        float trend = currentTension - lastTension;
        
        if (trend * lastTrend < 0) 
        {
            float tensionRange_kN = Mathf.Abs(currentTension - lastExtremum);
            if (tensionRange_kN > 5.0f) 
            {
                CalculateFatigueDamage(tensionRange_kN);
            }
            lastExtremum = currentTension;
        }

        if (Mathf.Abs(trend) > 0.001f) lastTrend = trend;
        lastTension = currentTension;

        // Apply visual rust blending
        material.color = Color.Lerp(cleanColor, rustColor, totalDamage);
    }

    void CalculateFatigueDamage(float tensionRange_kN)
    {
        float area_m2 = 2.0f * (Mathf.PI / 4.0f) * Mathf.Pow(nominalDiameter, 2);
        float stressRange_MPa = (tensionRange_kN / 1000.0f) / area_m2;
        
        // N = K * S^-m
        float cyclesToFailure = K_curve * Mathf.Pow(stressRange_MPa, -m_curve);
        float fractionalDamage = 0.5f / cyclesToFailure;
        
        totalDamage += (fractionalDamage * simulationTimeMultiplier);
        remainingUsefulLife = Mathf.Clamp01(1.0f - totalDamage);
    }
}
