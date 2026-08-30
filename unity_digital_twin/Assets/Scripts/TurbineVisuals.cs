using UnityEngine;

public class TurbineVisuals : MonoBehaviour
{
    void Start()
    {
        MeshRenderer mr = GetComponent<MeshRenderer>();
        if (mr != null) mr.enabled = false;

        // --- 0. PIPELINE-AGNOSTIC MATERIAL FIX ---
        GameObject tempCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Material baseMat = new Material(tempCube.GetComponent<Renderer>().sharedMaterial);
        Destroy(tempCube);

        Material steelMat = new Material(baseMat);
        steelMat.color = new Color(0.9f, 0.75f, 0.1f); // Offshore safety yellow
        if (steelMat.HasProperty("_Glossiness")) steelMat.SetFloat("_Glossiness", 0.6f);
        if (steelMat.HasProperty("_Metallic")) steelMat.SetFloat("_Metallic", 0.8f);

        Material whiteMat = new Material(baseMat);
        whiteMat.color = new Color(0.95f, 0.95f, 0.95f);
        if (whiteMat.HasProperty("_Glossiness")) whiteMat.SetFloat("_Glossiness", 0.5f);

        // --- 1. REALISTIC OC4 SEMI-SUBMERSIBLE PLATFORM ---
        // Main central column
        CreateCylinder(this.transform, new Vector3(0, -10f, 0), new Vector3(6.5f, 15f, 6.5f), steelMat);
        
        // 3 Outer columns
        float radius = 20f; 
        for(int i = 0; i < 3; i++) {
            float angle = i * 120f * Mathf.Deg2Rad;
            Vector3 offset = new Vector3(Mathf.Sin(angle) * radius, -10f, Mathf.Cos(angle) * radius);
            
            // Outer Pillar
            CreateCylinder(this.transform, offset, new Vector3(6f, 15f, 6f), steelMat);
            
            // Heavy Pontoon connecting to center
            GameObject pontoon = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            pontoon.transform.SetParent(this.transform);
            pontoon.transform.localPosition = new Vector3(offset.x/2, -15f, offset.z/2);
            pontoon.transform.localScale = new Vector3(2.5f, radius/2, 2.5f);
            pontoon.transform.up = offset.normalized; 
            pontoon.GetComponent<Renderer>().material = steelMat;
        }

        // --- 2. TOWER ---
        CreateCylinder(this.transform, new Vector3(0, 30f, 0), new Vector3(2.5f, 35f, 2.5f), whiteMat);

        // --- 3. NACELLE & HUB ---
        GameObject nacelle = GameObject.CreatePrimitive(PrimitiveType.Cube);
        nacelle.transform.SetParent(this.transform);
        nacelle.transform.localPosition = new Vector3(0, 65f, -3f);
        nacelle.transform.localScale = new Vector3(3.5f, 4f, 10f);
        nacelle.GetComponent<Renderer>().material = whiteMat;

        // Hub Cone
        GameObject hub = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        hub.transform.SetParent(this.transform);
        hub.transform.localPosition = new Vector3(0, 65f, 3f);
        hub.transform.localScale = new Vector3(4f, 4f, 5f);
        hub.GetComponent<Renderer>().material = whiteMat;

        // --- 4. BLADES ---
        GameObject rotor = new GameObject("Rotor");
        rotor.transform.SetParent(this.transform);
        rotor.transform.localPosition = new Vector3(0, 65f, 3f);
        
        for(int i = 0; i < 3; i++) 
        {
            GameObject blade = GameObject.CreatePrimitive(PrimitiveType.Cube);
            blade.transform.SetParent(rotor.transform);
            blade.transform.localScale = new Vector3(1f, 30f, 0.2f); // Sleek, aerodynamic blades
            
            float angle = i * 120f;
            blade.transform.localRotation = Quaternion.Euler(0, 0, angle);
            blade.transform.localPosition = blade.transform.localRotation * new Vector3(0, 15f, 0); 
            blade.GetComponent<Renderer>().material = whiteMat;
        }
        
        rotor.AddComponent<RotorSpin>();
    }

    void CreateCylinder(Transform parent, Vector3 pos, Vector3 scale, Material mat) {
        GameObject cyl = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cyl.transform.SetParent(parent);
        cyl.transform.localPosition = pos;
        cyl.transform.localScale = scale;
        cyl.GetComponent<Renderer>().material = mat;
    }
}

public class RotorSpin : MonoBehaviour 
{
    void Update() 
    {
        transform.Rotate(0, 0, -50f * Time.deltaTime); // Slower, more majestic spin
    }
}
