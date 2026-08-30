using UnityEngine;

public class OceanEnvironment : MonoBehaviour
{
    private MooringDigitalTwin twinTarget;

    void Start()
    {
        // 0. UNPARENT FROM TURBINE so the ocean stays flat while the turbine tilts
        this.transform.SetParent(null);
        this.transform.rotation = Quaternion.identity; 
        
        Camera cam = Camera.main;
        if (cam != null)
        {
            cam.transform.SetParent(null);
            cam.backgroundColor = new Color(0.55f, 0.75f, 0.95f);
            cam.clearFlags = CameraClearFlags.SolidColor; 
            cam.fieldOfView = 50f; 
        }

        // 1. MAKE WATER TRANSPARENT
        // Instead of fighting the URP shader, we just make the plane semi-transparent
        // by setting the alpha on the base color. For URP Lit, we need to set _Surface=1 (Transparent)
        // and use SetOverrideTag + renderQueue so URP actually treats it as transparent.
        MeshRenderer mr = GetComponent<MeshRenderer>();
        if (mr != null)
        {
            Material oceanMat = mr.material; // Use the EXISTING material instance, don't create new
            
            // Set the base color with moderate alpha — visible blue, but mooring ropes show through
            oceanMat.SetColor("_BaseColor", new Color(0.04f, 0.18f, 0.45f, 0.50f));
            
            // Tell URP this is a transparent surface
            if (oceanMat.HasProperty("_Surface"))
            {
                oceanMat.SetFloat("_Surface", 1.0f); // 0=Opaque, 1=Transparent in URP
                oceanMat.SetFloat("_Blend", 0.0f);   // 0=Alpha, 1=Premultiply, 2=Additive, 3=Multiply
            }
            
            // Override render type and queue
            oceanMat.SetOverrideTag("RenderType", "Transparent");
            oceanMat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            
            // Set blend modes
            oceanMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            oceanMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            oceanMat.SetInt("_ZWrite", 0);
            
            // Enable the relevant keyword for URP transparency
            oceanMat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            oceanMat.DisableKeyword("_SURFACE_TYPE_OPAQUE");
            
            mr.material = oceanMat;
        }

        // 2. SUNLIGHT
        Light[] lights = FindObjectsOfType<Light>();
        bool hasSun = false;
        foreach(var l in lights) { if(l.type == LightType.Directional) { hasSun = true; l.intensity = 1.3f; } }
        
        if (!hasSun)
        {
            GameObject sun = new GameObject("Sun");
            Light light = sun.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.3f;
            light.color = new Color(1f, 0.98f, 0.95f); 
            sun.transform.rotation = Quaternion.Euler(45f, -45f, 0f);
        }

        // 3. FIND THE TURBINE
        twinTarget = FindObjectOfType<MooringDigitalTwin>();
        
        // 4. CLEANUP AND AUTO-SPAWN MOORING LINES
        FatigueManager[] oldLines = FindObjectsOfType<FatigueManager>();
        foreach(var oldLine in oldLines) { Destroy(oldLine.gameObject); }

        // Pipeline-safe material for ropes
        GameObject tempCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Material ropeMat = new Material(tempCube.GetComponent<Renderer>().sharedMaterial);
        Destroy(tempCube);

        if (twinTarget != null)
        {
            // OC4 DeepCwind mooring spread: 120 degrees apart
            Vector3[] anchors = new Vector3[] { 
                new Vector3(80f, -80f, 0f), 
                new Vector3(-40f, -80f, 69.28f), 
                new Vector3(-40f, -80f, -69.28f) 
            };
            Vector3[] fairleads = new Vector3[] { 
                new Vector3(20f, -10f, 0f), 
                new Vector3(-10f, -10f, 17.32f), 
                new Vector3(-10f, -10f, -17.32f) 
            };

            for (int i = 0; i < 3; i++)
            {
                GameObject lineObj = new GameObject("MooringLine_Master_" + (i + 1));
                
                LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                lr.material = new Material(ropeMat); 
                lr.startWidth = 1.5f; 
                lr.endWidth = 1.5f;
                
                FatigueManager fm = lineObj.AddComponent<FatigueManager>();
                fm.digitalTwin = twinTarget;
                fm.lineNumber = i + 1;
                fm.seabedAnchor = anchors[i];
                fm.fairleadOffset = fairleads[i];
                fm.simulationTimeMultiplier = 1000f; 
            }
        }
    }

    void LateUpdate()
    {
        if (twinTarget == null || Camera.main == null) return;
        
        Camera cam = Camera.main;
        
        // Pull WAY back so the entire structure is visible:
        // Blade tips are at ~95m, mooring anchors at -80m. 
        // Need ~175m of vertical span visible.
        Vector3 targetCamPos = new Vector3(-100f, 20f, -100f);
        cam.transform.position = Vector3.Lerp(cam.transform.position, targetCamPos, Time.deltaTime * 3f);
        
        // Look at the center of the whole structure (midway between seabed anchors and nacelle)
        Vector3 lookTarget = twinTarget.transform.position + new Vector3(0, 10f, 0);
        Quaternion targetRot = Quaternion.LookRotation(lookTarget - cam.transform.position);
        cam.transform.rotation = Quaternion.Slerp(cam.transform.rotation, targetRot, Time.deltaTime * 5f);
    }
}
