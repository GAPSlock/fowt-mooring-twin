using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TelemetryReceiver : MonoBehaviour
{
    [Tooltip("Drag the telemetry_feed.csv file here from your Unity Project tab")]
    public TextAsset telemetryData;
    
    [Tooltip("Time between data frames (0.1s because our Python script exported at 10Hz)")]
    public float updateRate = 0.1f;

    [Tooltip("Multiply the movement so it is highly visible in the presentation")]
    public float motionMultiplier = 5f;

    private string[] rows;
    private int currentRow = 1; // skip header
    private float timer = 0f;

    private Rigidbody rb;

    void Start()
    {
        // Safely cache the Rigidbody so we don't call it every frame
        rb = GetComponent<Rigidbody>();

        if (telemetryData != null)
        {
            rows = telemetryData.text.Split(new char[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        }
        else
        {
            Debug.LogError("Missing Telemetry CSV! Please drag telemetry_feed.csv into the script slot.");
        }
    }

    void FixedUpdate()
    {
        // Safety check to ensure the object hasn't been destroyed
        if (this == null || gameObject == null) return;
        if (rows == null || currentRow >= rows.Length) return;

        timer += Time.fixedDeltaTime;
        if (timer >= updateRate)
        {
            timer = 0f;
            ProcessRow(rows[currentRow]);
            currentRow++;
            
            if (currentRow >= rows.Length) currentRow = 1; 
        }
    }

    void ProcessRow(string row)
    {
        string[] columns = row.Split(',');
        if (columns.Length < 7) return;

        float surge = float.Parse(columns[1]);
        float sway  = float.Parse(columns[2]);
        float heave = float.Parse(columns[3]);
        float roll  = float.Parse(columns[4]);
        float pitch = float.Parse(columns[5]);
        float yaw   = float.Parse(columns[6]);

        Vector3 newPosition = new Vector3(surge, heave, sway) * motionMultiplier;
        Quaternion newRotation = Quaternion.Euler(pitch * motionMultiplier, yaw * motionMultiplier, roll * motionMultiplier);
        
        if (rb != null)
        {
            rb.MovePosition(newPosition);
            rb.MoveRotation(newRotation);
        }
        else
        {
            transform.position = newPosition;
            transform.rotation = newRotation;
        }
    }
}
