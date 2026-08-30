using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TelemetryReceiver : MonoBehaviour
{
    [Tooltip("Drag the telemetry_feed.csv file here from your Unity Project tab")]
    public TextAsset telemetryData;
    
    [Tooltip("Time between data frames in the CSV (0.1s)")]
    public float csvTimeStep = 0.1f;

    [Tooltip("Multiply horizontal movement. Kept at 2 so it doesn't drift off screen.")]
    public float surgeSwayMultiplier = 2f;
    
    [Tooltip("Multiply vertical movement by a massive amount (15x) so the 0.4m heave becomes a visible, majestic 6m bob.")]
    public float heaveMultiplier = 15f;

    private string[] rows;
    private int currentRow = 1; 
    private float timer = 0f;

    // Cache the previous and next states for smooth interpolation
    private Vector3 posA, posB;
    private Quaternion rotA, rotB;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        if (telemetryData != null)
        {
            rows = telemetryData.text.Split(new char[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (rows.Length > 2)
            {
                ParseRow(rows[1], out posA, out rotA);
                ParseRow(rows[2], out posB, out rotB);
            }
        }
    }

    void FixedUpdate()
    {
        if (rows == null || currentRow >= rows.Length - 1) return;

        timer += Time.fixedDeltaTime;
        
        // Calculate how far we are between row A and row B (0.0 to 1.0)
        float t = Mathf.Clamp01(timer / csvTimeStep);

        // Smoothly interpolate position and rotation
        Vector3 lerpedPos = Vector3.Lerp(posA, posB, t);
        Quaternion lerpedRot = Quaternion.Slerp(rotA, rotB, t);

        if (rb != null)
        {
            rb.MovePosition(lerpedPos);
            rb.MoveRotation(lerpedRot);
        }
        else
        {
            transform.position = lerpedPos;
            transform.rotation = lerpedRot;
        }

        // If we reached the next CSV timestamp, shift everything forward
        if (timer >= csvTimeStep)
        {
            timer -= csvTimeStep; // keep the remainder for perfect timing
            currentRow++;
            
            // Loop if we hit the end
            if (currentRow >= rows.Length - 1) 
            {
                currentRow = 1;
            }

            posA = posB;
            rotA = rotB;
            ParseRow(rows[currentRow + 1], out posB, out rotB);
        }
    }

    void ParseRow(string row, out Vector3 position, out Quaternion rotation)
    {
        string[] columns = row.Split(',');
        if (columns.Length < 7) 
        {
            position = Vector3.zero;
            rotation = Quaternion.identity;
            return;
        }

        float surge = float.Parse(columns[1]);
        float sway  = float.Parse(columns[2]);
        float heave = float.Parse(columns[3]);
        float roll  = float.Parse(columns[4]);
        float pitch = float.Parse(columns[5]);
        float yaw   = float.Parse(columns[6]);

        // Multiply only position with decoupled axes. Rotation stays true physical degrees.
        position = new Vector3(surge * surgeSwayMultiplier, heave * heaveMultiplier, sway * surgeSwayMultiplier);
        rotation = Quaternion.Euler(pitch, yaw, roll);
    }
}
