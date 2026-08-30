using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;

public class MooringDigitalTwin : MonoBehaviour
{
    [Header("Neural Network")]
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    [Header("Turbine Rigidbody")]
    public Rigidbody platformRb;
    
    private Vector3 previousPosition;
    private Vector3 previousRotation;
    private Vector3 previousVelocity;
    private Vector3 previousAngularVelocity;

    [Header("Output Tensions (kN)")]
    public float tensionLine1;
    public float tensionLine2;
    public float tensionLine3;

    // Hardcoded normalization arrays from PyTorch training
    private readonly float[] x_mean = new float[18] { 6.874f, -0.121f, -0.027f, 0.009f, 2.467f, -0.387f, 0.0001f, -0.00001f, 0.00002f, 0.0000009f, 0.00005f, -0.00004f, -0.00001f, 0.00000005f, -0.000006f, -0.00000007f, 0.000001f, 0.0000001f };
    private readonly float[] x_std = new float[18] { 0.552f, 0.045f, 0.206f, 0.003f, 0.182f, 0.088f, 0.562f, 0.038f, 0.096f, 0.002f, 0.155f, 0.105f, 31.146f, 2.199f, 3.016f, 0.132f, 7.576f, 5.943f };

    void Start()
    {
        // Load the ONNX model to the GPU
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        
        if (platformRb != null) {
            previousPosition = platformRb.position;
            previousRotation = platformRb.rotation.eulerAngles;
        }
    }

    void FixedUpdate()
    {
        if (platformRb == null) return;

        float dt = Time.fixedDeltaTime;
        if (dt == 0) return;

        // 1. Get exact positions from the Unity Engine (Map Y-up to Z-up for Neural Net)
        float surge = platformRb.position.x;
        float sway = platformRb.position.z;
        float heave = platformRb.position.y;
        float roll = platformRb.rotation.eulerAngles.z;
        float pitch = platformRb.rotation.eulerAngles.x;
        float yaw = platformRb.rotation.eulerAngles.y;

        // 2. Calculate Velocities mathematically (since TelemetryReceiver is teleporting the Rigidbody)
        Vector3 currentPos = new Vector3(surge, sway, heave);
        Vector3 currentRot = new Vector3(roll, pitch, yaw);
        
        Vector3 vel = (currentPos - previousPosition) / dt;
        Vector3 angVel = (currentRot - previousRotation) / dt;

        // 3. Calculate Accelerations
        Vector3 acc = (vel - previousVelocity) / dt;
        Vector3 angAcc = (angVel - previousAngularVelocity) / dt;

        // 4. Build the 18-element Input Tensor array
        float[] inputs = new float[18] {
            surge, sway, heave, roll, pitch, yaw,
            vel.x, vel.y, vel.z, angVel.x, angVel.y, angVel.z,
            acc.x, acc.y, acc.z, angAcc.x, angAcc.y, angAcc.z
        };

        // 5. Normalize using PyTorch Mean and Std
        for(int i=0; i<18; i++)
        {
            inputs[i] = (inputs[i] - x_mean[i]) / (x_std[i] + 1e-8f);
        }

        // 6. Execute Neural Network
        using TensorFloat inputTensor = new TensorFloat(new TensorShape(1, 18), inputs);
        worker.Execute(inputTensor);

        // 7. Extract predicted Tensions (kN)
        TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
        // Make the data readable from GPU to CPU
        outputTensor.MakeReadable(); 
        
        // The ONNX model outputs normalized tensions. 
        // We must un-normalize them back into Kilonewtons (kN) using the exact mean/std from the training set.
        float[] y_mean = new float[3] { 1526.47f, 1507.23f, 1530.12f };
        float[] y_std = new float[3] { 48.33f, 55.86f, 49.11f };

        // Read the normalized outputs from the ONNX Tensor
        float normTen1 = outputTensor[0];
        float normTen2 = outputTensor[1];
        float normTen3 = outputTensor[2];

        // Un-normalize back into Kilonewtons for the Rainflow algorithm
        tensionLine1 = (normTen1 * y_std[0]) + y_mean[0];
        tensionLine2 = (normTen2 * y_std[1]) + y_mean[1];
        tensionLine3 = (normTen3 * y_std[2]) + y_mean[2];

        // Save state for next frame
        previousPosition = currentPos;
        previousRotation = currentRot;
        previousVelocity = vel;
        previousAngularVelocity = angVel;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
