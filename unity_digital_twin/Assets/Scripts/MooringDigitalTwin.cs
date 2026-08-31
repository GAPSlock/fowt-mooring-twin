using UnityEngine;
using Unity.InferenceEngine;
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

    [Header("De-Scale Multipliers (Must match TelemetryReceiver)")]
    public float surgeSwayMultiplier = 2f;
    public float heaveMultiplier = 15f;

    private readonly float[] x_mean = new float[18] { 6.874f, -0.121f, -0.027f, 0.009f, 2.467f, -0.387f, 0.0001f, -0.00001f, 0.00002f, 0.0000009f, 0.00005f, -0.00004f, -0.00001f, 0.00000005f, -0.000006f, -0.00000007f, 0.000001f, 0.0000001f };
    private readonly float[] x_std = new float[18] { 0.552f, 0.045f, 0.206f, 0.003f, 0.182f, 0.088f, 0.562f, 0.038f, 0.096f, 0.002f, 0.155f, 0.105f, 31.146f, 2.199f, 3.016f, 0.132f, 7.576f, 5.943f };

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute); 
        
        if (platformRb != null) {
            previousPosition = new Vector3(platformRb.position.x / surgeSwayMultiplier, 
                                           platformRb.position.z / surgeSwayMultiplier, 
                                           platformRb.position.y / heaveMultiplier);
            previousRotation = platformRb.rotation.eulerAngles;
        }
    }

    void FixedUpdate()
    {
        if (platformRb == null) return;

        float dt = Time.fixedDeltaTime;
        if (dt == 0) return;

        // STRIP THE VISUAL MULTIPLIERS TO GET TRUE PHYSICS DATA
        float surge = platformRb.position.x / surgeSwayMultiplier;
        float sway = platformRb.position.z / surgeSwayMultiplier;
        float heave = platformRb.position.y / heaveMultiplier;
        float roll = platformRb.rotation.eulerAngles.z;
        float pitch = platformRb.rotation.eulerAngles.x;
        float yaw = platformRb.rotation.eulerAngles.y;

        Vector3 currentPos = new Vector3(surge, sway, heave);
        Vector3 currentRot = new Vector3(roll, pitch, yaw);
        
        Vector3 vel = (currentPos - previousPosition) / dt;
        Vector3 angVel = (currentRot - previousRotation) / dt;

        Vector3 acc = (vel - previousVelocity) / dt;
        Vector3 angAcc = (angVel - previousAngularVelocity) / dt;

        float[] inputs = new float[18] {
            surge, sway, heave, roll, pitch, yaw,
            vel.x, vel.y, vel.z, angVel.x, angVel.y, angVel.z,
            acc.x, acc.y, acc.z, angAcc.x, angAcc.y, angAcc.z
        };

        for(int i=0; i<18; i++)
        {
            inputs[i] = (inputs[i] - x_mean[i]) / (x_std[i] + 1e-8f);
        }

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 18), inputs);
        worker.Schedule(inputTensor);

        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        float[] preds = outputTensor.DownloadToArray();

        // The ONNX model outputs normalized tensions. 
        // We must un-normalize them back into Kilonewtons (kN) using the exact mean/std from the training set.
        float[] y_mean = new float[3] { 966.21f, 1507.24f, 958.32f };
        float[] y_std = new float[3] { 13.94f, 55.87f, 13.90f };

        tensionLine1 = (preds[0] * y_std[0]) + y_mean[0];
        tensionLine2 = (preds[1] * y_std[1]) + y_mean[1];
        tensionLine3 = (preds[2] * y_std[2]) + y_mean[2];

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
