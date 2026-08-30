import torch
import os
import sys

# Add current dir to path to import model
sys.path.append(os.path.dirname(__file__))
from model import MooringPINN

def export_to_onnx():
    device = torch.device('cpu') # Export using CPU
    model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)
    
    # Load trained weights
    checkpoint_path = "checkpoints/pinn_adam_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
    
    # Create a dummy input tensor of the correct shape [batch_size, 6]
    # Unity will pass 1 frame at a time, so batch_size = 1
    dummy_input = torch.randn(1, 18, requires_grad=True).to(device)
    
    # Define dynamic axes in case we want to pass multiple frames at once in Unity
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    onnx_path = "checkpoints/fowt_mooring_twin.onnx"
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        onnx_path,                  # where to save the model
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes=dynamic_axes   # variable length axes
    )
    
    print("ONNX export complete! The model is ready to be dragged and dropped into Unity Sentis.")

if __name__ == "__main__":
    export_to_onnx()
