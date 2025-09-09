# Script to export model to ONNX
import torch
import torch.onnx
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def export_model_to_onnx(model_path, output_path, input_shape=(3, 224, 224), opset_version=11):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to the PyTorch model (.pth file)
        output_path (str): Path where to save the ONNX model
        input_shape (tuple): Input tensor shape (C, H, W)
        opset_version (int): ONNX opset version
    """
    try:
        # Load the model
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = torch.load(model_path, map_location=device)
        else:
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device)
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Model exported successfully to: {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 ** 2)  # Size in MB
        print(f"📁 ONNX model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting model: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the PyTorch model (.pth file)')
    parser.add_argument('--output_path', type=str, 
                       help='Output path for ONNX model (default: model.onnx)')
    parser.add_argument('--input_shape', nargs=3, type=int, default=[3, 224, 224],
                       help='Input shape as C H W (default: 3 224 224)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    
    args = parser.parse_args()
    
    if args.output_path is None:
        model_name = Path(args.model_path).stem
        args.output_path = f"{model_name}.onnx"
    
    print(f"🚀 Exporting model to ONNX...")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {args.output_path}")
    print(f"   Input shape: {args.input_shape}")
    
    success = export_model_to_onnx(
        args.model_path, 
        args.output_path, 
        tuple(args.input_shape),
        args.opset_version
    )
    
    if success:
        print(f"\n🎉 Export completed successfully!")
        print(f"   Use with ONNX Runtime:")
        print(f"   import onnxruntime")
        print(f"   session = onnxruntime.InferenceSession('{args.output_path}')")
    else:
        print(f"\n❌ Export failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
