import onnx
import onnxsim
import torch
from models.CountingAnything import CountingAnything


def export_onnx_model(model_ckpt, CFG, onnx_path, simplify: bool = True):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model_ckpt (str): Path to the model checkpoint.
        example_inputs (tuple): Example inputs for the model.
        onnx_path (str): Path to save the exported ONNX model.
    """

    CFG["resume_path"] = model_ckpt
    model = CountingAnything(CFG)
    example_inputs = torch.randn((3, 3, *CFG["img_size"]))
    print(f"Input shape: {example_inputs.shape}")

    # Export the model
    model.to_onnx(
        onnx_path,
        example_inputs,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"}
        },
    )

    # if simplify:
        # onnx_model_simplify, check = onnxsim.simplify(onnx.load(onnx_path))
        # onnx.save(onnx_model_simplify, onnx_path)
        # print(f'Simplify onnx model {check}...')

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)