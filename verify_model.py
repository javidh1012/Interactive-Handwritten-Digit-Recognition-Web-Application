"""
Verification script for model loading and compatibility checks.
Follows all verification steps specified in Section 3:
1. Verify that the checkpoint loads successfully.
2. Verify all four parameter tensors match.
3. Use strict=True when loading the weights.
4. Print the model architecture.
5. Run a dummy input with shape [1, 1, 28, 28].
6. Verify that the output shape is [1, 10].
7. Report any errors instead of hiding them.
"""

import sys
import torch
from model.cnn import make_model, resolve_model_weights_path, load_trained_model


def verify():
    print("=" * 60)
    print("Step 1: Locating checkpoint and loading state_dict...")
    weights_path = resolve_model_weights_path()
    print(f"Checkpoint located at: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    print("Checkpoint loaded successfully!")
    print(f"Number of keys in state_dict: {len(state_dict)}")

    print("\n" + "=" * 60)
    print("Step 2: Inspecting and verifying all parameter tensors:")
    expected_shapes = {
        "1.weight": (128, 784),
        "1.bias": (128,),
        "3.weight": (10, 128),
        "3.bias": (10,)
    }

    all_matched = True
    for key, shape in expected_shapes.items():
        if key not in state_dict:
            print(f"  [ERROR] Missing key: {key}")
            all_matched = False
        else:
            actual_shape = tuple(state_dict[key].shape)
            dtype = state_dict[key].dtype
            if actual_shape == shape:
                print(f"  [PASS] {key:10s} matches expected shape {shape} (dtype: {dtype})")
            else:
                print(f"  [FAIL] {key:10s} expected {shape}, found {actual_shape}")
                all_matched = False

    if not all_matched:
        raise ValueError("Parameter shape mismatch in checkpoint!")

    print("\n" + "=" * 60)
    print("Step 3: Loading model with strict=True and activation='ReLU'...")
    model = make_model(activation_name="ReLU")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded with strict=True successfully! No missing or unexpected keys.")

    print("\n" + "=" * 60)
    print("Step 4: Model architecture:")
    print(model)

    print("\n" + "=" * 60)
    print("Step 5: Testing with dummy input of shape [1, 1, 28, 28]...")
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    print(f"Dummy input shape: {list(dummy_input.shape)}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output raw logits: {output}")
    print(f"Output shape: {list(output.shape)}")

    print("\n" + "=" * 60)
    print("Step 6: Verifying output shape is [1, 10] and softmax probabilities...")
    assert output.shape == (1, 10), f"Expected shape (1, 10), got {output.shape}"
    print("  [PASS] Output shape strictly matches [1, 10]!")

    probabilities = torch.softmax(output, dim=1)
    prob_sum = probabilities.sum().item()
    print(f"Softmax probabilities sum: {prob_sum:.6f} (expected ~ 1.0)")
    assert abs(prob_sum - 1.0) < 1e-5, f"Softmax sum {prob_sum} deviates from 1.0"
    print("  [PASS] Probabilities sum to 1.0!")

    print("\n" + "=" * 60)
    print("ALL 7 VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"\n[FATAL ERROR] Verification failed: {e}", file=sys.stderr)
        sys.exit(1)

