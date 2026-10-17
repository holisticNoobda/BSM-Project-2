"""Local architecture verification for all four models (spec section 6).
Run: python verify_architectures.py — exits 0 only if all checks pass.
"""
import sys

import torch

sys.path.insert(0, r"C:\BSM Project 2")
from models import MODEL_NAMES, format_verification, verify_architecture

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
results = []
for name in MODEL_NAMES:
    torch.manual_seed(42)
    results.append(verify_architecture(name))
print(format_verification(results))
ok = all(r["all_ok"] for r in results)
print("\nARCHITECTURE VERIFICATION:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
