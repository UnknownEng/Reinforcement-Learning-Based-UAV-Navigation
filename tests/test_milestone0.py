import sys
import time
import json

print("==================================================")
print("     MILESTONE 0: TOOLCHAIN & HARDWARE AUDIT      ")
print("==================================================")

results = {}

# 1. PyTorch Import & Version
try:
    import torch
    results["torch_import"] = "SUCCESS"
    results["torch_version"] = torch.__version__
except Exception as e:
    results["torch_import"] = f"FAILED: {e}"
    results["torch_version"] = "N/A"

# 2 & 3 & 4. CUDA Availability, Version, and Device Name
if results["torch_import"] == "SUCCESS":
    results["cuda_available"] = bool(torch.cuda.is_available())
    results["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    results["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_capability"] = torch.cuda.get_device_capability(0)
    else:
        results["gpu_name"] = "None"
        results["gpu_capability"] = "N/A"
else:
    results["cuda_available"] = False
    results["cuda_version"] = "N/A"
    results["gpu_count"] = 0
    results["gpu_name"] = "N/A"
    results["gpu_capability"] = "N/A"

# 5. Tensor Benchmark (CPU vs GPU comparison)
if results.get("cuda_available", False):
    device = torch.device("cuda:0")
    
    # Warmup
    x_gpu = torch.randn(2048, 2048, device=device)
    _ = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()
    
    # Benchmark GPU Matrix Multiplication
    num_runs = 50
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / num_runs * 1000.0  # ms
    
    # Benchmark CPU Matrix Multiplication
    x_cpu = torch.randn(2048, 2048, device="cpu")
    t0 = time.perf_counter()
    for _ in range(10):
        _ = torch.matmul(x_cpu, x_cpu)
    cpu_time = (time.perf_counter() - t0) / 10 * 1000.0  # ms
    
    # 6. Memory Allocation
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    
    results["benchmark"] = {
        "matrix_size": "2048x2048 (FP32)",
        "gpu_time_ms": round(gpu_time, 2),
        "cpu_time_ms": round(cpu_time, 2),
        "speedup_factor": round(cpu_time / gpu_time, 2),
        "allocated_mb": round(allocated_mb, 2),
        "reserved_mb": round(reserved_mb, 2),
        "max_allocated_mb": round(max_allocated_mb, 2)
    }
else:
    results["benchmark"] = "CUDA not available for benchmark"

# 7. Gymnasium Verification
try:
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    step_obs, reward, terminated, truncated, _ = env.step(action)
    env.close()
    results["gymnasium"] = {
        "status": "SUCCESS",
        "version": gym.__version__,
        "smoke_test": "CartPole-v1 reset & step verified"
    }
except Exception as e:
    results["gymnasium"] = {"status": f"FAILED: {e}"}

# 8. Stable-Baselines3 Verification
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    results["stable_baselines3"] = {
        "status": "SUCCESS",
        "version": sb3.__version__,
        "ppo_class": str(PPO)
    }
except Exception as e:
    results["stable_baselines3"] = {"status": f"FAILED: {e}"}

# 9. SciPy Verification
try:
    import scipy
    from scipy.spatial.transform import Rotation
    # Test quaternion / rotation conversion
    r = Rotation.from_euler('zyx', [90, 0, 0], degrees=True)
    quat = r.as_quat().tolist()
    results["scipy"] = {
        "status": "SUCCESS",
        "version": scipy.__version__,
        "rotation_test": f"Euler [90,0,0] -> Quat {quat}"
    }
except Exception as e:
    results["scipy"] = {"status": f"FAILED: {e}"}

# 10. TensorBoard Verification
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="/tmp/tb_test")
    writer.add_scalar("test/metric", 1.0, 0)
    writer.close()
    results["tensorboard"] = {
        "status": "SUCCESS",
        "version": tensorboard.__version__,
        "summary_writer_test": "Scalar write to /tmp/tb_test verified"
    }
except Exception as e:
    results["tensorboard"] = {"status": f"FAILED: {e}"}

print(json.dumps(results, indent=2))
