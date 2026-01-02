import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

# -----------------------------
# Paths (your definitions)
# -----------------------------
i_dir = "../../examples/cavity"
time_cpu_paths = {
    "very_low": os.path.join(i_dir, "output_cpu_re1000_res_very_low", "solve_time.log"),
    "low":      os.path.join(i_dir, "output_cpu_re1000_res_low", "solve_time.log"),
    "mid":      os.path.join(i_dir, "output_cpu_re1000_res_mid", "solve_time.log"),
    "high":     os.path.join(i_dir, "output_cpu_re1000_res_high", "solve_time.log"),
}
time_gpu_paths = {
    "very_low": os.path.join(i_dir, "output_cuda_re1000_res_very_low", "solve_time.log"),
    "low":      os.path.join(i_dir, "output_cuda_re1000_res_low", "solve_time.log"),
    "mid":      os.path.join(i_dir, "output_cuda_re1000_res_mid", "solve_time.log"),
    "high":     os.path.join(i_dir, "output_cuda_re1000_res_high", "solve_time.log"),
}

o_dir = i_dir

# Mesh size h (given)
mesh_h = {
    "very_low": 0.047,
    "low":      0.031,
    "mid":      0.015,
    "high":     0.0077,
}

# -----------------------------
# Reader: parse solve_time.log
# -----------------------------
def read_solve_times(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    times = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip header or comment-like lines
            lower = line.lower()
            if "step" in lower or "solve time" in lower:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                t = float(parts[1])
                times.append(t)
            except ValueError:
                continue

    if len(times) == 0:
        raise ValueError(f"No solve times parsed from: {path}")

    return np.asarray(times, dtype=float)

def mean_solve_time(path: str) -> float:
    times = read_solve_times(path)
    return float(np.mean(times))

# -----------------------------
# Compute means per resolution
# -----------------------------
res_order = ["very_low", "low", "mid", "high"]

mean_cpu = {r: mean_solve_time(time_cpu_paths[r]) for r in res_order}
mean_gpu = {r: mean_solve_time(time_gpu_paths[r]) for r in res_order}

print("CPU mean solve time [s]:", mean_cpu)
print("GPU mean solve time [s]:", mean_gpu)

# -----------------------------
# Prepare x-axis as inverse mesh size (1/h)
# -----------------------------
x = np.array([1.0 / mesh_h[r] for r in res_order], dtype=float)

cpu_vals = np.array([mean_cpu[r] for r in res_order], dtype=float)
gpu_vals = np.array([mean_gpu[r] for r in res_order], dtype=float)

# Optional: sort by x (in case order changes later)
order = np.argsort(x)
x = x[order]
cpu_vals = cpu_vals[order]
gpu_vals = gpu_vals[order]
res_sorted = [res_order[i] for i in order]

# -----------------------------
# Plot 1: Mean solve time (CPU vs GPU) vs 1/h
# -----------------------------
plt.figure()
plt.plot(
    x, cpu_vals,
    marker="o",
    linestyle="-",
    label="CPU",
    color="red",
    markersize=6,
    markerfacecolor="none",   
    markeredgecolor="red"     
)

plt.plot(
    x, gpu_vals,
    marker="s",
    linestyle="--",
    label="GPU (CUDA)",
    color="blue",
    markersize=6,
    markerfacecolor="none",   
    markeredgecolor="blue"   
)


plt.xlabel(r"$1/h$")
plt.ylabel("Mean solve time [s]")
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

out1 = os.path.join(o_dir, "ave_solve_time.png")
plt.savefig(out1, dpi=900)
plt.close()

# -----------------------------
# Plot 2: Speedup (CPU/GPU) vs 1/h
# -----------------------------
speedup = cpu_vals / gpu_vals

plt.figure()
plt.plot(
    x, 
    speedup, 
    marker="o", 
    linestyle="-", 
    color="black", 
    markersize=6,
    markerfacecolor="none",   
    markeredgecolor="black"   
)

plt.xlabel(r"$1/h$")
plt.ylabel("Speedup")
plt.grid(True)
plt.tight_layout()

out2 = os.path.join(o_dir, "speedup.png")
plt.savefig(out2, dpi=900)
plt.close()

print(f"Saved: {out1}")
print(f"Saved: {out2}")

