import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

# -----------------------------
# Helpers
# -----------------------------
def shift_to_01(s):
    # If data looks like [-0.5, 0.5], shift by +0.5. Otherwise leave as is.
    smin, smax = float(s.min()), float(s.max())
    if smin < -1e-6 and smax <= 0.500001:
        return s + 0.5
    return s

# -----------------------------
# Read config (YAML)
# -----------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

i_dir = cfg["io"]["solution"]["i_dir"]
u_csv_path = os.path.join(i_dir, "u_x_0.csv")  # u(y) line data
v_csv_path = os.path.join(i_dir, "v_y_0.csv")  # v(x) line data

i_ref_dir = cfg["io"]["reference"]["i_dir"]
u_csv_ref_path = os.path.join(i_ref_dir, "u_x_0.csv")
v_csv_ref_path = os.path.join(i_ref_dir, "v_y_0.csv")

o_dir = cfg["io"]["output"]["o_dir"]
o_filename = cfg["io"]["output"]["filename"]
os.makedirs(o_dir, exist_ok=True)
out_path = os.path.join(o_dir, o_filename)

# -----------------------------
# Read CSV files
# -----------------------------
df_u = pd.read_csv(u_csv_path)
df_v = pd.read_csv(v_csv_path)

df_u_ref = pd.read_csv(u_csv_ref_path)
df_v_ref = pd.read_csv(v_csv_ref_path)

# -----------------------------
# Column mapping 
# -----------------------------
y_col_u = "Points_1"
u_col   = "velocity_0"

x_col_v = "Points_0" 
v_col   = "velocity_1" 

# Reference (Ghia etc. might be y,u and x,v)
y_col_u_ref = "y"
u_col_ref   = "u" 

x_col_v_ref = "x" 
v_col_ref   = "v"

# -----------------------------
# Sort
# -----------------------------
df_u = df_u.sort_values(by=y_col_u).reset_index(drop=True)
df_v = df_v.sort_values(by=x_col_v).reset_index(drop=True)

df_u_ref = df_u_ref.sort_values(by=y_col_u_ref).reset_index(drop=True)
df_v_ref = df_v_ref.sort_values(by=x_col_v_ref).reset_index(drop=True)

# Shift coordinates if needed ([-0.5,0.5] -> [0,1])
y_sol = shift_to_01(df_u[y_col_u])
x_sol = shift_to_01(df_v[x_col_v])

y_ref = shift_to_01(df_u_ref[y_col_u_ref])
x_ref = shift_to_01(df_v_ref[x_col_v_ref])

# -----------------------------
# Plot (one square with 4-side labels)
#   bottom: x, left: v
#   top: u, right: y
# -----------------------------
fig = plt.figure(figsize=(5.5, 5.0))

# Base axes: v(x)  -> x bottom, v left
ax_vx = fig.add_subplot(111)
ax_vx.plot(
    x_sol, df_v[v_col],
    color="black", linestyle="--", linewidth=1.0,
    label="Solution"
)
ax_vx.plot(
    x_ref, df_v_ref[v_col_ref],
    linestyle="None", marker="s", markersize=7,
    markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.0,
    label="Ghia et al."
)

ax_vx.set_xlabel("x", fontsize=14)
ax_vx.set_ylabel("v", fontsize=14)
ax_vx.tick_params(axis="both", which="major", labelsize=12)
ax_vx.grid(False)
ax_vx.set_xlim(-0.05, 1.05)   
ax_vx.set_ylim(-0.7, 0.5)   

# Overlay axes: u(y) -> u top, y right
pos = ax_vx.get_position()
ax_uy = fig.add_axes(pos, frameon=False)

ax_uy.plot(
    df_u[u_col], y_sol,
    color="black", linestyle="--", linewidth=1.0
)
ax_uy.plot(
    df_u_ref[u_col_ref], y_ref,
    linestyle="None", marker="s", markersize=7,
    markerfacecolor="none", markeredgecolor= "black", markeredgewidth=1.0
)

# Put axes on top/right only
ax_uy.xaxis.set_label_position("top")
ax_uy.xaxis.tick_top()
ax_uy.yaxis.set_label_position("right")
ax_uy.yaxis.tick_right()

ax_uy.set_xlabel("u", fontsize=14)
ax_uy.set_ylabel("y", fontsize=14)
ax_uy.tick_params(axis="both", which="major", labelsize=12)
ax_uy.set_xlim(-0.6, 1.1)   
ax_uy.set_ylim(-0.05, 1.05)   

# Hide bottom/left of overlay axes (to avoid duplicate ticks)
ax_uy.spines["bottom"].set_visible(False)
ax_uy.spines["left"].set_visible(False)

# Optional: if you want overlay grid off (usually nicer)
ax_uy.grid(False)

# Single legend (use base axes handles)
ax_vx.legend(
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    loc="center",
    bbox_to_anchor=(0.5, 0.2)  # ← y を少し下げる
)

plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
