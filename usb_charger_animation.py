"""
スケルトンUSB充電器 回路シミュレーション（改善版）
=====================================================
対応内容:
1. 日本語文字化け対策
2. 文字重なりの抑制
3. グラフの軸修正
4. 測定点（どこの電圧・電流か）を明確化
5. JIS風の回路図を追加
6. 重さを少し軽減（性能優先）
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Arc

warnings.filterwarnings("ignore")

# ============================================================
# フォント設定（日本語文字化け対策）
# ============================================================
def find_jp_font():
    candidates = [
        "Yu Gothic", "Meiryo", "MS Gothic", "MS PGothic",
        "Hiragino Sans", "Hiragino Kaku Gothic Pro",
        "Noto Sans CJK JP", "Noto Sans JP",
        "IPAGothic", "IPAPGothic", "TakaoPGothic", "VL Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

JP_FONT = find_jp_font()
if JP_FONT:
    matplotlib.rcParams["font.family"] = JP_FONT
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# 色設定
# ============================================================
BG      = "#0D1117"
PANEL   = "#161B22"
BORDER  = "#30363D"
WHITE   = "#E6EDF3"
GRAY    = "#8B949E"
CORAL   = "#FF6F61"
GOLD    = "#FFC857"
TEAL    = "#00838F"
MINT    = "#A8E6CF"
PLUM    = "#7B5EA7"
SKY     = "#4ECDC4"
ROSE    = "#FF6B9D"
LIME    = "#C5E99B"

# ============================================================
# 保存先
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CKT = os.path.join(OUT_DIR, "usb_charger_circuit_diagram_jis.png")
OUT_SIM = os.path.join(OUT_DIR, "usb_charger_all_graphs_fixed.png")

# ============================================================
# シミュレーション条件
# ============================================================
FS      = 200_000     # 1,000,000 → 200,000 に軽量化
T_SIM   = 0.05
t       = np.linspace(0, T_SIM, int(FS * T_SIM), endpoint=False)
dt      = t[1] - t[0]

F_AC    = 60.0
V_AC    = 100.0
V_PEAK  = V_AC * np.sqrt(2)

F_SW    = 65_000.0
V_OUT   = 5.0
I_LOAD1 = 1.2
I_LOAD2 = 1.0
I_LOAD  = I_LOAD1 + I_LOAD2
R_LOAD  = V_OUT / I_LOAD

# ============================================================
# 部品定数
# ============================================================
R = {
    "R1":  1e6,
    "R2":  1e6,
    "R3":  10.0,
    "R4":  47e3,
    "R5":  10e3,
    "R6":  1.0,
    "R7":  100e3,
    "R8":  22.0,
    "R9":  4.7e3,
    "R10": 100.0,
    "R11": 10.0,
    "R12": 10.0,
}

C = {
    "C1": 0.1e-6,
    "C2": 330e-6,
    "C3": 1e-9,
    "C4": 100e-12,
    "C5": 47e-6,
    "C6": 680e-6,
    "C7": 680e-6,
    "C8": 4.7e-9,
}

L = {
    "L1": 1.0e-3,
    "L2": 470e-6,
    "L3": 10e-6,
}

VF = {
    "D1": 0.7,
    "D2": 0.7,
    "D3": 0.45,
    "D4": 0.45,
    "D5": 5.1,
}

# ============================================================
# 波形計算
# ============================================================
v_ac = V_PEAK * np.sin(2 * np.pi * F_AC * t)

# EMIフィルタ後
fc_emi = 10e3
alpha = 1 / (1 + 1 / (2 * np.pi * fc_emi * dt))
v_emi = np.zeros_like(v_ac)
v_emi[0] = v_ac[0]
for i in range(1, len(t)):
    v_emi[i] = alpha * v_emi[i - 1] + (1 - alpha) * v_ac[i]

# 全波整流
v_rect = np.clip(np.abs(v_emi) - 2 * VF["D1"], 0, None)

# バルクコンデンサ平滑
tau_bulk = R_LOAD * C["C2"]
v_bulk = np.zeros_like(v_rect)
v_bulk[0] = V_PEAK - 2 * VF["D1"]
for i in range(1, len(t)):
    if v_rect[i] > v_bulk[i - 1]:
        v_bulk[i] = v_rect[i]
    else:
        v_bulk[i] = v_bulk[i - 1] * np.exp(-dt / tau_bulk)

# スイッチング信号
sw_signal = (np.sin(2 * np.pi * F_SW * t) > 0.3).astype(float)

# 一次側電流
i_primary = np.zeros_like(t)
L_mag = 800e-6
for i in range(1, len(t)):
    if sw_signal[i] > 0.5:
        i_primary[i] = i_primary[i - 1] + (v_bulk[i] / L_mag) * dt
        i_primary[i] = min(i_primary[i], 0.8)
    else:
        i_primary[i] = max(0, i_primary[i - 1] - 0.001)

# 二次側電圧
n_ratio = V_OUT / (V_PEAK - 2 * VF["D1"])
v_sec_raw = v_bulk * n_ratio * sw_signal

# 出力電圧
v_out1 = np.zeros_like(t)
v_out2 = np.zeros_like(t)
v_out1[0] = V_OUT
v_out2[0] = V_OUT
for i in range(1, len(t)):
    v_in1 = max(0, v_sec_raw[i] - VF["D3"])
    if v_in1 > v_out1[i - 1]:
        v_out1[i] = v_out1[i - 1] + (v_in1 - v_out1[i - 1]) * 0.1
    else:
        v_out1[i] = v_out1[i - 1] - (I_LOAD1 / C["C6"]) * dt
    v_out1[i] = np.clip(v_out1[i], 4.5, 5.5)

    v_in2 = max(0, v_sec_raw[i] - VF["D4"])
    if v_in2 > v_out2[i - 1]:
        v_out2[i] = v_out2[i - 1] + (v_in2 - v_out2[i - 1]) * 0.1
    else:
        v_out2[i] = v_out2[i - 1] - (I_LOAD2 / C["C7"]) * dt
    v_out2[i] = np.clip(v_out2[i], 4.5, 5.5)

# 抵抗
v_R = {
    "R1":  np.abs(v_ac) / 2,
    "R2":  np.abs(v_ac) / 2,
    "R3":  i_primary * R["R3"],
    "R4":  (v_out1 - 2.5) * (R["R4"] / (R["R4"] + R["R5"])),
    "R5":  v_out1 * R["R5"] / (R["R4"] + R["R5"]),
    "R6":  i_primary * sw_signal * R["R6"],
    "R7":  np.ones_like(t) * 1.2,
    "R8":  i_primary * R["R8"],
    "R9":  np.ones_like(t) * 2.5,
    "R10": i_primary * sw_signal * R["R10"] * 0.1,
    "R11": np.ones_like(t) * (I_LOAD1 * R["R11"]),
    "R12": np.ones_like(t) * (I_LOAD2 * R["R12"]),
}
i_R = {k: v_R[k] / R[k] for k in R}

# コンデンサ
v_C = {
    "C1": v_ac * 0.05,
    "C2": v_bulk,
    "C3": i_primary * sw_signal * 50,
    "C4": np.ones_like(t) * 12.0,
    "C5": np.ones_like(t) * 12.0,
    "C6": v_out1,
    "C7": v_out2,
    "C8": v_ac * 0.008,
}
i_C = {k: C[k] * np.gradient(v_C[k], dt) for k in C}

# インダクタ
dvemi = np.gradient(v_emi, dt)
v_L = {
    "L1": np.clip(dvemi * L["L1"] * 0.0008, -20, 20),
    "L2": np.clip(dvemi * L["L2"] * 0.0004, -10, 10),
    "L3": np.gradient(v_out1, dt) * L["L3"],
}
i_L = {
    "L1": np.cumsum(v_L["L1"]) * dt / L["L1"],
    "L2": np.cumsum(v_L["L2"]) * dt / L["L2"],
    "L3": I_LOAD * np.ones_like(t) + np.cumsum(v_L["L3"]) * dt / L["L3"] * 0.001,
}

# ダイオード
i_D = {
    "D1": np.clip(v_rect / R_LOAD, 0, 3.0),
    "D2": np.clip(v_rect / R_LOAD, 0, 3.0),
    "D3": np.clip((v_sec_raw - VF["D3"]) / (R_LOAD / 2), 0, 3.0) * sw_signal,
    "D4": np.clip((v_sec_raw - VF["D4"]) / (R_LOAD / 2), 0, 3.0) * sw_signal,
    "D5": np.where(v_bulk * 1.3 > (v_bulk + VF["D5"]), 0.1, 0.0),
}

# MOSFET / 制御
v_gate = sw_signal * 10.0
v_drain = v_bulk * (1 - sw_signal) + sw_signal * 0.5

err_ic = V_OUT - (v_out1 + v_out2) / 2
duty = np.clip(0.3 + np.cumsum(err_ic) * dt * 8, 0.05, 0.8)

# NTC
T_ntc = np.clip(25.0 + (i_primary**2 * R["R3"] * 0.001).cumsum() * dt * 8e3, 25, 80)
R_ntc = R["R3"] * np.exp(3950 * (1 / (T_ntc + 273.15) - 1 / 298.15))

# ============================================================
# プロット用間引き
# ============================================================
STEP = 100
ts = t[::STEP] * 1000  # ms

def ds(arr):
    return arr[::STEP]

# ============================================================
# 共通ユーティリティ
# ============================================================
def data_limits(*arrays, pad=0.12, symmetric=False, floor=None, ceil=None):
    vals = np.concatenate([np.ravel(np.asarray(a)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0, 1)
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if symmetric:
        m = max(abs(mn), abs(mx))
        if m == 0:
            m = 1.0
        if floor is not None:
            m = max(m, abs(floor))
        if ceil is not None:
            m = max(m, abs(ceil))
        return (-m * (1 + pad), m * (1 + pad))
    span = mx - mn
    if span == 0:
        span = abs(mx) if mx != 0 else 1.0
    lo = mn - span * pad
    hi = mx + span * pad
    if floor is not None:
        lo = min(lo, floor)
    if ceil is not None:
        hi = max(hi, ceil)
    return (lo, hi)

def style_ax(ax, title, ylabel, xlabel=None, show_xlabels=False):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=7, pad=3, loc="left")
    ax.set_ylabel(ylabel, color=WHITE, fontsize=6.6, labelpad=2)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=WHITE, fontsize=6.6, labelpad=2)
    elif not show_xlabels:
        ax.tick_params(labelbottom=False)
    ax.tick_params(colors=WHITE, labelsize=6)
    ax.grid(axis="y", color=BORDER, lw=0.4, alpha=0.45)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    return ax

def mark_meas(ax, text, color=GOLD, loc="upper right"):
    x = 0.98 if loc == "upper right" else 0.02
    ha = "right" if loc == "upper right" else "left"
    ax.text(
        x, 0.96, text, transform=ax.transAxes,
        ha=ha, va="top", fontsize=6.2, color=color,
        bbox=dict(facecolor="#1C2128", edgecolor=color, lw=0.6, boxstyle="round,pad=0.18"),
        zorder=10
    )

# ============================================================
# JIS風 回路図
# ============================================================
def draw_resistor(ax, x1, x2, y, amp=0.12, nzig=8, color=WHITE, lw=1.2):
    xs = np.linspace(x1, x2, nzig * 2 + 1)
    ys = np.full_like(xs, y, dtype=float)
    for i in range(1, len(xs)-1):
        ys[i] = y + amp if i % 2 else y - amp
    ax.plot([x1, xs[1]], [y, ys[1]], color=color, lw=lw)
    ax.plot(xs[1:-1], ys[1:-1], color=color, lw=lw)
    ax.plot([xs[-2], x2], [ys[-2], y], color=color, lw=lw)

def draw_capacitor(ax, x, y1, y2, plate_gap=0.08, color=WHITE, lw=1.2):
    ym = (y1 + y2) / 2
    ax.plot([x, x], [y1, ym - plate_gap], color=color, lw=lw)
    ax.plot([x - 0.12, x + 0.12], [ym - plate_gap, ym - plate_gap], color=color, lw=lw)
    ax.plot([x - 0.12, x + 0.12], [ym + plate_gap, ym + plate_gap], color=color, lw=lw)
    ax.plot([x, x], [ym + plate_gap, y2], color=color, lw=lw)

def draw_inductor(ax, x1, x2, y, n=4, color=WHITE, lw=1.2):
    xs = np.linspace(x1, x2, n * 60)
    span = (x2 - x1) / n
    ax.plot([x1, x1 + 0.05], [y, y], color=color, lw=lw)
    for i in range(n):
        xc = x1 + (i + 0.5) * span
        arc = Arc((xc, y), span * 0.95, 0.38, theta1=0, theta2=180, color=color, lw=lw)
        ax.add_patch(arc)
    ax.plot([x2 - 0.05, x2], [y, y], color=color, lw=lw)

def draw_diode(ax, x1, x2, y, color=WHITE, lw=1.2):
    xm = (x1 + x2) / 2
    tri = Polygon([[x1, y - 0.18], [x1, y + 0.18], [xm - 0.03, y]], closed=True, fill=False, edgecolor=color, lw=lw)
    ax.add_patch(tri)
    ax.plot([xm + 0.03, xm + 0.03], [y - 0.22, y + 0.22], color=color, lw=lw)
    ax.plot([x1 - 0.05, x1], [y, y], color=color, lw=lw)
    ax.plot([xm + 0.03, x2], [y, y], color=color, lw=lw)

def draw_box(ax, x, y, w, h, label, sublabel="", color="#264653", alpha=0.88):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=WHITE, linewidth=1.0, alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
            color="black", fontsize=8.5, fontweight="bold")
    if sublabel:
        ax.text(x + w/2, y + h*0.28, sublabel, ha="center", va="center",
                color="black", fontsize=6.5)

fig_ckt = plt.figure(figsize=(22, 11), facecolor=BG)
ax_ckt = fig_ckt.add_subplot(111)
ax_ckt.set_facecolor(BG)
ax_ckt.set_xlim(0, 24)
ax_ckt.set_ylim(0, 10)
ax_ckt.axis("off")
ax_ckt.set_title(
    "USB充電器 推定回路図（JIS風簡易図）\nAC100V/60Hz → EMI → 整流 → バルクC → フライバック → USB-A×2",
    color=WHITE, fontsize=13, fontweight="bold", pad=12
)

# GND
ax_ckt.plot([0.5, 23.4], [1.0, 1.0], color=GRAY, lw=1.0, ls="--")
ax_ckt.text(0.2, 1.0, "GND", color=GRAY, fontsize=8, va="center")

# AC source
draw_box(ax_ckt, 0.5, 3.9, 1.8, 2.1, "AC電源", "100V / 60Hz", color="#444444")
ax_ckt.add_patch(Circle((1.4, 4.95), 0.42, fill=False, ec=WHITE, lw=1.1))
ax_ckt.text(1.4, 4.95, "~", color=WHITE, ha="center", va="center", fontsize=12, fontweight="bold")
ax_ckt.plot([2.3, 3.0], [5.6, 5.6], color=WHITE, lw=1.2)
ax_ckt.plot([2.3, 3.0], [4.3, 4.3], color=WHITE, lw=1.2)

# NTC + EMI
draw_resistor(ax_ckt, 3.0, 4.0, 5.6, amp=0.11, color=CORAL)
ax_ckt.text(3.5, 5.95, "R3 NTC", color=CORAL, fontsize=7, ha="center")
ax_ckt.plot([4.0, 4.6], [5.6, 5.6], color=WHITE, lw=1.2)
draw_inductor(ax_ckt, 4.6, 6.0, 5.6, n=4, color=TEAL)
ax_ckt.text(5.3, 5.95, "L1", color=TEAL, fontsize=7, ha="center")
draw_capacitor(ax_ckt, 5.3, 5.6, 1.0, color=GOLD)
ax_ckt.text(5.45, 4.95, "C1", color=GOLD, fontsize=7, ha="left")
draw_inductor(ax_ckt, 6.0, 7.4, 5.6, n=3, color=TEAL)
ax_ckt.text(6.7, 5.95, "L2", color=TEAL, fontsize=7, ha="center")
draw_capacitor(ax_ckt, 7.0, 5.6, 1.0, color=LIME)
ax_ckt.text(7.15, 4.95, "C8", color=LIME, fontsize=7, ha="left")
ax_ckt.plot([7.4, 8.0], [5.6, 5.6], color=WHITE, lw=1.2)
ax_ckt.text(5.6, 7.25, "EMIフィルタ", color=MINT, fontsize=8, ha="center")

# Bridge rectifier
draw_box(ax_ckt, 8.0, 4.0, 2.0, 2.6, "整流ブリッジ", "D1/D2 等価", color="#5C4A8A")
draw_diode(ax_ckt, 8.3, 9.7, 5.8, color=WHITE)
draw_diode(ax_ckt, 8.3, 9.7, 4.8, color=WHITE)
ax_ckt.text(9.0, 6.85, "D1,D2", color=WHITE, fontsize=7, ha="center")
ax_ckt.plot([10.0, 10.6], [5.6, 5.6], color=WHITE, lw=1.2)
ax_ckt.text(9.0, 3.65, "交流入力", color=WHITE, fontsize=7, ha="center")
ax_ckt.text(10.4, 5.95, "DC+", color=WHITE, fontsize=7)
ax_ckt.text(10.4, 4.55, "DC-", color=WHITE, fontsize=7)

# Bulk capacitor
draw_capacitor(ax_ckt, 10.6, 5.6, 1.0, color=SKY)
ax_ckt.text(10.8, 4.95, "C2 330µF", color=SKY, fontsize=7, ha="left")
ax_ckt.plot([10.6, 11.8], [5.6, 5.6], color=WHITE, lw=1.2)
ax_ckt.plot([10.6, 11.8], [1.0, 1.0], color=WHITE, lw=1.2)
ax_ckt.text(11.2, 6.15, "DCバス", color=SKY, fontsize=8, ha="center")

# Flyback stage
draw_box(ax_ckt, 12.0, 3.8, 3.0, 3.0, "T1 + Q1", "フライバック", color="#3D5A3E")
draw_inductor(ax_ckt, 12.3, 13.3, 5.9, n=3, color=WHITE)
draw_inductor(ax_ckt, 14.2, 15.2, 5.2, n=3, color=WHITE)
ax_ckt.plot([13.3, 14.2], [5.9, 5.9], color=WHITE, lw=1.0)
ax_ckt.plot([13.3, 14.2], [5.2, 5.2], color=WHITE, lw=1.0)
ax_ckt.text(12.6, 6.35, "一次", color=WHITE, fontsize=7)
ax_ckt.text(14.6, 5.65, "二次", color=WHITE, fontsize=7)
ax_ckt.plot([12.0, 12.0], [1.0, 3.8], color=WHITE, lw=1.2)
ax_ckt.text(12.15, 3.05, "Q1", color=WHITE, fontsize=7)
ax_ckt.text(12.15, 2.65, "MOSFET", color=WHITE, fontsize=7)

# Control IC
draw_box(ax_ckt, 12.0, 1.7, 1.9, 1.4, "U3", "PWM IC", color="#5C4A1A")
ax_ckt.text(12.95, 2.45, "AP3706相当", color="black", fontsize=6.8, ha="center")
ax_ckt.annotate("", xy=(12.7, 3.1), xytext=(12.7, 3.85),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.3))
ax_ckt.text(12.83, 3.45, "Gate", color=GOLD, fontsize=7, va="center")
draw_box(ax_ckt, 14.1, 1.7, 1.4, 1.4, "R4/R5", "FB分圧", color="#4A4A4A")
ax_ckt.annotate("", xy=(14.1, 2.35), xytext=(13.9, 2.35),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
ax_ckt.text(14.8, 2.95, "FB", color=GRAY, fontsize=7, ha="center")

# Secondary rectification + outputs
draw_box(ax_ckt, 16.0, 4.7, 1.2, 0.9, "D3", "P1用", color=PLUM)
draw_box(ax_ckt, 16.0, 3.2, 1.2, 0.9, "D4", "P2用", color="#8A5C6B")
draw_capacitor(ax_ckt, 18.2, 5.6, 1.0, color=SKY)
draw_capacitor(ax_ckt, 18.2, 3.9, 1.0, color=SKY)
ax_ckt.text(18.35, 4.95, "C6 680µF", color=SKY, fontsize=7, ha="left")
ax_ckt.text(18.35, 3.25, "C7 680µF", color=SKY, fontsize=7, ha="left")

draw_box(ax_ckt, 19.5, 4.55, 2.4, 1.3, "USB-A Port1", "5V / 1.2A", color="#336633")
draw_box(ax_ckt, 19.5, 2.95, 2.4, 1.3, "USB-A Port2", "5V / 1.0A", color="#336633")
ax_ckt.text(18.9, 6.15, "二次整流", color=MINT, fontsize=8, ha="center")

# Snubber
draw_box(ax_ckt, 12.0, 6.95, 2.2, 0.9, "スナバ", "R10 + C3 + D5", color="#6B4E2A")
ax_ckt.text(13.1, 7.35, "スパイク吸収", color="black", fontsize=6.8, ha="center")

# Measurement note
legend_items = [
    ("①", "R3両端電圧 / NTC降下", GOLD),
    ("②", "EMIフィルタ後電圧", MINT),
    ("③", "整流ブリッジ出力電圧", CORAL),
    ("④", "DCバス電圧 (C2両端)", SKY),
    ("⑤", "T1一次側電流", TEAL),
    ("⑥", "T1二次側電圧", MINT),
    ("⑦", "Q1 Vds", ROSE),
    ("⑧", "PWMデューティ", GOLD),
    ("⑨", "FB電圧", GRAY),
    ("⑩", "D3/D4電流", SKY),
    ("⑪", "USB Port1 VBUS", MINT),
    ("⑫", "USB Port2 VBUS", ROSE),
]
for i, (num, txt, col) in enumerate(legend_items):
    y = 0.55 - (i % 6) * 0.22
    x = 0.55 if i < 6 else 8.0
    ax_ckt.add_patch(Rectangle((x, y), 0.18, 0.12, facecolor=col, edgecolor=col))
    ax_ckt.text(x + 0.25, y + 0.06, f"{num} {txt}", color=WHITE, fontsize=6.8, va="center")

fig_ckt.tight_layout()
fig_ckt.savefig(OUT_CKT, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig_ckt)

# ============================================================
# グラフ作成
# ============================================================
fig = plt.figure(figsize=(26, 42), facecolor=BG)
fig.suptitle(
    "スケルトンUSB充電器 回路シミュレーション（改善版）\n"
    "AC100V/60Hz → EMI → 整流 → フライバック → USB-A×2",
    color=WHITE, fontsize=15, fontweight="bold", y=0.993
)

gs = gridspec.GridSpec(
    11, 4, figure=fig,
    hspace=0.78, wspace=0.38,
    top=0.975, bottom=0.02, left=0.05, right=0.985
)

axes_all = []

# ROW 0: 主要フロー
ax0 = fig.add_subplot(gs[0, :])
style_ax(
    ax0,
    "主要電圧フロー\n[測定点: ①AC入力 → ②EMI後 → ③整流後 → ④DCバス]",
    "電圧 [V]"
)
ax0.plot(ts, ds(v_ac),   color=CORAL, lw=1.2, label="v_ac  [① AC入力]")
ax0.plot(ts, ds(v_emi),  color=GOLD,  lw=1.1, label="v_emi [② EMI後]", alpha=0.9)
ax0.plot(ts, ds(v_rect), color=TEAL,  lw=1.1, label="v_rect [③ 整流後]", alpha=0.9)
ax0.plot(ts, ds(v_bulk), color=MINT,  lw=1.3, label="v_bulk [④ DCバス]")
ax0.set_ylim(*data_limits(ds(v_ac), ds(v_emi), ds(v_rect), ds(v_bulk), pad=0.15))
ax0.legend(loc="upper right", fontsize=7.3, facecolor="#1C2128", labelcolor=WHITE, ncol=2, framealpha=0.92)
mark_meas(ax0, "測定点①②③④")
axes_all.append(ax0)

# ROW 1-2: 抵抗
R_info = {
    "R1": "R1 (1MΩ)\nAC入力ブリーダー上側\n測定: L–GND間",
    "R2": "R2 (1MΩ)\nAC入力ブリーダー下側\n測定: N–GND間",
    "R3": "R3 / NTC (10Ω)\n突入電流制限\n測定: ①NTC両端",
    "R4": "R4 (47kΩ)\nFB分圧 上側\n測定: ⑨FB–Vout",
    "R5": "R5 (10kΩ)\nFB分圧 下側\n測定: ⑨FB–GND",
    "R6": "R6 (1Ω)\nMOSFETゲート抵抗\n測定: Gate–IC",
    "R7": "R7 (100kΩ)\nソフトスタート\n測定: Cssノード",
    "R8": "R8 (22Ω)\n電流検出抵抗\n測定: ⑤Q1ソース–GND",
    "R9": "R9 (4.7kΩ)\nバイアス抵抗\n測定: VCC–FB",
    "R10": "R10 (100Ω)\nスナバ抵抗\n測定: スナバ両端",
    "R11": "R11 (10Ω)\n出力ダンピング P1\n測定: ⑪C6–USB端子",
    "R12": "R12 (10Ω)\n出力ダンピング P2\n測定: ⑫C7–USB端子",
}
R_keys = list(R.keys())
R_cols = [CORAL, GOLD, TEAL, MINT, PLUM, SKY, ROSE, LIME, CORAL, GOLD, TEAL, MINT]

for idx, rk in enumerate(R_keys):
    row = 1 + idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])
    style_ax(ax, R_info[rk], "電圧 [V]")
    y = ds(v_R[rk])
    ax.plot(ts, y, color=R_cols[idx], lw=1.0)
    ax.fill_between(ts, y, color=R_cols[idx], alpha=0.11)
    ax.set_ylim(*data_limits(y, pad=0.18))
    ax2 = ax.twinx()
    iy = ds(i_R[rk]) * 1000
    ax2.plot(ts, iy, color=R_cols[idx], lw=0.7, ls=":", alpha=0.65)
    ax2.set_ylabel("電流 [mA]", color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)
    mark_meas(ax, f"測定点: {rk}")
    axes_all.append(ax)

# ROW 3: コンデンサまとめ
ax_ci = fig.add_subplot(gs[3, :2])
style_ax(ax_ci, "全コンデンサの充放電電流\n[C·dV/dt]", "電流 [mA]")
for ck, col in zip(C.keys(), [CORAL, TEAL, GOLD, MINT, PLUM, SKY, ROSE, LIME]):
    ax_ci.plot(ts, np.clip(ds(i_C[ck]) * 1000, -400, 400), lw=0.8, label=ck, color=col)
ax_ci.axhline(0, color=GRAY, lw=0.5, ls="--")
ax_ci.legend(fontsize=7, ncol=2, facecolor="#1C2128", labelcolor=WHITE)
ax_ci.set_ylim(*data_limits(*[np.clip(ds(i_C[ck]) * 1000, -400, 400) for ck in C.keys()], pad=0.14))
axes_all.append(ax_ci)

ax_cv = fig.add_subplot(gs[3, 2:])
style_ax(ax_cv, "全コンデンサの電圧\n[どのノードの電圧かを併記]", "電圧 [V]")
for ck, col in zip(C.keys(), [CORAL, TEAL, GOLD, MINT, PLUM, SKY, ROSE, LIME]):
    ax_cv.plot(ts, ds(v_C[ck]), lw=0.9, label=ck, color=col)
ax_cv.legend(fontsize=7, ncol=2, facecolor="#1C2128", labelcolor=WHITE)
ax_cv.set_ylim(*data_limits(*[ds(v_C[ck]) for ck in C.keys()], pad=0.12))
axes_all.append(ax_cv)

# ROW 4: 個別コンデンサ電圧
C_keys = list(C.keys())
C_cols = [CORAL, TEAL, GOLD, MINT, PLUM, SKY, ROSE, LIME]
C_titles = {
    "C1": "C1 (0.1µF)\nCX EMIコンデンサ\n測定: ACライン間ノイズ",
    "C2": "C2 (330µF)\nバルク平滑\n測定: ④DCバス–GND",
    "C3": "C3 (1nF)\nスナバC\n測定: スナバノード–GND",
    "C4": "C4 (100pF)\nブートストラップC\n測定: VCC供給ノード",
    "C5": "C5 (47µF)\n補助電源平滑\n測定: 制御IC VCC",
    "C6": "C6 (680µF)\n出力平滑 P1\n測定: ⑪USB Port1 VBUS",
    "C7": "C7 (680µF)\n出力平滑 P2\n測定: ⑫USB Port2 VBUS",
    "C8": "C8 (4.7nF)\nYコンデンサ\n測定: ACライン–PE相当",
}
for idx, ck in enumerate(C_keys):
    ax = fig.add_subplot(gs[4, idx % 4])
    style_ax(ax, C_titles[ck], "電圧 [V]")
    y = ds(v_C[ck])
    ax.plot(ts, y, color=C_cols[idx], lw=1.0)
    ax.fill_between(ts, y, color=C_cols[idx], alpha=0.11)
    ax.set_ylim(*data_limits(y, pad=0.18))
    mark_meas(ax, f"測定点: {ck}")
    axes_all.append(ax)

# ROW 5: 個別コンデンサ電流 + まとめ
for idx, ck in enumerate(C_keys[:4]):
    ax = fig.add_subplot(gs[5, idx])
    style_ax(ax, f"{ck} の充放電電流\n測定: {ck} の電流", "電流 [mA]")
    y = np.clip(ds(i_C[ck]) * 1000, -500, 500)
    ax.plot(ts, y, color=C_cols[idx], lw=1.0)
    ax.set_ylim(*data_limits(y, pad=0.15))
    mark_meas(ax, f"測定点: {ck}")
    axes_all.append(ax)

ax_ci2 = fig.add_subplot(gs[5, 0:4], frame_on=False)
ax_ci2.set_visible(False)

for idx, ck in enumerate(C_keys[4:], start=4):
    ax = fig.add_subplot(gs[5, idx - 2])
    style_ax(ax, f"{ck} の充放電電流\n測定: {ck} の電流", "電流 [mA]")
    y = np.clip(ds(i_C[ck]) * 1000, -500, 500)
    ax.plot(ts, y, color=C_cols[idx], lw=1.0)
    ax.set_ylim(*data_limits(y, pad=0.15))
    mark_meas(ax, f"測定点: {ck}")
    axes_all.append(ax)

# ROW 6: インダクタ
L_titles = {
    "L1": "L1 (1mH)\nコモンモードチョーク\n測定: ACライン貫通ノード",
    "L2": "L2 (470µH)\n差動モードチョーク\n測定: L2両端",
    "L3": "L3 (10µH)\n出力インダクタ\n測定: D3/D4後–C6前",
}
L_cols = [TEAL, CORAL, GOLD]

for idx, lk in enumerate(L.keys()):
    ax = fig.add_subplot(gs[6, idx])
    style_ax(ax, L_titles[lk], "電圧 [V]")
    y = np.clip(ds(v_L[lk]), -30, 30)
    ax.plot(ts, y, color=L_cols[idx], lw=1.0)
    ax.fill_between(ts, y, color=L_cols[idx], alpha=0.11)
    ax.axhline(0, color=GRAY, lw=0.45, ls="--")
    ax.set_ylim(*data_limits(y, symmetric=True, pad=0.12))
    ax2 = ax.twinx()
    iy = np.clip(ds(i_L[lk]), -5, 5)
    ax2.plot(ts, iy, color=L_cols[idx], lw=0.8, ls="--", alpha=0.7)
    ax2.set_ylabel("電流 [A]", color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)
    mark_meas(ax, f"測定点: {lk}")
    axes_all.append(ax)

ax_lv = fig.add_subplot(gs[6, 3])
style_ax(ax_lv, "全インダクタの電圧比較", "電圧 [V]")
for lk, col in zip(L.keys(), L_cols):
    ax_lv.plot(ts, np.clip(ds(v_L[lk]), -30, 30), lw=0.9, label=lk, color=col)
ax_lv.axhline(0, color=GRAY, lw=0.45, ls="--")
ax_lv.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
axes_all.append(ax_lv)

# ROW 7: ダイオード
D_titles = {
    "D1": "D1 (0.7V)\n整流ブリッジ 上側\n測定: ブリッジ出力電流",
    "D2": "D2 (0.7V)\n整流ブリッジ 下側\n測定: ブリッジ出力電流",
    "D3": "D3 (0.45V)\n二次整流 P1\n測定: ⑩D3電流",
    "D4": "D4 (0.45V)\n二次整流 P2\n測定: ⑩D4電流",
}
D_cols = [MINT, PLUM, SKY, ROSE]

for idx, dk in enumerate(["D1", "D2", "D3", "D4"]):
    ax = fig.add_subplot(gs[7, idx])
    style_ax(ax, D_titles[dk], "電流 [A]")
    y = ds(i_D[dk])
    ax.plot(ts, y, color=D_cols[idx], lw=1.0)
    ax.fill_between(ts, y, color=D_cols[idx], alpha=0.11)
    ax.set_ylim(*data_limits(y, pad=0.18))
    ax2 = ax.twinx()
    ax2.axhline(VF[dk], color=WHITE, lw=0.8, ls=":", alpha=0.65)
    ax2.set_ylabel(f"VF={VF[dk]}V", color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)
    for sp in ax2.spines.values():
        sp.set_color(BORDER)
    mark_meas(ax, f"測定点: {dk}")
    axes_all.append(ax)

ax_d5 = fig.add_subplot(gs[8, 0])
style_ax(ax_d5, "D5 (5.1V)\nツェナークランプ\n測定: スナバ回路", "電流 [A]")
y = ds(i_D["D5"])
ax_d5.plot(ts, y, color=LIME, lw=1.0)
ax_d5.fill_between(ts, y, color=LIME, alpha=0.11)
ax_d5.set_ylim(*data_limits(y, pad=0.2))
ax_d5_2 = ax_d5.twinx()
ax_d5_2.axhline(VF["D5"], color=WHITE, lw=0.8, ls=":", alpha=0.65)
ax_d5_2.set_ylabel("Vz=5.1V", color=GRAY, fontsize=6)
ax_d5_2.tick_params(colors=GRAY, labelsize=6)
for sp in ax_d5_2.spines.values():
    sp.set_color(BORDER)
axes_all.append(ax_d5)

# ROW 8: MOSFET / 制御 / NTC / スナバ
ax_q1v = fig.add_subplot(gs[8, 1])
style_ax(ax_q1v, "Q1 MOSFET の電圧\n測定: Vgs(ゲート–ソース) / Vds(ドレイン–ソース)", "電圧 [V]")
ax_q1v.plot(ts, ds(v_gate), color=PLUM, lw=1.0, label="Vgs [ゲート–ソース]")
ax_q1v.plot(ts, ds(v_drain), color=SKY, lw=1.0, label="Vds [ドレイン–ソース]", alpha=0.9)
ax_q1v.set_ylim(*data_limits(ds(v_gate), ds(v_drain), pad=0.16))
ax_q1v.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_q1v, "測定点: ⑦")
axes_all.append(ax_q1v)

ax_q1i = fig.add_subplot(gs[8, 2])
style_ax(ax_q1i, "Q1 ドレイン電流\n測定: T1一次巻線–Q1ドレイン間", "電流 [A]")
y = ds(i_primary * sw_signal)
ax_q1i.plot(ts, y, color=CORAL, lw=1.0)
ax_q1i.fill_between(ts, y, color=CORAL, alpha=0.11)
ax_q1i.set_ylim(*data_limits(y, pad=0.18))
mark_meas(ax_q1i, "測定点: ⑤")
axes_all.append(ax_q1i)

ax_duty = fig.add_subplot(gs[8, 3])
style_ax(ax_duty, "制御IC U3 のPWMデューティ\n測定: ゲート駆動", "Duty [%]")
ax_duty.plot(ts, ds(duty) * 100, color=MINT, lw=1.1)
ax_duty.axhline(30, color=GOLD, lw=0.8, ls="--", alpha=0.75, label="30%目安")
ax_duty.set_ylim(0, 90)
ax_duty.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_duty, "測定点: ⑧")
axes_all.append(ax_duty)

ax_fb = fig.add_subplot(gs[8, 0])
style_ax(ax_fb, "FB電圧\n測定: R4/R5分圧点", "電圧 [V]")
v_fb = ds(v_R["R5"])
ax_fb.plot(ts, v_fb, color=LIME, lw=1.1)
ax_fb.axhline(2.5, color=GOLD, lw=0.8, ls="--", alpha=0.75, label="FB基準 2.5V")
ax_fb.set_ylim(*data_limits(v_fb, pad=0.18))
ax_fb.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_fb, "測定点: ⑨")
axes_all.append(ax_fb)

# ROW 9: トランス / NTC / スナバ / 電力
ax_t1 = fig.add_subplot(gs[9, :2])
style_ax(ax_t1, "トランス T1\n一次電流(⑤) / 二次電圧(⑥)", "一次電流 [A]")
ax_t1.plot(ts, ds(i_primary), color=TEAL, lw=1.1, label="一次電流 [A]")
ax_t1.fill_between(ts, ds(i_primary), color=TEAL, alpha=0.11)
ax2_t1 = ax_t1.twinx()
ax2_t1.plot(ts, ds(v_sec_raw), color=GOLD, lw=1.0, ls="--", alpha=0.85, label="二次側電圧 [V]")
ax2_t1.set_ylabel("二次電圧 [V]", color=GOLD, fontsize=7)
ax2_t1.tick_params(colors=GOLD, labelsize=7)
for sp in ax2_t1.spines.values():
    sp.set_color(BORDER)
ax_t1.set_ylim(*data_limits(ds(i_primary), pad=0.16))
ax2_t1.set_ylim(*data_limits(ds(v_sec_raw), pad=0.16))
ax_t1.legend(*ax_t1.get_legend_handles_labels(), fontsize=7, facecolor="#1C2128", labelcolor=WHITE, loc="upper left")
ax2_t1.legend(*ax2_t1.get_legend_handles_labels(), fontsize=7, facecolor="#1C2128", labelcolor=WHITE, loc="upper right")
mark_meas(ax_t1, "測定点: ⑤ / ⑥")
axes_all.append(ax_t1)

ax_ntc = fig.add_subplot(gs[9, 2])
style_ax(ax_ntc, "NTCサーミスタ R3\n抵抗値と温度", "抵抗 [Ω]")
ax_ntc.plot(ts, ds(R_ntc), color=CORAL, lw=1.1, label="R_ntc")
ax_ntc.set_ylim(*data_limits(ds(R_ntc), pad=0.16))
ax_ntc2 = ax_ntc.twinx()
ax_ntc2.plot(ts, ds(T_ntc), color=GOLD, lw=0.9, ls="--", alpha=0.8, label="温度 [°C]")
ax_ntc2.set_ylabel("温度 [°C]", color=GOLD, fontsize=7)
ax_ntc2.tick_params(colors=GOLD, labelsize=7)
for sp in ax_ntc2.spines.values():
    sp.set_color(BORDER)
mark_meas(ax_ntc, "測定点: ①")
axes_all.append(ax_ntc)

ax_snb = fig.add_subplot(gs[9, 3])
style_ax(ax_snb, "スナバ回路\nR10 + C3 + D5", "電圧 [V]")
ax_snb.plot(ts, ds(v_C["C3"]), color=LIME, lw=1.0, label="C3電圧")
ax_snb.plot(ts, ds(v_R["R10"]), color=CORAL, lw=1.0, label="R10電圧", alpha=0.85)
ax_snb.axhline(VF["D5"], color=PLUM, lw=0.8, ls=":", label="D5 Vz")
ax_snb.set_ylim(*data_limits(ds(v_C["C3"]), ds(v_R["R10"]), pad=0.2))
ax_snb.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
axes_all.append(ax_snb)

# ROW 10: USB出力 / 効率 / 電力フロー / 拡大波形
ax_p1 = fig.add_subplot(gs[10, 0])
style_ax(ax_p1, "USB-A Port1 出力電圧\n測定: ⑪ C6両端 / VBUS–GND", "電圧 [V]", xlabel="時間 [ms]", show_xlabels=True)
ax_p1.plot(ts, ds(v_out1), color=MINT, lw=1.2)
ax_p1.axhline(5.0, color=GOLD, lw=0.8, ls="--", label="5.0V")
ax_p1.axhline(4.75, color=CORAL, lw=0.6, ls=":", label="4.75V下限", alpha=0.8)
ax_p1.set_ylim(*data_limits(ds(v_out1), floor=4.3, ceil=5.7, pad=0.1))
ax_p1.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_p1, "測定点: ⑪")
axes_all.append(ax_p1)

ax_p2 = fig.add_subplot(gs[10, 1])
style_ax(ax_p2, "USB-A Port2 出力電圧\n測定: ⑫ C7両端 / VBUS–GND", "電圧 [V]", xlabel="時間 [ms]", show_xlabels=True)
ax_p2.plot(ts, ds(v_out2), color=ROSE, lw=1.2)
ax_p2.axhline(5.0, color=GOLD, lw=0.8, ls="--", label="5.0V")
ax_p2.axhline(4.75, color=CORAL, lw=0.6, ls=":", label="4.75V下限", alpha=0.8)
ax_p2.set_ylim(*data_limits(ds(v_out2), floor=4.3, ceil=5.7, pad=0.1))
ax_p2.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_p2, "測定点: ⑫")
axes_all.append(ax_p2)

p_in = v_ac * (v_ac / (R["R1"] + R["R2"])) + i_primary * v_bulk
p_in_avg = np.convolve(p_in, np.ones(5000) / 5000, mode="same")
eta = np.clip((V_OUT * I_LOAD) / np.where(p_in_avg > 0.1, p_in_avg, 1.0), 0, 1) * 100

ax_eff = fig.add_subplot(gs[10, 2])
style_ax(ax_eff, "変換効率 推定\nPout / Pin", "効率 [%]", xlabel="時間 [ms]", show_xlabels=True)
ax_eff.plot(ts, ds(eta), color=LIME, lw=1.2)
ax_eff.axhline(80, color=GOLD, lw=0.8, ls="--", label="80%目安")
ax_eff.set_ylim(0, 110)
ax_eff.legend(fontsize=7, facecolor="#1C2128", labelcolor=WHITE)
axes_all.append(ax_eff)

ax_pwr = fig.add_subplot(gs[10, 3])
style_ax(ax_pwr, "定常状態の電力フロー [W]", "", xlabel="種類", show_xlabels=True)
labels = ["AC入力", "EMIロス", "整流ロス", "FET損", "トランス", "USB出力"]
values = [V_AC * 0.22, 0.5, 2.0, 1.5, 0.8, V_OUT * I_LOAD]
colors_bar = [CORAL, TEAL, GOLD, PLUM, SKY, MINT]
bars = ax_pwr.barh(labels, values, color=colors_bar, edgecolor=BORDER, height=0.6)
ax_pwr.set_xlabel("電力 [W]", color=WHITE, fontsize=6.8)
ax_pwr.tick_params(colors=WHITE, labelsize=6.5)
for sp in ax_pwr.spines.values():
    sp.set_color(BORDER)
for bar, val in zip(bars, values):
    ax_pwr.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}W", va="center", color=WHITE, fontsize=7)
axes_all.append(ax_pwr)

# ============================================================
# 仕上げ
# ============================================================
for ax in axes_all:
    ax.margins(x=0)

fig.tight_layout(rect=[0, 0, 1, 0.988])
fig.savefig(OUT_SIM, dpi=140, bbox_inches="tight", facecolor=BG)
plt.show()

print(f"回路図 saved: {OUT_CKT}")
print(f"全グラフ saved: {OUT_SIM}")