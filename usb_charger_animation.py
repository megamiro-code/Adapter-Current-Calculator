"""
スケルトンUSB充電器 回路シミュレーション（重い版・波形が伸びるアニメ付き）
============================================================================
対応内容:
- 日本語フォント自動検出
- 文字重なりを抑えたレイアウト
- グラフはフレームごとに波形そのものが伸びる
- 測定点・ノード名を明確化
- 簡易JIS風回路図を追加
- GIF保存
"""

import os
import warnings
import numpy as np
import matplotlib

# GUI表示を優先
try:
    matplotlib.use("TkAgg", force=True)
except Exception as e:
    print(f"[警告] TkAgg に切り替えできませんでした: {e}")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Arc
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.filterwarnings("ignore")

# ============================================================
# フォント設定
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
    plt.rcParams["font.family"] = JP_FONT
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 色
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
OUT_CKT = os.path.join(BASE_DIR, "usb_charger_circuit_diagram_jis.png")
OUT_PNG = os.path.join(BASE_DIR, "usb_charger_all_graphs_first_frame.png")
OUT_GIF = os.path.join(BASE_DIR, "usb_charger_all_graphs_waveform.gif")

SAVE_GIF = True

# ============================================================
# シミュレーション条件
# ============================================================
FS      = 120_000     # 軽量化
T_SIM   = 0.04
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
    "R1": 1e6, "R2": 1e6, "R3": 10.0, "R4": 47e3, "R5": 10e3, "R6": 1.0,
    "R7": 100e3, "R8": 22.0, "R9": 4.7e3, "R10": 100.0, "R11": 10.0, "R12": 10.0
}
C = {
    "C1": 0.1e-6, "C2": 330e-6, "C3": 1e-9, "C4": 100e-12,
    "C5": 47e-6, "C6": 680e-6, "C7": 680e-6, "C8": 4.7e-9
}
L = {"L1": 1.0e-3, "L2": 470e-6, "L3": 10e-6}
VF = {"D1": 0.7, "D2": 0.7, "D3": 0.45, "D4": 0.45, "D5": 5.1}

# ============================================================
# 波形計算
# ============================================================
v_ac = V_PEAK * np.sin(2 * np.pi * F_AC * t)

# EMIフィルタ後
fc_emi = 10e3
alpha = 2 * np.pi * fc_emi * dt / (1 + 2 * np.pi * fc_emi * dt)
v_emi = np.zeros_like(v_ac)
v_emi[0] = v_ac[0]
for i in range(1, len(t)):
    v_emi[i] = v_emi[i - 1] + alpha * (v_ac[i] - v_emi[i - 1])

# 整流
v_rect = np.clip(np.abs(v_emi) - 2 * VF["D1"], 0, None)

# バルクC
tau_bulk = R_LOAD * C["C2"]
v_bulk = np.zeros_like(v_rect)
v_bulk[0] = V_PEAK - 2 * VF["D1"]
for i in range(1, len(t)):
    if v_rect[i] > v_bulk[i - 1]:
        v_bulk[i] = v_rect[i]
    else:
        v_bulk[i] = v_bulk[i - 1] * np.exp(-dt / tau_bulk)

# スイッチング
sw_signal = (np.sin(2 * np.pi * F_SW * t) > 0.3).astype(float)

# 一次電流
i_primary = np.zeros_like(t)
L_mag = 800e-6
for i in range(1, len(t)):
    if sw_signal[i] > 0.5:
        i_primary[i] = i_primary[i - 1] + (v_bulk[i] / L_mag) * dt
        i_primary[i] = min(i_primary[i], 0.8)
    else:
        i_primary[i] = max(0, i_primary[i - 1] - 0.001)

# 二次側
n_ratio = V_OUT / (V_PEAK - 2 * VF["D1"])
v_sec_raw = v_bulk * n_ratio * sw_signal

# 出力
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
    "R1": np.abs(v_ac) / 2,
    "R2": np.abs(v_ac) / 2,
    "R3": i_primary * R["R3"],
    "R4": (v_out1 - 2.5) * (R["R4"] / (R["R4"] + R["R5"])),
    "R5": v_out1 * R["R5"] / (R["R4"] + R["R5"]),
    "R6": i_primary * sw_signal * R["R6"],
    "R7": np.ones_like(t) * 1.2,
    "R8": i_primary * R["R8"],
    "R9": np.ones_like(t) * 2.5,
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

# NTC / FB
T_ntc = np.clip(25.0 + (i_primary**2 * R["R3"] * 0.001).cumsum() * dt * 8e3, 25, 80)
R_ntc = R["R3"] * np.exp(3950 * (1 / (T_ntc + 273.15) - 1 / 298.15))
v_fb = v_R["R5"]

# ============================================================
# 間引き
# ============================================================
STEP = 80
ts = t[::STEP] * 1000  # ms

def ds(arr):
    return arr[::STEP]

def data_limits(*arrays, pad=0.12, symmetric=False):
    vals = np.concatenate([np.ravel(np.asarray(a)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0, 1)
    mn = float(vals.min())
    mx = float(vals.max())
    if symmetric:
        m = max(abs(mn), abs(mx))
        if m == 0:
            m = 1.0
        return (-m * (1 + pad), m * (1 + pad))
    span = mx - mn
    if span == 0:
        span = abs(mx) if mx != 0 else 1.0
    return (mn - span * pad, mx + span * pad)

# ============================================================
# 共通関数
# ============================================================
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

def mark_meas(ax, text, color=GOLD):
    ax.text(
        0.98, 0.96, text, transform=ax.transAxes,
        ha="right", va="top", fontsize=6.1, color=color,
        bbox=dict(facecolor="#1C2128", edgecolor=color, lw=0.6, boxstyle="round,pad=0.18"),
        zorder=10
    )

# ============================================================
# 回路図用の図形
# ============================================================
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

def draw_resistor(ax, x1, x2, y, amp=0.12, nzig=8, color=WHITE, lw=1.2):
    xs = np.linspace(x1, x2, nzig * 2 + 1)
    ys = np.full_like(xs, y, dtype=float)
    for i in range(1, len(xs) - 1):
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
    span = (x2 - x1) / n
    ax.plot([x1, x1 + 0.05], [y, y], color=color, lw=lw)
    for i in range(n):
        xc = x1 + (i + 0.5) * span
        arc = Arc((xc, y), span * 0.95, 0.38, theta1=0, theta2=180, color=color, lw=lw)
        ax.add_patch(arc)
    ax.plot([x2 - 0.05, x2], [y, y], color=color, lw=lw)

def draw_diode(ax, x1, x2, y, color=WHITE, lw=1.2):
    xm = (x1 + x2) / 2
    tri = Polygon([[x1, y - 0.18], [x1, y + 0.18], [xm - 0.03, y]],
                  closed=True, fill=False, edgecolor=color, lw=lw)
    ax.add_patch(tri)
    ax.plot([xm + 0.03, xm + 0.03], [y - 0.22, y + 0.22], color=color, lw=lw)
    ax.plot([x1 - 0.05, x1], [y, y], color=color, lw=lw)
    ax.plot([xm + 0.03, x2], [y, y], color=color, lw=lw)

# ============================================================
# Figure 1: 回路図（上: 図 / 下: 補足）
# ============================================================
fig_ckt = plt.figure(figsize=(22, 13), facecolor=BG)
gs_ckt = gridspec.GridSpec(2, 1, figure=fig_ckt, height_ratios=[5.2, 1.25], hspace=0.08)

ax_ckt = fig_ckt.add_subplot(gs_ckt[0])
ax_note = fig_ckt.add_subplot(gs_ckt[1])

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

# EMI
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

# Rectifier
draw_box(ax_ckt, 8.0, 4.0, 2.0, 2.6, "整流ブリッジ", "D1/D2 等価", color="#5C4A8A")
draw_diode(ax_ckt, 8.3, 9.7, 5.8, color=WHITE)
draw_diode(ax_ckt, 8.3, 9.7, 4.8, color=WHITE)
ax_ckt.text(9.0, 6.85, "D1,D2", color=WHITE, fontsize=7, ha="center")
ax_ckt.plot([10.0, 10.6], [5.6, 5.6], color=WHITE, lw=1.2)
ax_ckt.text(10.4, 5.95, "DC+", color=WHITE, fontsize=7)
ax_ckt.text(10.4, 4.55, "DC-", color=WHITE, fontsize=7)

# Bulk cap
draw_capacitor(ax_ckt, 10.6, 5.6, 1.0, color=SKY)
ax_ckt.text(10.8, 4.95, "C2 330µF", color=SKY, fontsize=7, ha="left")
ax_ckt.text(11.2, 6.15, "DCバス", color=SKY, fontsize=8, ha="center")

# Flyback
draw_box(ax_ckt, 12.0, 3.8, 3.0, 3.0, "T1 + Q1", "フライバック", color="#3D5A3E")
draw_inductor(ax_ckt, 12.3, 13.3, 5.9, n=3, color=WHITE)
draw_inductor(ax_ckt, 14.2, 15.2, 5.2, n=3, color=WHITE)
ax_ckt.plot([13.3, 14.2], [5.9, 5.9], color=WHITE, lw=1.0)
ax_ckt.plot([13.3, 14.2], [5.2, 5.2], color=WHITE, lw=1.0)
ax_ckt.text(12.6, 6.35, "一次", color=WHITE, fontsize=7)
ax_ckt.text(14.6, 5.65, "二次", color=WHITE, fontsize=7)

draw_box(ax_ckt, 12.0, 1.7, 1.9, 1.4, "U3", "PWM IC", color="#5C4A1A")
ax_ckt.text(12.95, 2.45, "AP3706相当", color="black", fontsize=6.8, ha="center")
ax_ckt.annotate("", xy=(12.7, 3.1), xytext=(12.7, 3.85),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.3))
ax_ckt.text(12.83, 3.45, "Gate", color=GOLD, fontsize=7, va="center")

draw_box(ax_ckt, 14.1, 1.7, 1.4, 1.4, "R4/R5", "FB分圧", color="#4A4A4A")
ax_ckt.annotate("", xy=(14.1, 2.35), xytext=(13.9, 2.35),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
ax_ckt.text(14.8, 2.95, "FB", color=GRAY, fontsize=7, ha="center")

draw_box(ax_ckt, 16.0, 4.7, 1.2, 0.9, "D3", "P1用", color=PLUM)
draw_box(ax_ckt, 16.0, 3.2, 1.2, 0.9, "D4", "P2用", color="#8A5C6B")
draw_capacitor(ax_ckt, 18.2, 5.6, 1.0, color=SKY)
draw_capacitor(ax_ckt, 18.2, 3.9, 1.0, color=SKY)
ax_ckt.text(18.35, 4.95, "C6 680µF", color=SKY, fontsize=7, ha="left")
ax_ckt.text(18.35, 3.25, "C7 680µF", color=SKY, fontsize=7, ha="left")
draw_box(ax_ckt, 19.5, 4.55, 2.4, 1.3, "USB-A Port1", "5V / 1.2A", color="#336633")
draw_box(ax_ckt, 19.5, 2.95, 2.4, 1.3, "USB-A Port2", "5V / 1.0A", color="#336633")
ax_ckt.text(18.9, 6.15, "二次整流", color=MINT, fontsize=8, ha="center")

draw_box(ax_ckt, 12.0, 6.95, 2.2, 0.9, "スナバ", "R10 + C3 + D5", color="#6B4E2A")
ax_ckt.text(13.1, 7.35, "スパイク吸収", color="black", fontsize=6.8, ha="center")

# 補足欄（重ならないように4分割）
ax_note.set_facecolor(BG)
ax_note.set_xlim(0, 100)
ax_note.set_ylim(0, 100)
ax_note.axis("off")

def note_box(ax, x, y, w, h, title, body, edgecolor):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4",
        facecolor="#11161C", edgecolor=edgecolor, linewidth=1.0
    )
    ax.add_patch(patch)
    ax.text(x + 2, y + h - 6, title, color=WHITE, fontsize=9, fontweight="bold", va="top")
    ax.text(x + 2, y + h - 14, body, color=WHITE, fontsize=7.4, va="top", linespacing=1.35)

note_box(
    ax_note, 1, 8, 24, 84,
    "測定点",
    "① AC入力 (v_ac)\n② EMI後電圧 (v_emi)\n③ 整流後電圧 (v_rect)\n④ DCバス電圧 (v_bulk)\n⑤ 一次電流 (i_primary)\n⑥ 二次側電圧 (v_sec_raw)\n⑦ MOSFET Vgs / Vds\n⑧ PWM Duty\n⑨ FB電圧\n⑩ D3/D4電流\n⑪ USB出力1\n⑫ USB出力2",
    GOLD
)
note_box(
    ax_note, 26, 8, 23, 84,
    "推定仕様",
    "入力: AC100V / 60Hz\n出力: USB-A ×2\nPort1: 5V / 1.2A\nPort2: 5V / 1.0A\nスイッチング周波数: 約65kHz\n出力はフライバック方式の推定",
    MINT
)
note_box(
    ax_note, 51, 8, 22, 84,
    "主要部品",
    "R1-R12\nC1-C8\nL1-L3\nD1-D5\nQ1: MOSFET\nU3: PWMコントローラ\nT1: フライバックトランス",
    SKY
)
note_box(
    ax_note, 75, 8, 24, 84,
    "補足",
    "この図は推定回路です。\n実機とは異なる可能性があります。\n記号・定数はシミュレーション用の近似です。",
    ROSE
)

fig_ckt.tight_layout()
fig_ckt.savefig(OUT_CKT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig_ckt)

# ============================================================
# Graph figure: 波形が伸びるアニメ
# ============================================================
fig = plt.figure(figsize=(20, 34), facecolor=BG)
fig.suptitle(
    "スケルトンUSB充電器 回路シミュレーション（重い版）\n"
    "AC100V/60Hz → EMI → 整流 → フライバック → USB-A×2",
    color=WHITE, fontsize=14, fontweight="bold", y=0.988
)

gs = gridspec.GridSpec(
    6, 2, figure=fig,
    hspace=0.62, wspace=0.26,
    top=0.93, bottom=0.03, left=0.05, right=0.985
)

animated = []

def add_anim_line(ax, x, y, **kwargs):
    (line,) = ax.plot([], [], **kwargs)
    animated.append((line, x, y))
    return line

# 1) 主要フロー
ax0 = fig.add_subplot(gs[0, :])
style_ax(ax0, "主要電圧フロー\n[①AC入力 → ②EMI後 → ③整流後 → ④DCバス → ⑪/⑫USB出力]", "電圧 [V]")
ax0.set_xlim(ts[0], ts[-1])
ax0.set_ylim(*data_limits(ds(v_ac), ds(v_emi), ds(v_rect), ds(v_bulk), ds(v_out1)*20, ds(v_out2)*20, pad=0.12))
add_anim_line(ax0, ts, ds(v_ac),   color=CORAL, lw=1.15, label="v_ac [① AC入力]")
add_anim_line(ax0, ts, ds(v_emi),  color=GOLD,  lw=1.05, label="v_emi [② EMI後]")
add_anim_line(ax0, ts, ds(v_rect), color=TEAL,  lw=1.05, label="v_rect [③ 整流後]")
add_anim_line(ax0, ts, ds(v_bulk), color=MINT,  lw=1.15, label="v_bulk [④ DCバス]")
add_anim_line(ax0, ts, ds(v_out1) * 20, color=SKY, lw=1.0, label="v_out1×20 [⑪]")
add_anim_line(ax0, ts, ds(v_out2) * 20, color=ROSE, lw=1.0, label="v_out2×20 [⑫]")
ax0.legend(loc="upper right", fontsize=7.0, facecolor="#1C2128", labelcolor=WHITE, ncol=3, framealpha=0.92)
mark_meas(ax0, "測定点①②③④⑪⑫")

# 2) 抵抗 R1-R6
ax_r1 = fig.add_subplot(gs[1, 0])
style_ax(ax_r1, "抵抗群 R1-R6\n測定: 各抵抗両端電圧", "電圧 [V]")
ax_r1.set_xlim(ts[0], ts[-1])
ax_r1.set_ylim(*data_limits(*[ds(v_R[k]) for k in ["R1", "R2", "R3", "R4", "R5", "R6"]], pad=0.15))
for rk, col in zip(["R1", "R2", "R3", "R4", "R5", "R6"], [CORAL, GOLD, TEAL, MINT, PLUM, SKY]):
    add_anim_line(ax_r1, ts, ds(v_R[rk]), color=col, lw=0.9, label=rk)
ax_r1.legend(fontsize=6.6, ncol=3, facecolor="#1C2128", labelcolor=WHITE)

# 3) 抵抗 R7-R12
ax_r2 = fig.add_subplot(gs[1, 1])
style_ax(ax_r2, "抵抗群 R7-R12\n測定: 各抵抗両端電圧", "電圧 [V]")
ax_r2.set_xlim(ts[0], ts[-1])
ax_r2.set_ylim(*data_limits(*[ds(v_R[k]) for k in ["R7", "R8", "R9", "R10", "R11", "R12"]], pad=0.15))
for rk, col in zip(["R7", "R8", "R9", "R10", "R11", "R12"], [ROSE, LIME, CORAL, GOLD, TEAL, MINT]):
    add_anim_line(ax_r2, ts, ds(v_R[rk]), color=col, lw=0.9, label=rk)
ax_r2.legend(fontsize=6.6, ncol=3, facecolor="#1C2128", labelcolor=WHITE)

# 4) コンデンサ C1-C4
ax_c1 = fig.add_subplot(gs[2, 0])
style_ax(ax_c1, "コンデンサ C1-C4\n測定: AC/EMI/スナバ/補助", "電圧 [V]")
ax_c1.set_xlim(ts[0], ts[-1])
ax_c1.set_ylim(*data_limits(*[ds(v_C[k]) for k in ["C1", "C2", "C3", "C4"]], pad=0.15))
for ck, col in zip(["C1", "C2", "C3", "C4"], [CORAL, TEAL, GOLD, MINT]):
    add_anim_line(ax_c1, ts, ds(v_C[ck]), color=col, lw=0.9, label=ck)
ax_c1.legend(fontsize=6.6, ncol=2, facecolor="#1C2128", labelcolor=WHITE)

# 5) コンデンサ C5-C8
ax_c2 = fig.add_subplot(gs[2, 1])
style_ax(ax_c2, "コンデンサ C5-C8\n測定: 補助/出力/Yコン", "電圧 [V]")
ax_c2.set_xlim(ts[0], ts[-1])
ax_c2.set_ylim(*data_limits(*[ds(v_C[k]) for k in ["C5", "C6", "C7", "C8"]], pad=0.15))
for ck, col in zip(["C5", "C6", "C7", "C8"], [PLUM, SKY, ROSE, LIME]):
    add_anim_line(ax_c2, ts, ds(v_C[ck]), color=col, lw=0.9, label=ck)
ax_c2.legend(fontsize=6.6, ncol=2, facecolor="#1C2128", labelcolor=WHITE)

# 6) インダクタ L1-L3
ax_l = fig.add_subplot(gs[3, 0])
style_ax(ax_l, "インダクタ L1-L3\n測定: コイル電圧", "電圧 [V]")
ax_l.set_xlim(ts[0], ts[-1])
ax_l.set_ylim(*data_limits(*[ds(v_L[k]) for k in ["L1", "L2", "L3"]], symmetric=True, pad=0.15))
for lk, col in zip(["L1", "L2", "L3"], [TEAL, CORAL, GOLD]):
    add_anim_line(ax_l, ts, ds(v_L[lk]), color=col, lw=0.9, label=f"{lk} V")
ax_l.axhline(0, color=GRAY, lw=0.45, ls="--")
ax_l.legend(fontsize=6.6, ncol=3, facecolor="#1C2128", labelcolor=WHITE)

# 7) ダイオード D1-D5
ax_d = fig.add_subplot(gs[3, 1])
style_ax(ax_d, "ダイオード D1-D5\n測定: 整流 / 二次整流 / スナバ", "電流 [A]")
ax_d.set_xlim(ts[0], ts[-1])
ax_d.set_ylim(*data_limits(*[ds(i_D[k]) for k in ["D1", "D2", "D3", "D4", "D5"]], pad=0.15))
for dk, col in zip(["D1", "D2", "D3", "D4", "D5"], [MINT, PLUM, SKY, ROSE, LIME]):
    add_anim_line(ax_d, ts, ds(i_D[dk]), color=col, lw=0.9, label=dk)
ax_d.legend(fontsize=6.6, ncol=3, facecolor="#1C2128", labelcolor=WHITE)

# 8) 制御IC / MOSFET
ax_ctrl = fig.add_subplot(gs[4, 0])
style_ax(ax_ctrl, "制御IC U3 / MOSFET Q1\n測定: Vgs / Vds / PWM Duty", "電圧 [V]")
ax_ctrl.set_xlim(ts[0], ts[-1])
ax_ctrl.set_ylim(*data_limits(ds(v_gate), ds(v_drain), pad=0.15))
add_anim_line(ax_ctrl, ts, ds(v_gate), color=PLUM, lw=1.0, label="Vgs [⑦]")
add_anim_line(ax_ctrl, ts, ds(v_drain), color=SKY, lw=1.0, label="Vds [⑦]")
ax_ctrl.legend(fontsize=6.6, facecolor="#1C2128", labelcolor=WHITE)
ax_ctrl2 = ax_ctrl.twinx()
ax_ctrl2.set_ylim(0, 100)
ax_ctrl2.set_ylabel("Duty [%]", color=GRAY, fontsize=6)
ax_ctrl2.tick_params(colors=GRAY, labelsize=6)
for sp in ax_ctrl2.spines.values():
    sp.set_color(BORDER)
add_anim_line(ax_ctrl2, ts, ds(duty) * 100, color=LIME, lw=0.85, ls="--", label="Duty [⑧]")
mark_meas(ax_ctrl, "⑦⑧")

# 9) USB出力
ax_out = fig.add_subplot(gs[4, 1])
style_ax(ax_out, "USB-A 出力 P1 / P2\n測定: C6 / C7 両端", "電圧 [V]")
ax_out.set_xlim(ts[0], ts[-1])
ax_out.set_ylim(4.3, 5.7)
add_anim_line(ax_out, ts, ds(v_out1), color=MINT, lw=1.05, label="Port1 [⑪]")
add_anim_line(ax_out, ts, ds(v_out2), color=ROSE, lw=1.05, label="Port2 [⑫]")
add_anim_line(ax_out, ts, ds(v_bulk), color=TEAL, lw=0.8, alpha=0.8, label="v_bulk")
ax_out.axhline(5.0, color=GOLD, lw=0.7, ls="--", label="5.0V")
ax_out.legend(fontsize=6.6, facecolor="#1C2128", labelcolor=WHITE)
mark_meas(ax_out, "⑪⑫")

# 10) NTC / FB
ax_ntc = fig.add_subplot(gs[5, 0])
style_ax(ax_ntc, "NTCサーミスタ R3 / FB電圧\n測定: 抵抗値 / 分圧点", "抵抗 [Ω]")
ax_ntc.set_xlim(ts[0], ts[-1])
ax_ntc.set_ylim(*data_limits(ds(R_ntc), pad=0.15))
add_anim_line(ax_ntc, ts, ds(R_ntc), color=CORAL, lw=1.0, label="R_ntc")
ax_ntc2 = ax_ntc.twinx()
ax_ntc2.set_ylim(*data_limits(ds(v_fb), pad=0.15))
ax_ntc2.set_ylabel("電圧 [V]", color=GOLD, fontsize=6)
ax_ntc2.tick_params(colors=GOLD, labelsize=6)
for sp in ax_ntc2.spines.values():
    sp.set_color(BORDER)
add_anim_line(ax_ntc2, ts, ds(v_fb), color=GOLD, lw=0.9, ls="--", label="FB [⑨]")
mark_meas(ax_ntc, "①⑨")

# 11) 効率
ax_eff = fig.add_subplot(gs[5, 1])
style_ax(ax_eff, "変換効率 推定", "効率 [%]")
p_in = v_ac * (v_ac / (R["R1"] + R["R2"])) + i_primary * v_bulk
p_in_avg = np.convolve(p_in, np.ones(2500) / 2500, mode="same")
eta = np.clip((V_OUT * I_LOAD) / np.where(p_in_avg > 0.1, p_in_avg, 1.0), 0, 1) * 100
ax_eff.set_xlim(ts[0], ts[-1])
ax_eff.set_ylim(0, 110)
add_anim_line(ax_eff, ts, ds(eta), color=LIME, lw=1.05, label="Efficiency")
ax_eff.axhline(80, color=GOLD, lw=0.7, ls="--", label="80%")
ax_eff.legend(fontsize=6.6, facecolor="#1C2128", labelcolor=WHITE)

# 右上の時間表示は、タイトルと重ならないよう少し下へ
time_text = fig.text(
    0.985, 0.952, "", color=WHITE, ha="right", va="top", fontsize=12,
    bbox=dict(facecolor="#1C2128", edgecolor=BORDER, boxstyle="round,pad=0.25")
)

# アニメーション更新
def update(frame):
    n = frame + 1
    x_now = ts[frame]
    for line, x, y in animated:
        line.set_data(x[:n], y[:n])
    time_text.set_text(f"{x_now:.2f} ms")
    return [item[0] for item in animated] + [time_text]

# フレーム数を多めにして、波形が見えるようにする
frames = np.arange(4, len(ts), max(1, len(ts) // 140)).astype(int)
ani = FuncAnimation(fig, update, frames=frames, interval=35, blit=False, repeat=True)

# ★ 最後のフレームを描画（これを追加！）
last_frame = len(ts) - 1
update(last_frame)

# 初期静止画像保存
fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor=BG)

# GIF保存
if SAVE_GIF:
    try:
        ani.save(OUT_GIF, writer=PillowWriter(fps=12), dpi=80)
        print(f"GIF saved: {OUT_GIF}")
    except Exception as e:
        print(f"GIF保存に失敗: {e}")

print(f"回路図 saved: {OUT_CKT}")
print(f"静止画 saved: {OUT_PNG}")

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()