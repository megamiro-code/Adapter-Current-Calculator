"""
USB Charger Circuit Simulation - All Graphs Version v2
AC100V/60Hz -> Flyback Converter -> USB-A 5V x2
"""

import os, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings('ignore')

# ── フォント設定（文字化け対策）──
# Windows / Mac / Linux 共通で日本語フォントを自動検出
import matplotlib.font_manager as fm

def find_jp_font():
    candidates = [
        # Windows
        'Yu Gothic', 'Meiryo', 'MS Gothic', 'MS PGothic',
        # Mac
        'Hiragino Sans', 'Hiragino Kaku Gothic Pro',
        # Linux / fallback
        'Noto Sans CJK JP', 'IPAGothic', 'IPAPGothic',
        'TakaoPGothic', 'VL Gothic',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None   # 見つからない場合は英語表示

JP_FONT = find_jp_font()

def jlabel(jp, en):
    """日本語フォントが使えれば日本語、なければ英語を返す"""
    return jp if JP_FONT else en

if JP_FONT:
    matplotlib.rcParams['font.family'] = JP_FONT
matplotlib.rcParams['axes.unicode_minus'] = False

# ─── カラーパレット ───
BG      = '#0D1117'
PANEL   = '#161B22'
BORDER  = '#30363D'
CORAL   = '#FF6F61'
GOLD    = '#FFC857'
TEAL    = '#00838F'
MINT    = '#A8E6CF'
PLUM    = '#7B5EA7'
SKY     = '#4ECDC4'
ROSE    = '#FF6B9D'
LIME    = '#C5E99B'
WHITE   = '#E6EDF3'
GRAY    = '#8B949E'

# ─── シミュレーションパラメータ ───
# 速度優先: サンプリング周波数を下げ、間引き後データ点数を統一
FS      = 200_000      # 200kHz (65kHz SWの3倍以上確保しつつ軽量化)
T_SIM   = 0.05         # 50ms = AC 3サイクル
t       = np.linspace(0, T_SIM, int(FS * T_SIM), endpoint=False)
dt      = t[1] - t[0]

F_AC    = 60.0
V_AC    = 100.0
V_PEAK  = V_AC * np.sqrt(2)   # ~141.4 V

F_SW    = 65_000.0
V_OUT   = 5.0
I_LOAD1 = 1.2    # Port1 負荷電流 [A]
I_LOAD2 = 1.0    # Port2 負荷電流 [A]
I_LOAD  = I_LOAD1 + I_LOAD2
R_LOAD  = V_OUT / I_LOAD      # 等価負荷抵抗

# ─── 部品定数 ───
R_val = dict(
    R1=1e6, R2=1e6, R3=10.0, R4=47e3, R5=10e3,
    R6=1.0, R7=100e3, R8=22.0, R9=4.7e3, R10=100.0,
    R11=10.0, R12=10.0
)
C_val = dict(
    C1=0.1e-6, C2=330e-6, C3=1e-9, C4=100e-12,
    C5=47e-6,  C6=680e-6, C7=680e-6, C8=4.7e-9
)
L_val = dict(L1=1.0e-3, L2=470e-6, L3=10e-6)
VF    = dict(D1=0.7, D2=0.7, D3=0.45, D4=0.45, D5=5.1)

# ─── 波形計算 ───
# ① AC入力  [測定点: ACコンセント端子間 = N点とL点]
v_ac = V_PEAK * np.sin(2 * np.pi * F_AC * t)

# ② EMIフィルタ後  [測定点: L1/L2後の整流ブリッジ入力ノード]
fc_emi = 8e3
alpha  = 2*np.pi*fc_emi*dt / (1 + 2*np.pi*fc_emi*dt)
v_emi  = np.zeros_like(v_ac); v_emi[0] = v_ac[0]
for i in range(1, len(t)):
    v_emi[i] = v_emi[i-1] + alpha*(v_ac[i] - v_emi[i-1])

# ③ 全波整流後  [測定点: ブリッジ出力 = C2正極端子]
v_rect = np.clip(np.abs(v_emi) - 2*VF['D1'], 0, None)

# ④ バルクC平滑後  [測定点: C2両端 = MOSFETドレイン側バス電圧]
tau_bulk = R_LOAD * C_val['C2']
v_bulk = np.zeros_like(v_rect); v_bulk[0] = V_PEAK - 2*VF['D1']
for i in range(1, len(t)):
    v_bulk[i] = v_rect[i] if v_rect[i] > v_bulk[i-1] \
                else v_bulk[i-1] * np.exp(-dt / tau_bulk)

# ⑤ MOSFET スイッチング信号
sw = (np.sin(2*np.pi*F_SW*t) > 0.3).astype(float)

# ⑥ トランス一次側電流  [測定点: T1一次巻線とQ1ドレイン間]
i_pri = np.zeros_like(t); L_mag = 800e-6
for i in range(1, len(t)):
    if sw[i] > 0.5:
        i_pri[i] = min(i_pri[i-1] + (v_bulk[i]/L_mag)*dt, 0.8)
    else:
        i_pri[i] = max(0, i_pri[i-1] - 5e-4)

# ⑦ トランス二次側電圧  [測定点: T1二次巻線端子 = D3/D4アノード]
n_ratio = V_OUT / (V_PEAK - 2*VF['D1'])
v_sec   = v_bulk * n_ratio * sw

# ⑧ USB出力電圧  [測定点: C6/C7両端 = USB-Aコネクタ VBUS-GND間]
v_out1 = np.zeros_like(t); v_out1[0] = V_OUT
v_out2 = np.zeros_like(t); v_out2[0] = V_OUT
for i in range(1, len(t)):
    vi1 = max(0, v_sec[i] - VF['D3'])
    v_out1[i] = np.clip(
        v_out1[i-1] + (vi1 - v_out1[i-1])*0.1 if vi1 > v_out1[i-1]
        else v_out1[i-1] - (I_LOAD1/C_val['C6'])*dt, 4.5, 5.5)
    vi2 = max(0, v_sec[i] - VF['D4'])
    v_out2[i] = np.clip(
        v_out2[i-1] + (vi2 - v_out2[i-1])*0.1 if vi2 > v_out2[i-1]
        else v_out2[i-1] - (I_LOAD2/C_val['C7'])*dt, 4.5, 5.5)

# ⑨ 各部品の電圧・電流
# -- 抵抗 --
vR = {
    'R1': np.abs(v_ac)/2,                              # AC入力から GND 間の分圧
    'R2': np.abs(v_ac)/2,                              # 同上（対称）
    'R3': i_pri * R_val['R3'],                         # NTC 両端 = 突入制限時の降下
    'R4': v_out1 * R_val['R4']/(R_val['R4']+R_val['R5']),  # FB分圧上
    'R5': v_out1 * R_val['R5']/(R_val['R4']+R_val['R5']),  # FB分圧下
    'R6': i_pri * sw * R_val['R6'],                    # ゲート抵抗降下
    'R7': np.full_like(t, 1.2),                        # ソフトスタートC充電中の降下
    'R8': i_pri * R_val['R8'],                         # 電流検出 (Vcs)
    'R9': np.full_like(t, 2.5),                        # バイアス
    'R10': i_pri * sw * R_val['R10'] * 0.05,           # スナバ
    'R11': np.full_like(t, I_LOAD1 * R_val['R11']),    # 出力ダンピング Port1
    'R12': np.full_like(t, I_LOAD2 * R_val['R12']),    # 出力ダンピング Port2
}
iR = {k: vR[k] / R_val[k] for k in R_val}

# -- コンデンサ --
vC = {
    'C1': v_ac * 0.05,          # CXコン: EMIノイズ電圧
    'C2': v_bulk,               # バルクC: DCバス電圧
    'C3': i_pri * sw * 30,      # スナバC: スパイク吸収
    'C4': np.full_like(t, 12.), # ブートストラップ: Vcc
    'C5': np.full_like(t, 12.), # 補助電源
    'C6': v_out1,               # 出力C Port1: USB VBUS
    'C7': v_out2,               # 出力C Port2: USB VBUS
    'C8': v_ac * 0.008,         # Yコン: コモンモードノイズ
}
iC = {k: C_val[k] * np.gradient(vC[k], dt) for k in C_val}

# -- インダクタ --
# L1/L2: コモン/差動モードチョーク（EMIフィルタ）、L3: 出力インダクタ
dvemi = np.gradient(v_emi, dt)
vL = {
    'L1': np.clip(dvemi * L_val['L1'] * 0.0008, -20, 20),
    'L2': np.clip(dvemi * L_val['L2'] * 0.0004, -10, 10),
    'L3': np.gradient(v_out1, dt) * L_val['L3'],
}
iL = {
    'L1': np.cumsum(vL['L1'])*dt / L_val['L1'],
    'L2': np.cumsum(vL['L2'])*dt / L_val['L2'],
    'L3': I_LOAD * np.ones_like(t) + np.cumsum(vL['L3'])*dt / L_val['L3'] * 0.001,
}

# -- ダイオード --
iD = {
    'D1': np.clip(v_rect / R_LOAD, 0, 3.),                                    # ブリッジ整流電流
    'D2': np.clip(v_rect / R_LOAD, 0, 3.),                                    # 同上
    'D3': np.clip((v_sec - VF['D3']) / (R_LOAD/2), 0, 3.) * sw,              # 二次整流 Port1
    'D4': np.clip((v_sec - VF['D4']) / (R_LOAD/2), 0, 3.) * sw,              # 二次整流 Port2
    'D5': np.where(v_bulk*1.3 > v_bulk + VF['D5'], 0.1, 0.),                 # ツェナークランプ
}

# -- MOSFET & 制御IC --
v_gate  = sw * 10.
v_drain = v_bulk * (1-sw) + sw * 0.5

err_ic  = V_OUT - (v_out1+v_out2)/2
duty    = np.clip(0.3 + np.cumsum(err_ic)*dt*8, 0.05, 0.8)

# -- NTCサーミスタ --
T_ntc = np.clip(25. + (i_pri**2 * R_val['R3'] * 0.001).cumsum()*dt*8e3, 25, 80)
R_ntc = R_val['R3'] * np.exp(3950*(1/(T_ntc+273.15) - 1/298.15))

# ─── 間引き（プロット用） ───
STEP = 20   # FS=200k → 10k点 → 十分滑らか
ts = t[::STEP] * 1000   # [ms]

def ds(a): return a[::STEP]

# ════════════════════════════════════════════════
#  FIGURE 1: 回路図
# ════════════════════════════════════════════════
fig_ckt, ax_ckt = plt.subplots(1, 1, figsize=(20, 10), facecolor=BG)
ax_ckt.set_facecolor(BG)
ax_ckt.set_xlim(0, 20); ax_ckt.set_ylim(0, 10)
ax_ckt.axis('off')
ax_ckt.set_title(
    jlabel('USB充電器 推定回路図  (AC100V/60Hz → フライバックコンバータ → USB-A 5V×2)',
           'USB Charger Circuit Diagram  (AC100V/60Hz -> Flyback -> USB-A 5Vx2)'),
    color=WHITE, fontsize=13, fontweight='bold', pad=12
)

def draw_box(ax, x, y, w, h, label, sublabel='', color=TEAL, alpha=0.85):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                          facecolor=color, edgecolor=WHITE, linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2 + (0.18 if sublabel else 0), label,
            ha='center', va='center', color='black', fontsize=9, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x+w/2, y+h/2 - 0.28, sublabel,
                ha='center', va='center', color='black', fontsize=7, zorder=4)

def arrow(ax, x1, y1, x2, y2, col=WHITE, lw=1.5):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw), zorder=5)

def line_h(ax, x1, x2, y, col=WHITE, lw=1.2, ls='-'):
    ax.plot([x1,x2], [y,y], color=col, lw=lw, ls=ls, zorder=2)

def line_v(ax, x, y1, y2, col=WHITE, lw=1.2, ls='-'):
    ax.plot([x,x], [y1,y2], color=col, lw=lw, ls=ls, zorder=2)

def meas_dot(ax, x, y, label, col=GOLD, offset=(0.1,0.18)):
    ax.plot(x, y, 'o', color=col, ms=7, zorder=6)
    ax.text(x+offset[0], y+offset[1], label,
            color=col, fontsize=7.5, fontweight='bold', zorder=7)

# ── GNDライン (y=1.0) ──
line_h(ax_ckt, 0.3, 19.7, 1.0, col=GRAY, lw=1.0, ls='--')
ax_ckt.text(0.1, 1.0, 'GND', color=GRAY, fontsize=8, va='center')

# ── AC電源 ──
draw_box(ax_ckt, 0.3, 3.5, 1.5, 2.5, jlabel('AC電源', 'AC Source'),
         'AC100V\n60Hz', color='#444')
line_v(ax_ckt, 1.05, 6.0, 7.5); line_v(ax_ckt, 1.05, 1.0, 3.5)   # L/N線

# ── NTCサーミスタ (R3) ──
draw_box(ax_ckt, 2.2, 6.8, 1.0, 0.8, 'R3\nNTC', '10Ω@25°C', color=CORAL, alpha=0.9)
line_h(ax_ckt, 1.05, 2.2, 7.2); line_h(ax_ckt, 3.2, 3.7, 7.2)
meas_dot(ax_ckt, 2.7, 7.2, jlabel('①NTC降下', '①NTC drop'), col=GOLD, offset=(0,-0.35))

# ── EMIフィルタ ──
draw_box(ax_ckt, 3.7, 5.8, 2.2, 2.8,
         jlabel('EMIフィルタ', 'EMI Filter'),
         'L1(1mH)+L2(470μH)\nC1(0.1μF)+C8(4.7nF)', color='#2E5E6E', alpha=0.9)
line_h(ax_ckt, 3.2, 3.7, 7.2); line_h(ax_ckt, 5.9, 6.5, 7.2)
line_h(ax_ckt, 1.05, 3.7, 1.0)
meas_dot(ax_ckt, 5.9, 7.2, jlabel('②EMI後', '②post-EMI'), col=MINT, offset=(0.05,0.18))

# ── 整流ブリッジ (D1-D4等価) ──
draw_box(ax_ckt, 6.5, 5.5, 1.8, 2.2,
         jlabel('整流ブリッジ', 'Bridge'),
         'D1/D2\nVF=0.7V×2', color='#5C4A8A', alpha=0.9)
line_h(ax_ckt, 5.9, 6.5, 7.2); line_h(ax_ckt, 8.3, 8.8, 7.2)
line_v(ax_ckt, 8.3, 1.0, 5.5)
meas_dot(ax_ckt, 8.3, 7.2, jlabel('③整流後', '③rectified'), col=CORAL, offset=(0.05,0.18))

# ── バルクC (C2) ──
draw_box(ax_ckt, 8.8, 5.5, 1.3, 2.5,
         'C2', '330μF\n200V', color='#3A6EA5', alpha=0.9)
line_h(ax_ckt, 8.3, 8.8, 7.2); line_h(ax_ckt, 10.1, 10.6, 7.2)
line_v(ax_ckt, 10.1, 1.0, 5.5)
meas_dot(ax_ckt, 10.1, 7.2, jlabel('④DCバス', '④DC bus'), col=GOLD, offset=(0.05,0.18))

# ── トランス T1 ──
draw_box(ax_ckt, 10.6, 4.8, 2.0, 3.2,
         'T1', 'QL015Z\nn≈1:15', color='#3D5A3E', alpha=0.9)
# 一次側
line_h(ax_ckt, 10.1, 10.6, 7.2)
# 二次側
line_h(ax_ckt, 12.6, 13.2, 7.5)
line_h(ax_ckt, 12.6, 13.2, 5.2)
meas_dot(ax_ckt, 10.85, 7.2, jlabel('⑤一次', '⑤primary'), col=SKY, offset=(0,-0.35))
meas_dot(ax_ckt, 12.8, 7.5, jlabel('⑥二次', '⑥secondary'), col=MINT, offset=(0.05,0.15))

# ── MOSFET Q1 ──
draw_box(ax_ckt, 10.6, 2.5, 1.5, 1.8, 'Q1\nMOSFET', 'N-ch', color='#6B3E3E', alpha=0.9)
line_v(ax_ckt, 11.35, 4.8, 4.3)
meas_dot(ax_ckt, 11.35, 4.0, jlabel('⑦Vds', '⑦Vds'), col=ROSE, offset=(0.12,0))
line_v(ax_ckt, 11.35, 2.5, 1.0)

# ── 制御IC U3 ──
draw_box(ax_ckt, 9.0, 2.5, 1.3, 1.8, 'U3\nPWM IC', 'AP3706\n相当', color='#5C4A1A', alpha=0.9)
arrow(ax_ckt, 10.3, 3.4, 10.6, 3.4, col=GOLD)
ax_ckt.text(10.42, 3.55, jlabel('Gate', 'Gate'), color=GOLD, fontsize=7)
meas_dot(ax_ckt, 10.3, 3.4, jlabel('⑧Duty', '⑧Duty'), col=GOLD, offset=(-0.9,0.12))

# R4/R5 帰還
draw_box(ax_ckt, 8.0, 2.5, 0.8, 1.8, 'R4/R5', '47k/10k', color='#4A4A4A', alpha=0.9)
arrow(ax_ckt, 8.0, 3.4, 9.0, 3.4, col=GRAY)
meas_dot(ax_ckt, 8.4, 3.4, jlabel('⑨FB', '⑨FB'), col=GRAY, offset=(0,0.18))

# ── D3 ショットキー Port1 ──
draw_box(ax_ckt, 13.2, 7.0, 1.2, 0.9, 'D3', jlabel('Port1用', 'Port1'), color=PLUM, alpha=0.9)
line_h(ax_ckt, 12.6, 13.2, 7.5)
arrow(ax_ckt, 14.4, 7.45, 15.0, 7.45, col=SKY)
meas_dot(ax_ckt, 13.8, 7.45, jlabel('⑩D3電流', '⑩D3 I'), col=SKY, offset=(0,0.18))

# ── D4 ショットキー Port2 ──
draw_box(ax_ckt, 13.2, 4.8, 1.2, 0.9, 'D4', jlabel('Port2用', 'Port2'), color='#8A5C6B', alpha=0.9)
line_h(ax_ckt, 12.6, 13.2, 5.2)
arrow(ax_ckt, 14.4, 5.25, 15.0, 5.25, col=ROSE)

# ── C6 / C7 出力C ──
draw_box(ax_ckt, 15.0, 6.8, 1.2, 1.4, 'C6', '680μF\n10V', color='#3A6EA5', alpha=0.9)
draw_box(ax_ckt, 15.0, 4.6, 1.2, 1.4, 'C7', '680μF\n10V', color='#3A6EA5', alpha=0.9)
line_v(ax_ckt, 15.0, 1.0, 4.6); line_v(ax_ckt, 15.0, 5.8, 6.8)
meas_dot(ax_ckt, 16.2, 7.5, jlabel('⑪USB-P1\nVBUS', '⑪P1 VBUS'), col=MINT, offset=(0.05,0.1))
meas_dot(ax_ckt, 16.2, 5.25, jlabel('⑫USB-P2\nVBUS', '⑫P2 VBUS'), col=ROSE, offset=(0.05,0.1))

# ── USB-A コネクタ ──
draw_box(ax_ckt, 16.8, 6.8, 2.0, 1.4,
         jlabel('USB-A Port1', 'USB-A Port1'), '5V / 1.2A', color='#336633', alpha=0.9)
draw_box(ax_ckt, 16.8, 4.6, 2.0, 1.4,
         jlabel('USB-A Port2', 'USB-A Port2'), '5V / 1.0A', color='#336633', alpha=0.9)
line_h(ax_ckt, 16.2, 16.8, 7.5); line_h(ax_ckt, 16.2, 16.8, 5.25)
line_v(ax_ckt, 18.8, 1.0, 4.6); line_v(ax_ckt, 18.8, 5.8, 6.8)  # GND

# ── スナバ回路 ──
draw_box(ax_ckt, 10.6, 8.2, 1.8, 1.0,
         jlabel('スナバ', 'Snubber'), 'R10+C3+D5', color='#6B4E2A', alpha=0.9)
line_v(ax_ckt, 11.5, 8.2, 8.1); line_v(ax_ckt, 11.5, 8.1, 7.5)
line_h(ax_ckt, 10.6, 10.1, 8.7)

# ── 電流方向矢印 ──
arrow(ax_ckt, 1.8, 7.45, 2.2, 7.45, col=CORAL, lw=2)   # AC→NTC
arrow(ax_ckt, 5.9, 7.45, 6.5, 7.45, col=MINT, lw=2)    # EMI→Bridge

# ── 測定点凡例 ──
legend_items = [
    mpatches.Patch(color=GOLD,  label=jlabel('① NTC両端電圧降下', '① NTC voltage drop')),
    mpatches.Patch(color=MINT,  label=jlabel('② EMIフィルタ後電圧', '② post-EMI voltage')),
    mpatches.Patch(color=CORAL, label=jlabel('③ ブリッジ整流後電圧', '③ bridge output')),
    mpatches.Patch(color=GOLD,  label=jlabel('④ DCバス電圧 (C2両端)', '④ DC bus (C2)')),
    mpatches.Patch(color=SKY,   label=jlabel('⑤ T1一次側電流', '⑤ T1 primary current')),
    mpatches.Patch(color=MINT,  label=jlabel('⑥ T1二次側電圧', '⑥ T1 secondary')),
    mpatches.Patch(color=ROSE,  label=jlabel('⑦ Q1 Vds', '⑦ Q1 Vds')),
    mpatches.Patch(color=GOLD,  label=jlabel('⑧ PWMデューティ', '⑧ PWM duty')),
    mpatches.Patch(color=GRAY,  label=jlabel('⑨ 帰還電圧 FB', '⑨ feedback FB')),
    mpatches.Patch(color=SKY,   label=jlabel('⑩ D3/D4電流', '⑩ D3/D4 current')),
    mpatches.Patch(color=MINT,  label=jlabel('⑪ USB Port1 VBUS', '⑪ USB Port1 VBUS')),
    mpatches.Patch(color=ROSE,  label=jlabel('⑫ USB Port2 VBUS', '⑫ USB Port2 VBUS')),
]
ax_ckt.legend(handles=legend_items, loc='lower left', fontsize=7.5,
              facecolor='#1C2128', edgecolor=BORDER, labelcolor=WHITE,
              ncol=2, framealpha=0.9)

fig_ckt.tight_layout()

# ════════════════════════════════════════════════
#  FIGURE 2: 全グラフ（静止画）
# ════════════════════════════════════════════════
fig = plt.figure(figsize=(26, 44), facecolor=BG)
fig.suptitle(
    jlabel(
        'USB充電器 回路シミュレーション  全部品 電圧/電流グラフ\n'
        'AC100V/60Hz → EMIフィルタ → 全波整流 → フライバックコンバータ → USB-A 5V×2',
        'USB Charger Simulation - All Component Voltage/Current Graphs\n'
        'AC100V/60Hz -> EMI Filter -> Bridge Rectifier -> Flyback -> USB-A 5Vx2'
    ),
    color=WHITE, fontsize=14, fontweight='bold', y=0.995
)

gs = gridspec.GridSpec(11, 4, figure=fig,
                       hspace=0.60, wspace=0.38,
                       top=0.988, bottom=0.015, left=0.06, right=0.98)

# ─── ユーティリティ ───
def make_ax(pos, title, ylabel, xlabel=jlabel('時間 [ms]','Time [ms]')):
    ax = fig.add_subplot(pos)
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=8, pad=4)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=7)
    ax.set_xlabel(xlabel, color=WHITE, fontsize=7)
    ax.tick_params(colors=WHITE, labelsize=7)
    ax.xaxis.label.set_color(WHITE)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.grid(axis='y', color=BORDER, lw=0.4, alpha=0.5)
    return ax

def annotate_meas(ax, text, col=GOLD):
    """測定点ラベルを右上に表示"""
    ax.text(0.98, 0.97, text, transform=ax.transAxes,
            ha='right', va='top', fontsize=6.5, color=col,
            bbox=dict(facecolor='#1C2128', edgecolor=col, lw=0.6,
                      boxstyle='round,pad=0.2'), zorder=10)

COLORS = [CORAL, GOLD, TEAL, MINT, PLUM, SKY, ROSE, LIME,
          CORAL, GOLD, TEAL, MINT]

# ══ ROW 0: 主要電圧フロー（全段） ══
ax0 = make_ax(gs[0, :],
    jlabel('主要電圧フロー (AC入力 → EMI → 整流 → DCバス → USB出力)',
           'Main Voltage Flow (AC -> EMI -> Rectifier -> DC bus -> USB)'),
    jlabel('電圧 [V]', 'Voltage [V]'), '')
ax0.plot(ts, ds(v_ac),   color=CORAL, lw=1.4, label=jlabel('v_ac  : AC入力電圧 (コンセント端子間)', 'v_ac: AC input'))
ax0.plot(ts, ds(v_emi),  color=GOLD,  lw=1.2, label=jlabel('v_emi : EMIフィルタ後', 'v_emi: post-EMI'), alpha=0.8)
ax0.plot(ts, ds(v_rect), color=TEAL,  lw=1.2, label=jlabel('v_rect: ブリッジ整流後', 'v_rect: after bridge'), alpha=0.8)
ax0.plot(ts, ds(v_bulk), color=MINT,  lw=1.5, label=jlabel('v_bulk: DCバス (C2両端)', 'v_bulk: DC bus (C2)'))
ax0.plot(ts, ds(v_out1)*20, color=SKY, lw=1.2,
         label=jlabel('v_out1×20: USB-A Port1 出力', 'v_out1×20: USB Port1 output'))
ax0.axhline(0, color=GRAY, lw=0.5, ls='--')
ax0.legend(loc='upper right', fontsize=7.5, facecolor='#1C2128',
           labelcolor=WHITE, ncol=2, framealpha=0.9)
annotate_meas(ax0, jlabel('測定点①②③④⑪', 'Points①②③④⑪'))

# ══ ROW 1-2: 抵抗 R1-R12 電圧 ══
R_info = {
    'R1':  jlabel('R1 (1MΩ)\nAC入力ブリーダー上側\n[測定点: L-GND間]', 'R1(1MΩ) AC bleeder top'),
    'R2':  jlabel('R2 (1MΩ)\nAC入力ブリーダー下側\n[測定点: N-GND間]', 'R2(1MΩ) AC bleeder bot'),
    'R3':  jlabel('R3/NTC (10Ω)\n突入電流制限\n[測定点: ①NTC両端]', 'R3/NTC(10Ω) inrush limit'),
    'R4':  jlabel('R4 (47kΩ)\n出力帰還分圧 上\n[測定点: ⑨FB-Vout間]', 'R4(47kΩ) FB upper'),
    'R5':  jlabel('R5 (10kΩ)\n出力帰還分圧 下\n[測定点: ⑨FB-GND間]', 'R5(10kΩ) FB lower'),
    'R6':  jlabel('R6 (1Ω)\nMOSFETゲート抵抗\n[測定点: Gate端子-IC間]', 'R6(1Ω) gate resistor'),
    'R7':  jlabel('R7 (100kΩ)\nソフトスタート抵抗\n[測定点: Css充電ノード]', 'R7(100kΩ) soft-start'),
    'R8':  jlabel('R8 (22Ω)\n電流検出抵抗 Vcs\n[測定点: ⑤Q1ソース-GND間]', 'R8(22Ω) current sense'),
    'R9':  jlabel('R9 (4.7kΩ)\nバイアス抵抗\n[測定点: VCC-FB間]', 'R9(4.7kΩ) bias'),
    'R10': jlabel('R10 (100Ω)\nスナバ抵抗\n[測定点: スナバ回路両端]', 'R10(100Ω) snubber'),
    'R11': jlabel('R11 (10Ω)\n出力ダンピング Port1\n[測定点: ⑪C6-USB端子間]', 'R11(10Ω) damping P1'),
    'R12': jlabel('R12 (10Ω)\n出力ダンピング Port2\n[測定点: ⑫C7-USB端子間]', 'R12(10Ω) damping P2'),
}
for idx, (rk, info) in enumerate(R_info.items()):
    row = 1 + idx // 4
    col = idx % 4
    ax = make_ax(gs[row, col], info, jlabel('電圧 [V]', 'V'))
    ax.plot(ts, ds(vR[rk]), color=COLORS[idx], lw=1.0)
    ax.fill_between(ts, ds(vR[rk]), alpha=0.12, color=COLORS[idx])
    # 電流を右軸に追加
    ax2 = ax.twinx()
    ax2.plot(ts, ds(iR[rk])*1000, color=COLORS[idx], lw=0.7, ls=':', alpha=0.6)
    ax2.set_ylabel(jlabel('電流[mA]', 'I[mA]'), color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)

# ══ ROW 3: コンデンサ C1-C8 電圧 ══
C_info = {
    'C1': jlabel('C1 (0.1μF/275V)\nCX EMIフィルタC\n[測定点: ACライン間のノイズ]', 'C1(0.1μF) CX EMI'),
    'C2': jlabel('C2 (330μF/200V)\nバルク平滑C\n[測定点: ④DCバス-GND間]', 'C2(330μF) bulk cap'),
    'C3': jlabel('C3 (1nF)\nスナバC\n[測定点: スナバノード-GND間]', 'C3(1nF) snubber cap'),
    'C4': jlabel('C4 (100pF)\nブートストラップC\n[測定点: VCC供給ノード]', 'C4(100pF) bootstrap'),
    'C5': jlabel('C5 (47μF)\n補助電源平滑C\n[測定点: 制御IC VCC端子]', 'C5(47μF) aux power'),
    'C6': jlabel('C6 (680μF/10V)\n出力平滑C Port1\n[測定点: ⑪USB Port1 VBUS-GND間]', 'C6(680μF) output P1'),
    'C7': jlabel('C7 (680μF/10V)\n出力平滑C Port2\n[測定点: ⑫USB Port2 VBUS-GND間]', 'C7(680μF) output P2'),
    'C8': jlabel('C8 (4.7nF)\nYコンデンサ (青)\n[測定点: ACライン-PE間のノイズ]', 'C8(4.7nF) Y-cap'),
}
for idx, (ck, info) in enumerate(C_info.items()):
    ax = make_ax(gs[3, idx % 4], info, jlabel('電圧 [V]', 'V'))
    ax.plot(ts, ds(vC[ck]), color=COLORS[idx], lw=1.0)
    ax.fill_between(ts, ds(vC[ck]), alpha=0.12, color=COLORS[idx])
    # 充放電電流を右軸
    ax2 = ax.twinx()
    ic_clip = np.clip(ds(iC[ck])*1000, -300, 300)
    ax2.plot(ts, ic_clip, color=COLORS[idx], lw=0.7, ls=':', alpha=0.6)
    ax2.set_ylabel(jlabel('電流[mA]', 'I[mA]'), color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)

# ══ ROW 4: コンデンサ充放電電流まとめ ══
ax_ci = make_ax(gs[4, :2],
    jlabel('全コンデンサ 充放電電流 (C·dV/dt)\n[実線:充電方向, 点線:放電方向]',
           'All Cap Charge/Discharge Current'),
    jlabel('電流 [mA]', 'Current [mA]'))
for ck, col in zip(C_info.keys(), COLORS):
    ic = np.clip(ds(iC[ck])*1000, -400, 400)
    ax_ci.plot(ts, ic, lw=0.9, label=ck, color=col, alpha=0.85)
ax_ci.axhline(0, color=GRAY, lw=0.5, ls='--')
ax_ci.legend(fontsize=7, ncol=2, facecolor='#1C2128', labelcolor=WHITE)

ax_cv = make_ax(gs[4, 2:],
    jlabel('全コンデンサ 電圧\n[C2(DCバス)が最大, C6/C7が出力5V]',
           'All Capacitor Voltage'),
    jlabel('電圧 [V]', 'V'))
for ck, col in zip(C_info.keys(), COLORS):
    ax_cv.plot(ts, ds(vC[ck]), lw=0.9, label=ck, color=col)
ax_cv.legend(fontsize=7, ncol=2, facecolor='#1C2128', labelcolor=WHITE)

# ══ ROW 5: インダクタ L1-L3 電圧・電流 ══
L_info = {
    'L1': jlabel('L1 (1mH)\nコモンモードチョーク\n[測定点: ACライン貫通ノード間]', 'L1(1mH) CMC'),
    'L2': jlabel('L2 (470μH)\n差動モードチョーク\n[測定点: L2両端電圧]', 'L2(470μH) DM choke'),
    'L3': jlabel('L3 (10μH)\n出力インダクタ\n[測定点: D3/D4後-C6前]', 'L3(10μH) output'),
}
L_cols = [TEAL, CORAL, GOLD]
for idx, (lk, info) in enumerate(L_info.items()):
    # 電圧
    ax = make_ax(gs[5, idx*2 if idx<2 else 2],
                 info, jlabel('電圧 [V]', 'V'))
    vl = np.clip(ds(vL[lk]), -25, 25)
    ax.plot(ts, vl, color=L_cols[idx], lw=1.0)
    ax.fill_between(ts, vl, alpha=0.12, color=L_cols[idx])
    ax.axhline(0, color=GRAY, lw=0.4, ls='--')
    # 電流 右軸
    ax2 = ax.twinx()
    il = np.clip(ds(iL[lk]), -3, 5)
    ax2.plot(ts, il, color=L_cols[idx], lw=0.8, ls='--', alpha=0.75,
             label=jlabel('電流[A]', 'I[A]'))
    ax2.set_ylabel(jlabel('電流 [A]', 'I [A]'), color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)

# インダクタまとめ
ax_lv = make_ax(gs[5, 3],
    jlabel('全インダクタ 電圧 (比較)', 'All Inductor Voltage'),
    jlabel('電圧 [V]', 'V'))
for lk, col in zip(L_info.keys(), L_cols):
    ax_lv.plot(ts, np.clip(ds(vL[lk]), -25, 25), lw=0.9, label=lk, color=col)
ax_lv.axhline(0, color=GRAY, lw=0.4, ls='--')
ax_lv.legend(fontsize=8, facecolor='#1C2128', labelcolor=WHITE)

# ══ ROW 6: ダイオード D1-D5 ══
D_info = {
    'D1': jlabel('D1 (VF=0.7V)\n整流ブリッジ 上アーム\n[測定点: ③ブリッジ出力電流]', 'D1 bridge upper'),
    'D2': jlabel('D2 (VF=0.7V)\n整流ブリッジ 下アーム\n[測定点: ③ブリッジ出力電流]', 'D2 bridge lower'),
    'D3': jlabel('D3 (VF=0.45V)\n二次整流 ショットキー Port1\n[測定点: ⑩D3アノード電流]', 'D3 schottky P1'),
    'D4': jlabel('D4 (VF=0.45V)\n二次整流 ショットキー Port2\n[測定点: ⑩D4アノード電流]', 'D4 schottky P2'),
    'D5': jlabel('D5 (Vz=5.1V)\nツェナークランプ\n[測定点: スナバ回路内クランプ電流]', 'D5 zener clamp'),
}
D_cols = [MINT, PLUM, SKY, ROSE, LIME]
for idx, (dk, info) in enumerate(D_info.items()):
    ax = make_ax(gs[6, idx % 4], info, jlabel('電流 [A]', 'Current [A]'))
    ax.plot(ts, ds(iD[dk]), color=D_cols[idx], lw=1.0)
    ax.fill_between(ts, ds(iD[dk]), alpha=0.12, color=D_cols[idx])
    ax.set_ylim(bottom=-0.05)
    # VF水平線
    ax2 = ax.twinx()
    ax2.axhline(VF[dk], color=WHITE, lw=0.8, ls=':', alpha=0.6)
    ax2.set_ylabel(f'VF={VF[dk]}V', color=GRAY, fontsize=6)
    ax2.tick_params(colors=GRAY, labelsize=6)
    ax2.set_ylim(0, max(VF[dk]*3, 1))

# ══ ROW 7: MOSFET Q1 ══
ax_q1v = make_ax(gs[7, 0],
    jlabel('Q1 MOSFET  Vgs / Vds\n[測定点: ⑦ドレイン-ソース間, ゲート-ソース間]',
           'Q1 MOSFET Vgs/Vds [⑦]'),
    jlabel('電圧 [V]', 'V'))
ax_q1v.plot(ts, ds(v_gate),  color=PLUM, lw=1.0, label='Vgs (gate)')
ax_q1v.plot(ts, ds(v_drain), color=SKY,  lw=1.0, label='Vds (drain)', alpha=0.85)
ax_q1v.set_ylim(-5, 160)
ax_q1v.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
annotate_meas(ax_q1v, jlabel('測定点⑦', 'Point⑦'))

ax_q1i = make_ax(gs[7, 1],
    jlabel('Q1 MOSFET  ドレイン電流\n[測定点: T1一次巻線-Q1ドレイン間]',
           'Q1 Drain Current'),
    jlabel('電流 [A]', 'I [A]'))
ax_q1i.plot(ts, ds(i_pri)*ds(sw), color=CORAL, lw=1.0)
ax_q1i.fill_between(ts, ds(i_pri)*ds(sw), alpha=0.15, color=CORAL)
ax_q1i.set_ylim(bottom=-0.02)
annotate_meas(ax_q1i, jlabel('測定点⑤', 'Point⑤'))

# ══ ROW 7: 制御IC U3 ══
ax_duty = make_ax(gs[7, 2],
    jlabel('制御IC U3  PWMデューティ比\n[測定点: ⑧ゲート駆動パルス幅]',
           'Control IC U3 PWM Duty [⑧]'),
    jlabel('Duty [%]', 'Duty [%]'))
ax_duty.plot(ts, ds(duty)*100, color=MINT, lw=1.2)
ax_duty.axhline(30, color=GOLD, lw=0.8, ls='--', alpha=0.7,
                label=jlabel('定常デューティ 30%', 'nominal 30%'))
ax_duty.set_ylim(0, 90)
ax_duty.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
annotate_meas(ax_duty, jlabel('測定点⑧', 'Point⑧'))

# 帰還電圧
ax_fb = make_ax(gs[7, 3],
    jlabel('制御IC 帰還電圧 R4/R5分圧\n[測定点: ⑨FB端子-GND間]',
           'Feedback voltage R4/R5 [⑨]'),
    jlabel('電圧 [V]', 'V'))
v_fb = ds(vR['R5'])   # FB = R5両端 (GND基準)
ax_fb.plot(ts, v_fb, color=LIME, lw=1.1)
ax_fb.axhline(2.5, color=GOLD, lw=0.8, ls='--', alpha=0.7,
              label=jlabel('FB基準 2.5V', 'FB ref 2.5V'))
ax_fb.set_ylim(0, 6)
ax_fb.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
annotate_meas(ax_fb, jlabel('測定点⑨', 'Point⑨'))

# ══ ROW 8: トランス T1 ══
ax_t1 = make_ax(gs[8, :2],
    jlabel('トランス T1 (QL015Z)  一次電流 / 二次電圧\n[一次: ⑤T1一次巻線電流  /  二次: ⑥T1二次端子電圧]',
           'Transformer T1: Primary current [⑤] / Secondary voltage [⑥]'),
    jlabel('一次電流 [A]', 'Primary I [A]'))
ax_t1.plot(ts, ds(i_pri),  color=TEAL, lw=1.1, label=jlabel('一次電流 [A]  ⑤', 'Primary I [A] ⑤'))
ax_t1.fill_between(ts, ds(i_pri), alpha=0.12, color=TEAL)
ax2_t1 = ax_t1.twinx()
ax2_t1.plot(ts, ds(v_sec), color=GOLD, lw=1.0, ls='--', alpha=0.85,
            label=jlabel('二次電圧 [V]  ⑥', 'Secondary V [V] ⑥'))
ax2_t1.set_ylabel(jlabel('二次電圧 [V]', 'Secondary V [V]'), color=GOLD, fontsize=7)
ax2_t1.tick_params(colors=GOLD, labelsize=7)
ax2_t1.set_ylim(-0.5, 12)
ax_t1.set_ylim(-0.02, 1.0)
lines1, lab1 = ax_t1.get_legend_handles_labels()
lines2, lab2 = ax2_t1.get_legend_handles_labels()
ax_t1.legend(lines1+lines2, lab1+lab2, fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)

# ══ ROW 8: NTCサーミスタ ══
ax_ntc = make_ax(gs[8, 2],
    jlabel('NTCサーミスタ R3\n抵抗値 vs 温度上昇\n[測定点: ①R3両端インピーダンス]',
           'NTC Thermistor R3 [①]'),
    jlabel('抵抗 [Ω]', 'R [Ω]'))
ax_ntc.plot(ts, ds(R_ntc), color=CORAL, lw=1.2)
ax_ntc2 = ax_ntc.twinx()
ax_ntc2.plot(ts, ds(T_ntc), color=GOLD, lw=0.9, ls='--', alpha=0.75)
ax_ntc2.set_ylabel(jlabel('温度 [°C]', 'Temp [°C]'), color=GOLD, fontsize=7)
ax_ntc2.tick_params(colors=GOLD, labelsize=7)
annotate_meas(ax_ntc, jlabel('測定点①', 'Point①'))

# スナバ電圧・電流
ax_snb = make_ax(gs[8, 3],
    jlabel('スナバ回路 (R10+C3+D5)\n電圧スパイク吸収の様子',
           'Snubber R10+C3+D5'),
    jlabel('電圧 [V]', 'V'))
ax_snb.plot(ts, ds(vC['C3']), color=LIME,  lw=1.0, label='C3 voltage')
ax_snb.plot(ts, ds(vR['R10']),color=CORAL, lw=1.0, label='R10 voltage', alpha=0.8)
ax_snb.axhline(VF['D5'], color=PLUM, lw=0.8, ls=':', label=f'D5 Vz={VF["D5"]}V')
ax_snb.set_ylim(bottom=-1)
ax_snb.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)

# ══ ROW 9: USB出力 Port1/Port2 ══
ax_p1 = make_ax(gs[9, 0],
    jlabel('USB-A Port1  出力電圧\n[測定点: ⑪C6両端 / USB VBUS-GND間]',
           'USB-A Port1 Output [⑪]'),
    jlabel('電圧 [V]', 'V'))
ax_p1.plot(ts, ds(v_out1), color=MINT, lw=1.3)
ax_p1.axhline(5.0, color=GOLD, lw=0.8, ls='--', label='5.0V')
ax_p1.axhline(4.75, color=CORAL, lw=0.6, ls=':', label=jlabel('下限 4.75V', 'min 4.75V'), alpha=0.7)
ax_p1.set_ylim(4.3, 5.7)
ax_p1.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
annotate_meas(ax_p1, jlabel('測定点⑪', 'Point⑪'))

ax_p2 = make_ax(gs[9, 1],
    jlabel('USB-A Port2  出力電圧\n[測定点: ⑫C7両端 / USB VBUS-GND間]',
           'USB-A Port2 Output [⑫]'),
    jlabel('電圧 [V]', 'V'))
ax_p2.plot(ts, ds(v_out2), color=ROSE, lw=1.3)
ax_p2.axhline(5.0, color=GOLD, lw=0.8, ls='--', label='5.0V')
ax_p2.axhline(4.75, color=CORAL, lw=0.6, ls=':', label=jlabel('下限 4.75V', 'min 4.75V'), alpha=0.7)
ax_p2.set_ylim(4.3, 5.7)
ax_p2.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
annotate_meas(ax_p2, jlabel('測定点⑫', 'Point⑫'))

# ══ ROW 9: 効率 ══
p_in_arr  = np.abs(v_ac) * np.abs(v_ac) / (R_val['R1']+R_val['R2']) + i_pri*v_bulk
kern      = np.ones(int(FS*0.002)) / int(FS*0.002)   # 2ms 移動平均
p_in_avg2 = np.convolve(p_in_arr, kern, mode='same')
eta2      = np.clip((V_OUT*I_LOAD) / np.where(p_in_avg2>0.1, p_in_avg2, 1.), 0, 1)*100

ax_eff2 = make_ax(gs[9, 2],
    jlabel('変換効率 推定\n[P_out=V_out×I_load / P_in=AC入力電力]',
           'Conversion Efficiency'),
    jlabel('効率 [%]', 'Efficiency [%]'))
ax_eff2.plot(ts, ds(eta2), color=LIME, lw=1.2)
ax_eff2.axhline(80, color=GOLD, lw=0.8, ls='--', label='80%')
ax_eff2.set_ylim(0, 110)
ax_eff2.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)

# ══ ROW 9: 電力フロー棒グラフ ══
ax_pwr2 = make_ax(gs[9, 3],
    jlabel('定常状態 電力フロー [W]', 'Steady-state Power Flow [W]'), '')
pwr_labels = [
    jlabel('AC入力\n(実効)', 'AC input'),
    jlabel('EMIロス', 'EMI loss'),
    jlabel('整流ロス\nD1/D2', 'Bridge loss'),
    jlabel('FET\nスイッチ損', 'FET sw loss'),
    jlabel('トランス\n鉄損/銅損', 'Transformer'),
    jlabel('USB出力\n×2合計', 'USB output'),
]
pwr_vals = [V_AC*I_LOAD/0.80, 0.3, 2*VF['D1']*I_LOAD, 1.2, 0.8, V_OUT*I_LOAD]
pwr_cols = [CORAL, GRAY, GOLD, PLUM, TEAL, MINT]
bars2 = ax_pwr2.barh(pwr_labels, pwr_vals, color=pwr_cols, edgecolor=BORDER, height=0.6)
ax_pwr2.set_xlabel(jlabel('電力 [W]', 'Power [W]'), color=WHITE, fontsize=7)
ax_pwr2.tick_params(colors=WHITE, labelsize=6.5)
for sp in ax_pwr2.spines.values(): sp.set_color(BORDER)
for bar, val in zip(bars2, pwr_vals):
    ax_pwr2.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}W', va='center', color=WHITE, fontsize=7)

# ══ ROW 10: リップル拡大 ══
# 最後の1サイクル付近を拡大
t_zoom_start = 40.0   # 40ms〜50ms
mask = ts >= t_zoom_start

ax_rip1 = make_ax(gs[10, 0],
    jlabel('USB-A Port1 出力リップル拡大 (40〜50ms)\n[C6の充放電による微小変動]',
           'Port1 Output Ripple Zoom (40-50ms)'),
    jlabel('電圧 [V]', 'V'))
ax_rip1.plot(ts[mask], ds(v_out1)[mask], color=MINT, lw=1.2)
ax_rip1.axhline(5.0, color=GOLD, lw=0.8, ls='--')
v1z = ds(v_out1)[mask]
if len(v1z) > 0:
    ax_rip1.set_ylim(v1z.min()-0.01, v1z.max()+0.01)

ax_rip2 = make_ax(gs[10, 1],
    jlabel('DCバス リップル拡大 (40〜50ms)\n[60Hz整流リップル]',
           'DC Bus Ripple Zoom (40-50ms)'),
    jlabel('電圧 [V]', 'V'))
ax_rip2.plot(ts[mask], ds(v_bulk)[mask], color=TEAL, lw=1.2)
vbz = ds(v_bulk)[mask]
if len(vbz) > 0:
    ax_rip2.set_ylim(vbz.min()-1, vbz.max()+1)

ax_rip3 = make_ax(gs[10, 2],
    jlabel('トランス一次電流 拡大 (40〜50ms)\n[65kHz 鋸波状スイッチング電流]',
           'Primary Current Zoom (40-50ms)'),
    jlabel('電流 [A]', 'I [A]'))
ax_rip3.plot(ts[mask], ds(i_pri)[mask], color=CORAL, lw=1.1)
ax_rip3.fill_between(ts[mask], ds(i_pri)[mask], alpha=0.15, color=CORAL)

ax_rip4 = make_ax(gs[10, 3],
    jlabel('全出力電流 (Port1+Port2)\n[定常負荷電流 合計 2.2A]',
           'Total Output Current (Port1+Port2)'),
    jlabel('電流 [A]', 'I [A]'))
i_total = ds(iD['D3']) + ds(iD['D4'])
ax_rip4.plot(ts, i_total, color=SKY, lw=1.1,
             label=jlabel('合計出力電流', 'total output'))
ax_rip4.axhline(I_LOAD, color=GOLD, lw=0.8, ls='--',
                label=f'{I_LOAD:.1f}A')
ax_rip4.legend(fontsize=7.5, facecolor='#1C2128', labelcolor=WHITE)
ax_rip4.set_ylim(bottom=-0.1)

fig.tight_layout(rect=[0, 0, 1, 0.994])

# ─── 保存 ───
base = os.path.dirname(os.path.abspath(__file__))
p_ckt  = os.path.join(base, 'usb_charger_circuit_diagram.png')
p_all  = os.path.join(base, 'usb_charger_all_graphs.png')

fig_ckt.savefig(p_ckt, dpi=140, bbox_inches='tight', facecolor=BG)
fig.savefig(p_all,     dpi=120, bbox_inches='tight', facecolor=BG)
print(f'回路図 saved: {p_ckt}')
print(f'全グラフ saved: {p_all}')
plt.show()