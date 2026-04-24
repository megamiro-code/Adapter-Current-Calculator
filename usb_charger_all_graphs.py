"""
スケルトンUSB充電器 回路シミュレーション（全グラフ表示版）
=============================================================
回路推定（画像解析より）:
  AC入力 100V/60Hz (日本規格, Type-A 2ピン)
  → EMIフィルタ（CXコンデンサ + コモンモードチョーク）
  → 整流ブリッジ（ダイオード4個）
  → バルクコンデンサ（330μF/200V）
  → フライバックコンバータ（NTCサーミスタ付き）
    - 制御IC: U3 (推定: AP3706 or 類似 PWMコントローラ)
    - トランス: QL015Z XOH 2501
    - スイッチングFET
  → 二次側整流（ショットキーダイオード × 2）
  → 出力フィルタ（ECOT 10V 680μF × 2）
  → USB-A出力ポート × 2 (5V/2.4A max each)

部品リスト (Gemini推定値を採用):
  抵抗:  R1〜R12  (12個)
  コンデンサ: C1〜C8  (8個)
  インダクタ: L1〜L3  (3個)
  ダイオード: D1〜D5  (5個)
  その他: NTCサーミスタ, 制御IC(U3), トランス(T1), MOSFET(Q1)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ─────────────── グローバルパラメータ ───────────────
FS      = 1_000_000   # サンプリング周波数 1MHz
T_SIM   = 0.05        # シミュレーション時間 50ms (AC3サイクル分)
t       = np.linspace(0, T_SIM, int(FS * T_SIM), endpoint=False)
dt      = t[1] - t[0]

F_AC    = 60.0        # AC周波数 [Hz]
V_AC    = 100.0       # AC実効値 [V] (日本規格)
V_PEAK  = V_AC * np.sqrt(2)   # ≈141.4 V

F_SW    = 65_000.0    # スイッチング周波数 65kHz
V_OUT   = 5.0         # 出力電圧 [V]
I_LOAD1 = 1.2         # USB-A ポート1 負荷電流 [A]
I_LOAD2 = 1.0         # USB-A ポート2 負荷電流 [A]
I_LOAD  = I_LOAD1 + I_LOAD2   # 合計負荷電流 2.2A
R_LOAD  = V_OUT / I_LOAD      # 等価負荷抵抗

# ─────────────── 部品定数 ───────────────
# --- 抵抗 (Ω) ---
R = {
    'R1':  1e6,    # AC入力ブリーダー
    'R2':  1e6,    # AC入力ブリーダー (対称)
    'R3':  10.0,   # NTC初期値 (常温25℃)
    'R4':  47e3,   # 帰還分圧 上
    'R5':  10e3,   # 帰還分圧 下
    'R6':  1.0,    # ゲート抵抗
    'R7':  100e3,  # ソフトスタート
    'R8':  22.0,   # 電流検出
    'R9':  4.7e3,  # バイアス
    'R10': 100.0,  # スナバ
    'R11': 10.0,   # 出力ダンピング CH1
    'R12': 10.0,   # 出力ダンピング CH2
}

# --- コンデンサ (F) ---
C = {
    'C1': 0.1e-6,   # CX EMIフィルタ (0.1μF/275V)
    'C2': 330e-6,   # バルク電解 (330μF/200V)
    'C3': 1e-9,     # スナバ
    'C4': 100e-12,  # ブートストラップ
    'C5': 47e-6,    # 補助電源平滑
    'C6': 680e-6,   # 出力電解 CH1 (ECOT 10V 680μF)
    'C7': 680e-6,   # 出力電解 CH2 (ECOT 10V 680μF)
    'C8': 4.7e-9,   # Yコンデンサ (青い部品)
}

# --- インダクタ (H) ---
L = {
    'L1': 1.0e-3,   # コモンモードチョーク (1mH)
    'L2': 470e-6,   # 差動モードチョーク
    'L3': 10e-6,    # 出力フィルタインダクタ
}

# --- ダイオード特性 ---
VF = {
    'D1': 0.7,   # 整流ブリッジ (×4等価, シリコン)
    'D2': 0.7,   # 整流ブリッジ
    'D3': 0.45,  # 二次側ショットキー CH1
    'D4': 0.45,  # 二次側ショットキー CH2
    'D5': 5.1,   # ツェナー (クランプ用)
}

# ─────────────── 波形計算 ───────────────

# 1) AC入力電圧
v_ac = V_PEAK * np.sin(2 * np.pi * F_AC * t)

# 2) EMIフィルタ後 (LPF近似, fc≈10kHz)
fc_emi = 10e3
alpha  = 1 / (1 + 1/(2*np.pi*fc_emi*dt))
v_emi  = np.zeros_like(v_ac)
v_emi[0] = v_ac[0]
for i in range(1, len(t)):
    v_emi[i] = alpha * v_emi[i-1] + (1-alpha) * v_ac[i]

# 3) 全波整流
v_rect = np.abs(v_emi) - 2 * VF['D1']   # ブリッジ2個分のVF
v_rect = np.clip(v_rect, 0, None)

# 4) バルクコンデンサ平滑 (簡易リップルシミュレーション)
tau_bulk = R_LOAD * C['C2']
v_bulk   = np.zeros_like(v_rect)
v_bulk[0] = V_PEAK - 2*VF['D1']
for i in range(1, len(t)):
    if v_rect[i] > v_bulk[i-1]:
        v_bulk[i] = v_rect[i]
    else:
        v_bulk[i] = v_bulk[i-1] * np.exp(-dt / tau_bulk)

# 5) スイッチング波形 (65kHz PWM)
sw_signal = (np.sin(2 * np.pi * F_SW * t) > 0.3).astype(float)

# 6) フライバックトランス一次側電流
# 鋸波状（エネルギー蓄積・放出）
i_primary = np.zeros_like(t)
L_mag = 800e-6   # 励磁インダクタンス
for i in range(1, len(t)):
    if sw_signal[i] > 0.5:
        i_primary[i] = i_primary[i-1] + (v_bulk[i] / L_mag) * dt
        if i_primary[i] > 0.8:
            i_primary[i] = 0.8   # 電流制限
    else:
        i_primary[i] = max(0, i_primary[i-1] - 0.3*dt/dt*0.001)

# 7) 二次側電圧（トランス巻数比 n≈1/15 → 5V出力）
n_ratio = V_OUT / (V_PEAK - 2*VF['D1'])  # ≈0.038
v_sec_raw = v_bulk * n_ratio * sw_signal

# 8) 出力電圧（C6/C7で平滑, ショットキー経由）
tau_out = (R_LOAD/2) * C['C6']
v_out1  = np.zeros_like(t)
v_out2  = np.zeros_like(t)
v_out1[0] = V_OUT
v_out2[0] = V_OUT
for i in range(1, len(t)):
    v_in = max(0, v_sec_raw[i] - VF['D3'])
    if v_in > v_out1[i-1]:
        v_out1[i] = v_out1[i-1] + (v_in - v_out1[i-1]) * 0.1
    else:
        v_out1[i] = v_out1[i-1] - (I_LOAD1 / C['C6']) * dt
    v_out1[i] = np.clip(v_out1[i], 4.5, 5.5)

    v_in2 = max(0, v_sec_raw[i] - VF['D4'])
    if v_in2 > v_out2[i-1]:
        v_out2[i] = v_out2[i-1] + (v_in2 - v_out2[i-1]) * 0.1
    else:
        v_out2[i] = v_out2[i-1] - (I_LOAD2 / C['C7']) * dt
    v_out2[i] = np.clip(v_out2[i], 4.5, 5.5)

# 9) 各抵抗の電圧
v_R = {
    'R1': np.abs(v_ac) / 2,
    'R2': np.abs(v_ac) / 2,
    'R3': i_primary * R['R3'],           # NTC (簡易)
    'R4': (v_out1 - 2.5) * (R['R4']/(R['R4']+R['R5'])),
    'R5': v_out1 * R['R5']/(R['R4']+R['R5']),
    'R6': i_primary * sw_signal * R['R6'],
    'R7': np.ones_like(t)*1.2,           # ソフトスタート (定常)
    'R8': i_primary * R['R8'],
    'R9': np.ones_like(t)*2.5,
    'R10': i_primary * sw_signal * R['R10'] * 0.1,
    'R11': I_LOAD1 * R['R11'] * np.ones_like(t),
    'R12': I_LOAD2 * R['R12'] * np.ones_like(t),
}
i_R = {k: v_R[k]/R[k] for k in R}

# 10) 各コンデンサの電圧
v_C = {
    'C1': v_ac * 0.05,
    'C2': v_bulk,
    'C3': i_primary * sw_signal * 50,    # スナバ
    'C4': np.ones_like(t) * 12.0,       # ブートストラップ
    'C5': np.ones_like(t) * 12.0,       # 補助電源
    'C6': v_out1,
    'C7': v_out2,
    'C8': v_ac * 0.01,                   # Yコン
}
i_C = {}
for k in C:
    dv = np.gradient(v_C[k], dt)
    i_C[k] = C[k] * dv

# 11) 各インダクタの電圧
v_L = {
    'L1': np.gradient(v_emi, dt) * L['L1'] * 0.001,
    'L2': np.gradient(v_emi, dt) * L['L2'] * 0.0005,
    'L3': np.gradient(v_out1, dt) * L['L3'],
}
i_L = {}
i_L['L1'] = np.cumsum(v_L['L1']) * dt / L['L1']
i_L['L2'] = np.cumsum(v_L['L2']) * dt / L['L2']
i_L['L3'] = I_LOAD * np.ones_like(t) + np.cumsum(v_L['L3']) * dt / L['L3'] * 0.001

# 12) 各ダイオードの電流
i_D = {
    'D1': np.clip(v_rect / R_LOAD, 0, 3.0),
    'D2': np.clip(v_rect / R_LOAD, 0, 3.0),
    'D3': np.clip((v_sec_raw - VF['D3']) / (R_LOAD/2), 0, 3.0) * sw_signal,
    'D4': np.clip((v_sec_raw - VF['D4']) / (R_LOAD/2), 0, 3.0) * sw_signal,
    'D5': np.zeros_like(t),  # ツェナー（通常OFF）
}
# D5: スナバ動作時のみON
v_peak_sw = v_bulk * 1.3  # スパイク電圧
i_D['D5'] = np.where(v_peak_sw > (v_bulk + VF['D5']), 0.1, 0.0)

v_D = {k: VF[k] * np.sign(i_D[k]) for k in VF}

# ─────────────── プロット ───────────────
# プロット用に間引き（1/100）
STEP  = 100
ts    = t[::STEP] * 1000   # ms単位

def ds(arr): return arr[::STEP]

TEAL  = '#00838F'
CORAL = '#FF6F61'
GOLD  = '#FFC857'
MINT  = '#A8E6CF'
PLUM  = '#7B5EA7'
SKY   = '#4ECDC4'
ROSE  = '#FF6B9D'
LIME  = '#C5E99B'
NAVY  = '#2C3E7A'

fig = plt.figure(figsize=(24, 40), facecolor='#0D1117')
fig.suptitle(
    'スケルトンUSB充電器 回路シミュレーション\n'
    '(AC100V/60Hz → フライバックコンバータ → USB-A 5V×2ポート)',
    fontsize=16, color='white', fontweight='bold', y=0.99
)

gs = gridspec.GridSpec(
    10, 4,
    figure=fig,
    hspace=0.55, wspace=0.35,
    top=0.97, bottom=0.02, left=0.06, right=0.98
)

# ────── ROW 0: 主要電圧フロー ──────
ax_main = fig.add_subplot(gs[0, :])
ax_main.set_facecolor('#161B22')
ax_main.plot(ts, ds(v_ac),   color=CORAL, lw=1.5, label='AC入力 v_ac')
ax_main.plot(ts, ds(v_rect), color=GOLD,  lw=1.2, label='整流後 v_rect', alpha=0.8)
ax_main.plot(ts, ds(v_bulk), color=TEAL,  lw=1.5, label='バルクC平滑 v_bulk')
ax_main.plot(ts, ds(v_out1)*10, color=MINT, lw=1.5, label='USB出力×10 v_out1')
ax_main.axhline(0, color='gray', lw=0.5, ls='--')
ax_main.set_title('主要電圧フロー（全段）', color='white')
ax_main.set_xlabel('時間 [ms]', color='white')
ax_main.set_ylabel('電圧 [V]', color='white')
ax_main.legend(loc='upper right', fontsize=8, facecolor='#21262D', labelcolor='white')
ax_main.tick_params(colors='white')
for sp in ax_main.spines.values(): sp.set_color('#30363D')

# ────── ROW 1: 抵抗 (R1〜R6) ──────
R_keys1 = ['R1','R2','R3','R4','R5','R6']
colors_r = [CORAL, GOLD, TEAL, MINT, PLUM, SKY]
for idx, (rk, col) in enumerate(zip(R_keys1, colors_r)):
    ax = fig.add_subplot(gs[1, idx % 4] if idx < 4 else gs[2, idx % 4])
    ax.set_facecolor('#161B22')
    ax.plot(ts, ds(v_R[rk]), color=col, lw=1.0)
    ax.fill_between(ts, ds(v_R[rk]), alpha=0.15, color=col)
    ax.set_title(f'{rk} ({R[rk]:.0f}Ω)\n電圧', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_ylabel('V', color='white', fontsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

# ────── ROW 1〜2: 抵抗 (R7〜R12) ──────
R_keys2 = ['R7','R8','R9','R10','R11','R12']
colors_r2 = [ROSE, LIME, CORAL, GOLD, TEAL, MINT]
axes_r = []
for idx, (rk, col) in enumerate(zip(R_keys2, colors_r2)):
    row = 1 + (idx + 2) // 4  # 2〜3行目に配置
    col_idx = (idx + 2) % 4
    ax = fig.add_subplot(gs[row, col_idx])
    ax.set_facecolor('#161B22')
    ax.plot(ts, ds(v_R[rk]), color=col, lw=1.0)
    ax.fill_between(ts, ds(v_R[rk]), alpha=0.15, color=col)
    ax.set_title(f'{rk} ({R[rk]:.0f}Ω)\n電圧', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_ylabel('V', color='white', fontsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

# ────── ROW 3: 抵抗電流 ──────
ax_ri = fig.add_subplot(gs[3, :2])
ax_ri.set_facecolor('#161B22')
for rk, col in zip(list(R.keys()), [CORAL,GOLD,TEAL,MINT,PLUM,SKY,ROSE,LIME,CORAL,GOLD,TEAL,MINT]):
    ax_ri.plot(ts, ds(i_R[rk])*1000, lw=0.8, label=rk, color=col, alpha=0.8)
ax_ri.set_title('全抵抗の電流', color='white')
ax_ri.set_ylabel('電流 [mA]', color='white')
ax_ri.set_xlabel('時間 [ms]', color='white')
ax_ri.legend(loc='upper right', fontsize=6, ncol=2, facecolor='#21262D', labelcolor='white')
ax_ri.tick_params(colors='white')
for sp in ax_ri.spines.values(): sp.set_color('#30363D')

# ────── ROW 3: コンデンサ電圧 (まとめ) ──────
ax_cv = fig.add_subplot(gs[3, 2:])
ax_cv.set_facecolor('#161B22')
colors_c = [CORAL, TEAL, GOLD, MINT, PLUM, SKY, ROSE, LIME]
for ck, col in zip(C.keys(), colors_c):
    ax_cv.plot(ts, ds(v_C[ck]), lw=0.9, label=ck, color=col)
ax_cv.set_title('全コンデンサの電圧', color='white')
ax_cv.set_ylabel('電圧 [V]', color='white')
ax_cv.set_xlabel('時間 [ms]', color='white')
ax_cv.legend(loc='upper right', fontsize=7, ncol=2, facecolor='#21262D', labelcolor='white')
ax_cv.tick_params(colors='white')
for sp in ax_cv.spines.values(): sp.set_color('#30363D')

# ────── ROW 4: コンデンサ個別電圧 ──────
C_keys = list(C.keys())
for idx, (ck, col) in enumerate(zip(C_keys, colors_c)):
    ax = fig.add_subplot(gs[4, idx % 4])
    ax.set_facecolor('#161B22')
    ax.plot(ts, ds(v_C[ck]), color=col, lw=1.0)
    ax.fill_between(ts, ds(v_C[ck]), alpha=0.15, color=col)
    ax.set_title(f'{ck} ({C[ck]*1e6:.1f}μF)\n電圧', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_ylabel('V', color='white', fontsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

# ────── ROW 5: コンデンサ電流 ──────
for idx, (ck, col) in enumerate(zip(C_keys[4:], colors_c[4:])):
    ax = fig.add_subplot(gs[5, idx % 4])
    ax.set_facecolor('#161B22')
    ax.plot(ts, ds(i_C[C_keys[4+idx]])*1000, color=col, lw=1.0)
    ax.set_title(f'{C_keys[4+idx]} 電流\n(充放電)', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_ylabel('mA', color='white', fontsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

# コンデンサ電流まとめ
ax_ci = fig.add_subplot(gs[5, 2:])
ax_ci.set_facecolor('#161B22')
for ck, col in zip(C_keys, colors_c):
    ic = np.clip(ds(i_C[ck])*1000, -500, 500)
    ax_ci.plot(ts, ic, lw=0.8, label=ck, color=col)
ax_ci.set_title('全コンデンサの電流（充放電）', color='white')
ax_ci.set_ylabel('電流 [mA]', color='white')
ax_ci.legend(loc='upper right', fontsize=7, ncol=2, facecolor='#21262D', labelcolor='white')
ax_ci.tick_params(colors='white')
for sp in ax_ci.spines.values(): sp.set_color('#30363D')

# ────── ROW 6: インダクタ電圧・電流 ──────
L_keys = list(L.keys())
colors_l = [TEAL, CORAL, GOLD]
for idx, (lk, col) in enumerate(zip(L_keys, colors_l)):
    # 電圧
    ax = fig.add_subplot(gs[6, idx])
    ax.set_facecolor('#161B22')
    vl = np.clip(ds(v_L[lk]), -50, 50)
    ax.plot(ts, vl, color=col, lw=1.0)
    ax.fill_between(ts, vl, alpha=0.15, color=col)
    ax.set_title(f'{lk} ({L[lk]*1e6:.0f}μH)\n電圧', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.set_ylabel('V', color='white', fontsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

    # 電流
    ax2 = fig.add_subplot(gs[6, idx+1]) if idx < 2 else fig.add_subplot(gs[7, 0])
    ax2.set_facecolor('#161B22')
    il = np.clip(ds(i_L[lk]), -5, 5)
    ax2.plot(ts, il, color=col, lw=1.0, ls='--')
    ax2.set_title(f'{lk} 電流', color='white', fontsize=8)
    ax2.tick_params(colors='white', labelsize=7)
    ax2.set_ylabel('A', color='white', fontsize=7)
    for sp in ax2.spines.values(): sp.set_color('#30363D')

# ────── ROW 7: ダイオード電流・電圧 ──────
D_keys = list(VF.keys())
colors_d = [MINT, PLUM, SKY, ROSE, LIME]
for idx, (dk, col) in enumerate(zip(D_keys, colors_d)):
    ax = fig.add_subplot(gs[7, idx % 4])
    ax.set_facecolor('#161B22')
    ax2 = ax.twinx()
    ax.plot(ts, ds(i_D[dk]), color=col, lw=1.0, label='電流')
    ax2.plot(ts, ds(np.ones_like(t)*VF[dk]), color='white', lw=0.8, ls=':', alpha=0.5, label='VF')
    ax.set_title(f'{dk} (VF={VF[dk]}V)\n電流 & VF', color='white', fontsize=8)
    ax.set_ylabel('A', color=col, fontsize=7)
    ax2.set_ylabel('V', color='white', fontsize=7)
    ax.tick_params(colors='white', labelsize=7)
    ax2.tick_params(colors='white', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#30363D')

# ────── ROW 8: 特殊部品 ──────
# NTCサーミスタ特性
T_amb = 25.0
B_coeff = 3950
T_ntc = T_amb + (i_primary**2 * R['R3'] * 0.001).cumsum() * dt * 1e4
T_ntc = np.clip(T_ntc, 25, 80)
R_ntc = R['R3'] * np.exp(B_coeff * (1/(T_ntc+273.15) - 1/298.15))

ax_ntc = fig.add_subplot(gs[8, 0])
ax_ntc.set_facecolor('#161B22')
ax_ntc.plot(ts, ds(R_ntc), color=CORAL, lw=1.2)
ax_ntc.set_title('NTCサーミスタ\n抵抗値変化', color='white', fontsize=8)
ax_ntc.set_ylabel('抵抗 [Ω]', color='white', fontsize=7)
ax_ntc.tick_params(colors='white', labelsize=7)
for sp in ax_ntc.spines.values(): sp.set_color('#30363D')

# トランス一次・二次電流
ax_tx = fig.add_subplot(gs[8, 1])
ax_tx.set_facecolor('#161B22')
ax_tx.plot(ts, ds(i_primary), color=TEAL, lw=1.0, label='一次電流')
i_sec = ds(i_primary) * (V_PEAK/V_OUT) * 0.038
ax_tx.plot(ts, i_sec * 0.1, color=GOLD, lw=1.0, label='二次電流/10')
ax_tx.set_title('トランスT1\n一次/二次電流', color='white', fontsize=8)
ax_tx.legend(fontsize=7, facecolor='#21262D', labelcolor='white')
ax_tx.tick_params(colors='white', labelsize=7)
ax_tx.set_ylabel('A', color='white', fontsize=7)
for sp in ax_tx.spines.values(): sp.set_color('#30363D')

# 制御IC (U3) - PWMデューティ比
duty = np.zeros_like(t)
# PID制御の簡易近似
err = V_OUT - (v_out1 + v_out2) / 2
duty_raw = 0.3 + np.cumsum(err) * dt * 10
duty = np.clip(duty_raw, 0.05, 0.8)

ax_ic = fig.add_subplot(gs[8, 2])
ax_ic.set_facecolor('#161B22')
ax_ic.plot(ts, ds(duty)*100, color=MINT, lw=1.0)
ax_ic.set_title('制御IC(U3)\nPWMデューティ比', color='white', fontsize=8)
ax_ic.set_ylabel('Duty [%]', color='white', fontsize=7)
ax_ic.set_ylim(0, 100)
ax_ic.tick_params(colors='white', labelsize=7)
for sp in ax_ic.spines.values(): sp.set_color('#30363D')

# MOSFET Q1 ゲート/ドレイン電圧
ax_fet = fig.add_subplot(gs[8, 3])
ax_fet.set_facecolor('#161B22')
v_gate   = ds(sw_signal) * 10.0
v_drain  = ds(v_bulk) * (1 - ds(sw_signal)) + ds(sw_signal) * 0.5
ax_fet.plot(ts, v_gate,  color=PLUM, lw=1.0, label='Vgs (ゲート)')
ax_fet.plot(ts, v_drain * 0.3, color=SKY, lw=0.8, label='Vds/3 (ドレイン)')
ax_fet.set_title('MOSFET (Q1)\nゲート/ドレイン電圧', color='white', fontsize=8)
ax_fet.legend(fontsize=7, facecolor='#21262D', labelcolor='white')
ax_fet.tick_params(colors='white', labelsize=7)
ax_fet.set_ylabel('V', color='white', fontsize=7)
for sp in ax_fet.spines.values(): sp.set_color('#30363D')

# ────── ROW 9: USB出力 & 電力 ──────
ax_usb1 = fig.add_subplot(gs[9, 0])
ax_usb1.set_facecolor('#161B22')
ax_usb1.plot(ts, ds(v_out1), color=MINT, lw=1.2)
ax_usb1.axhline(5.0, color=GOLD, lw=0.8, ls='--', label='5V基準')
ax_usb1.set_ylim(4.0, 6.0)
ax_usb1.set_title('USB-A ポート1\n出力電圧', color='white', fontsize=8)
ax_usb1.set_ylabel('電圧 [V]', color='white', fontsize=7)
ax_usb1.legend(fontsize=7, facecolor='#21262D', labelcolor='white')
ax_usb1.tick_params(colors='white', labelsize=7)
for sp in ax_usb1.spines.values(): sp.set_color('#30363D')

ax_usb2 = fig.add_subplot(gs[9, 1])
ax_usb2.set_facecolor('#161B22')
ax_usb2.plot(ts, ds(v_out2), color=SKY, lw=1.2)
ax_usb2.axhline(5.0, color=GOLD, lw=0.8, ls='--', label='5V基準')
ax_usb2.set_ylim(4.0, 6.0)
ax_usb2.set_title('USB-A ポート2\n出力電圧', color='white', fontsize=8)
ax_usb2.set_ylabel('電圧 [V]', color='white', fontsize=7)
ax_usb2.legend(fontsize=7, facecolor='#21262D', labelcolor='white')
ax_usb2.tick_params(colors='white', labelsize=7)
for sp in ax_usb2.spines.values(): sp.set_color('#30363D')

# 電力効率
p_in  = v_ac * (v_ac / (R['R1'] + R['R2'])) + i_primary * v_bulk
p_out = v_out1 * I_LOAD1 + v_out2 * I_LOAD2
# 移動平均
def moving_avg(arr, w=1000):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')

p_in_avg  = moving_avg(p_in, 5000)
p_out_val = V_OUT * I_LOAD  # 定常出力
eta       = np.clip(p_out_val / np.where(p_in_avg > 0.1, p_in_avg, 1.0), 0, 1) * 100

ax_eff = fig.add_subplot(gs[9, 2])
ax_eff.set_facecolor('#161B22')
ax_eff.plot(ts, ds(eta), color=LIME, lw=1.2)
ax_eff.axhline(80, color=GOLD, lw=0.8, ls='--', label='80%ライン')
ax_eff.set_ylim(0, 110)
ax_eff.set_title('変換効率\n推定', color='white', fontsize=8)
ax_eff.set_ylabel('効率 [%]', color='white', fontsize=7)
ax_eff.legend(fontsize=7, facecolor='#21262D', labelcolor='white')
ax_eff.tick_params(colors='white', labelsize=7)
for sp in ax_eff.spines.values(): sp.set_color('#30363D')

# 電力フローバー
ax_pwr = fig.add_subplot(gs[9, 3])
ax_pwr.set_facecolor('#161B22')
labels = ['AC入力', 'EMIロス', '整流ロス', 'FETスイッチ', 'USB-A×2出力']
values = [V_AC*0.22, 0.5, 2.0, 1.5, V_OUT*I_LOAD]
colors_bar = [CORAL, TEAL, GOLD, PLUM, MINT]
bars = ax_pwr.barh(labels, values, color=colors_bar, edgecolor='#30363D')
ax_pwr.set_title('定常状態\n電力フロー [W]', color='white', fontsize=8)
ax_pwr.set_xlabel('電力 [W]', color='white', fontsize=7)
ax_pwr.tick_params(colors='white', labelsize=7)
for sp in ax_pwr.spines.values(): sp.set_color('#30363D')
for bar, val in zip(bars, values):
    ax_pwr.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}W', va='center', color='white', fontsize=7)

plt.savefig('/mnt/user-data/outputs/usb_charger_all_graphs.png',
            dpi=130, bbox_inches='tight', facecolor='#0D1117')
print("all_graphs 保存完了")
