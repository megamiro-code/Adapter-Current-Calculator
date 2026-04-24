import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings; warnings.filterwarnings('ignore')

FS=100_000; T_SIM=0.05
t=np.linspace(0,T_SIM,int(FS*T_SIM),endpoint=False)
dt=t[1]-t[0]
V_PEAK=100*np.sqrt(2); F_AC=60; F_SW=65000; V_OUT=5; R_LOAD=5/2.2

v_ac=V_PEAK*np.sin(2*np.pi*F_AC*t)
v_rect=np.clip(np.abs(v_ac)-1.4,0,None)
v_bulk=np.zeros_like(v_rect); v_bulk[0]=V_PEAK-1.4
for i in range(1,len(t)):
    v_bulk[i]=v_rect[i] if v_rect[i]>v_bulk[i-1] else v_bulk[i-1]*np.exp(-dt/(R_LOAD*330e-6))
sw=(np.sin(2*np.pi*F_SW*t)>0.3).astype(float)
i_sw=np.zeros_like(t)
for i in range(1,len(t)):
    if sw[i]>0.5:
        i_sw[i]=min(i_sw[i-1]+v_bulk[i]/800e-6*dt,0.8)
    else:
        i_sw[i]=max(0,i_sw[i-1]-0.001)
v_sec=v_bulk*(V_OUT/(V_PEAK-1.4))*sw
v_o1=np.zeros_like(t); v_o1[0]=5.0
v_o2=np.zeros_like(t); v_o2[0]=5.0
for i in range(1,len(t)):
    vi=max(0,v_sec[i]-0.45)
    if vi>v_o1[i-1]:
        v_o1[i]=v_o1[i-1]+(vi-v_o1[i-1])*0.1
    else:
        v_o1[i]=v_o1[i-1]-(1.2/680e-6)*dt
    v_o1[i]=np.clip(v_o1[i],4.5,5.5)
    if vi>v_o2[i-1]:
        v_o2[i]=v_o2[i-1]+(vi-v_o2[i-1])*0.1
    else:
        v_o2[i]=v_o2[i-1]-(1.0/680e-6)*dt
    v_o2[i]=np.clip(v_o2[i],4.5,5.5)

S=50; ts=t[::S]*1000; N=len(ts)
arrs=[v_ac[::S],v_rect[::S],v_bulk[::S],i_sw[::S],v_sec[::S],v_o1[::S],v_o2[::S]]
cols=['#FF6F61','#FFC857','#00838F','#7B5EA7','#4ECDC4','#A8E6CF','#FF6B9D']
titles=['AC入力 [V]','整流後 [V]','バルクC [V]','FET電流 [A]','二次側 [V]','USB-Port1 [V]','USB-Port2 [V]']
ylims=[(-160,160),(-5,160),(80,160),(-0.05,1.0),(-1,15),(4.0,6.0),(4.0,6.0)]

FRAMES=100; SKIP=max(1,N//FRAMES); WIN=200

fig,axes=plt.subplots(3,3,figsize=(16,10),facecolor='#0D1117')
fig.suptitle('スケルトンUSB充電器 アニメーション / AC100V → USB-A 5V×2', color='white', fontsize=12, fontweight='bold')

line_objs=[]; dot_objs=[]
for i in range(7):
    ax=axes.flat[i]
    ax.set_facecolor('#161B22')
    ax.set_title(titles[i],color='white',fontsize=8)
    ax.set_ylim(*ylims[i])
    ax.tick_params(colors='white',labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#30363D')
    ln,=ax.plot([],[],color=cols[i],lw=1.3)
    dt2,=ax.plot([],[],marker='o',color='white',ms=5,zorder=5,ls='')
    if 'Port' in titles[i]:
        ax.axhline(5,color='#FFC857',lw=0.8,ls='--',alpha=0.7)
    line_objs.append(ln); dot_objs.append(dt2)

# hide unused subplots, use last 2 for block diagram text
for j in [7,8]:
    axes.flat[j].set_facecolor('#0D1117')
    axes.flat[j].axis('off')

# Circuit info text box
ax_info=axes.flat[7]
ax_info.set_facecolor('#161B22'); ax_info.set_visible(True)
info_txt=ax_info.text(0.05,0.5,'',transform=ax_info.transAxes,color='white',fontsize=8,va='center',
                      bbox=dict(boxstyle='round',facecolor='#21262D',edgecolor='#30363D'))
ax_info.set_title('現在値',color='white',fontsize=9)

ax_bar=axes.flat[8]; ax_bar.set_visible(True)
ax_bar.set_facecolor('#161B22'); ax_bar.set_title('電圧バー',color='white',fontsize=9)
blabels=['AC','Rect','Bulk','FET×100','Sec×5','P1','P2']
bcolors=['#FF6F61','#FFC857','#00838F','#7B5EA7','#4ECDC4','#A8E6CF','#FF6B9D']
bcont=ax_bar.bar(blabels,[0]*7,color=bcolors,edgecolor='#30363D')
ax_bar.set_ylim(-20,200); ax_bar.tick_params(colors='white',labelsize=7)
for sp in ax_bar.spines.values(): sp.set_color('#30363D')

tlbl=fig.text(0.5,0.01,'',ha='center',color='#FFC857',fontsize=9)
plt.tight_layout(rect=[0,0.04,1,0.96])

def update(frame):
    idx=min(frame*SKIP,N-1); st=max(0,idx-WIN); tw=ts[st:idx+1]
    for i,(ln,dt2) in enumerate(zip(line_objs,dot_objs)):
        arr2=arrs[i][st:idx+1]
        ln.set_data(tw,arr2)
        if len(arr2):
            dt2.set_data([ts[idx]],[arr2[-1]])
        axes.flat[i].set_xlim(ts[max(0,idx-WIN)],ts[idx]+0.3)
    bnow=[arrs[0][idx],arrs[1][idx],arrs[2][idx],arrs[3][idx]*100,arrs[4][idx]*5,arrs[5][idx],arrs[6][idx]]
    for bar,val in zip(bcont,bnow):
        bar.set_height(val)
    info_txt.set_text(
        f"v_ac   = {arrs[0][idx]:.1f} V\n"
        f"v_rect = {arrs[1][idx]:.1f} V\n"
        f"v_bulk = {arrs[2][idx]:.1f} V\n"
        f"i_FET  = {arrs[3][idx]:.3f} A\n"
        f"v_sec  = {arrs[4][idx]:.3f} V\n"
        f"Port1  = {arrs[5][idx]:.4f} V\n"
        f"Port2  = {arrs[6][idx]:.4f} V"
    )
    tlbl.set_text(f"t={ts[idx]:.2f}ms  |  Bulk={arrs[2][idx]:.1f}V  |  Port1={arrs[5][idx]:.4f}V  |  Port2={arrs[6][idx]:.4f}V")
    return []

ani=animation.FuncAnimation(fig,update,frames=FRAMES,interval=60,blit=False)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usb_charger_animation.gif')
ani.save(out_path, writer=animation.PillowWriter(fps=15), dpi=85)
print(f'GIF 保存完了: {out_path}')
plt.show()