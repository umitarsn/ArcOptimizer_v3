import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

def draw_box(ax, x, y, z, dx, dy, dz, color="gray", alpha=0.4):
    X = np.array([[x, x + dx], [x, x + dx]])
    Y = np.array([[y, y], [y + dy, y + dy]])
    Zb = np.array([[z, z], [z, z]])
    Zt = np.array([[z + dz, z + dz], [z + dz, z + dz]])
    ax.plot_surface(X, Y, Zb, color=color, alpha=alpha)
    ax.plot_surface(X, Y, Zt, color=color, alpha=alpha)

def draw_ferrochrome_plant(ax):
    Xg, Yg = np.meshgrid(np.linspace(-40, 40, 2), np.linspace(-20, 20, 2))
    Zg = np.zeros_like(Xg)
    ax.plot_surface(Xg, Yg, Zg, color="#444444", alpha=0.3)
    draw_box(ax, -5, -5, 0, 10, 10, 8)
    r=3
    theta=np.linspace(0,2*np.pi,40)
    zc=np.linspace(0,6,20)
    T,Z=np.meshgrid(theta,zc)
    X=r*np.cos(T); Y=r*np.sin(T)
    ax.plot_surface(X,Y,Z,color="#999999",alpha=0.8)

def draw_eaf_temperature(ax):
    R_max=3; Z_min=0; Z_max=6
    r=np.linspace(0,R_max,40)
    z=np.linspace(Z_min,Z_max,40)
    R,Z=np.meshgrid(r,z)
    T=1600+ (2800-1600)*np.exp(-((R)**2 + (Z-4.5)**2)/(2*1.0**2))
    theta=np.linspace(0,2*np.pi,60)
    Theta,_=np.meshgrid(theta,z)
    R3=np.repeat(R[:,:,None], len(theta),axis=2)
    Z3=np.repeat(Z[:,:,None], len(theta),axis=2)
    T3=np.repeat(T[:,:,None], len(theta),axis=2)
    Theta3=np.repeat(Theta[None,:,:], len(r),axis=0)
    X=R3*np.cos(Theta3); Y=R3*np.sin(Theta3)
    norm=plt.Normalize(T3.min(),T3.max()); colors=cm.inferno(norm(T3))
    for k in range(0,40,2):
        ax.plot_surface(X[k],Y[k],Z3[k],facecolors=colors[k],linewidth=0,shade=False)
    mappable=cm.ScalarMappable(cmap=cm.inferno,norm=norm)
    plt.colorbar(mappable, ax=ax, shrink=0.6,pad=0.1)

st.set_page_config(page_title="Ferrochrome 3D", layout="wide")
st.title("Ferrochrome Tesisi 3D + DC EAF İç CFD Demo")

fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(1,2,1,projection="3d")
draw_ferrochrome_plant(ax1)

ax2=fig.add_subplot(1,2,2,projection="3d")
draw_eaf_temperature(ax2)

st.pyplot(fig)
