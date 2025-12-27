import matplotlib.pyplot as plt
import matplotlib.animation as anim
from core.simulation_3d import SimulationController3D

class SliceVisualizer:
    def __init__(self, sim: SimulationController3D):
        self.sim = sim
        self.slice_k = sim.nz // 2  # Z-slice index (middle)
        
    def step(self):
        self.sim.step()
        
    def get_slice(self):
        # Get 2D slice of smoke field at z=slice_k
        smoke_np = self.sim.grid.smoke0.numpy()
        return smoke_np[:, :, self.slice_k].T  # Transpose for correct orientation
    
    def step_and_render(self, frame_num=None, img=None):
        self.step()
        
        if img is not None:
            img.set_array(self.get_slice())
        
        return (img,)

def run_visualization(sim: SimulationController3D):
    # Create visualizer
    viz = SliceVisualizer(sim)
    
    fig, ax = plt.subplots()
    ax.set_title(f"Smoke Density (Z-slice at k={viz.slice_k})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Initial image
    img = ax.imshow(
        viz.get_slice(),
        origin="lower",
        animated=True,
        interpolation="antialiased",
        cmap="gray",
        vmin=0.0,
        vmax=1.0
    )
    plt.colorbar(img, ax=ax, label="Density")
    
    # Animation
    seq = anim.FuncAnimation(
        fig,
        viz.step_and_render,
        fargs=(img,),
        frames=100000,
        blit=True,
        interval=16,
        repeat=False
    )
    
    plt.show()
