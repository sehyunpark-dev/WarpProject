import matplotlib.pyplot as plt
import matplotlib.animation as anim
from core.simulation_2d import SimulationController2D

class Visualizer2D:
    def __init__(self, sim: SimulationController2D):
        self.sim = sim

    def step(self):
        self.sim.step()

    def get_field(self):
        # Get 2D smoke field directly (no slicing needed)
        smoke_np = self.sim.grid.smoke0.numpy()
        return smoke_np.T  # Transpose for correct orientation (X horizontal, Y vertical)

    def step_and_render(self, frame_num=None, img=None):
        self.step()

        if img is not None:
            img.set_array(self.get_field())

        return (img,)

def run_visualization(sim: SimulationController2D):
    # Create visualizer
    viz = Visualizer2D(sim)

    fig, ax = plt.subplots()
    ax.set_title("2D Smoke Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Initial image
    img = ax.imshow(
        viz.get_field(),
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
