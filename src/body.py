from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlyBody:
    """Small virtual body: not biology, just a visible actuator/state layer."""
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    energy: float = 1.0
    wing_left: float = 0.0
    wing_right: float = 0.0
    moves_seen: int = 0

    def step(self, confidence: float, legal: bool = True) -> None:
        self.moves_seen += 1
        self.energy = max(0.0, self.energy - 0.002)
        flap = max(0.0, min(1.0, confidence)) if legal else 0.1
        self.wing_left = flap
        self.wing_right = 1.0 - flap
        self.x += 0.02 * (flap - 0.5)
        self.heading += 0.03 * (flap - 0.5)

    def draw(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        ax.clear()
        ax.set_aspect("equal")
        ax.add_patch(Ellipse((self.x, self.y), 0.6, 0.35, color="black"))
        ax.add_patch(Ellipse((self.x - 0.18, self.y + 0.23), 0.45, 0.12, angle=20, alpha=0.55, color="skyblue"))
        ax.add_patch(Ellipse((self.x + 0.18, self.y + 0.23), 0.45, 0.12, angle=-20, alpha=0.55, color="skyblue"))
        ax.text(-1.4, 1.0, f"energy={self.energy:.2f}  moves={self.moves_seen}")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.0, 1.2); ax.axis("off")
        return ax
