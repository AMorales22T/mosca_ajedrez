import dataclasses

@dataclasses.dataclass
class HardwareProfile:
    name: str
    max_neurons: int
    batch_size: int

PROFILES = {
    "tiny": HardwareProfile("tiny", 5000, 16),
    "laptop": HardwareProfile("laptop", 30000, 64),
    "full": HardwareProfile("full", 150000, 2), # full connectome, small batch
}

def get_profile(name: str = "laptop") -> HardwareProfile:
    return PROFILES.get(name, PROFILES["laptop"])
