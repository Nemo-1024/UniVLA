from lightning.pytorch.cli import LightningCLI
from core.lam_lightinng import VJEPA_LAM
from genie.dataset import LightningOpenX
from genie.model import DINO_LAM

cli = LightningCLI(
    VJEPA_LAM,
    # DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
    save_config_kwargs={"overwrite": True}
)
