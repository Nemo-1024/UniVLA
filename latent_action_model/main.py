from lightning.pytorch.cli import LightningCLI
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.genie.dataset import LightningOpenX





cli = LightningCLI(
    VJEPA_LAM,
    # DINO_LAM,
    LightningOpenX,
    seed_everything_default=42,
    save_config_kwargs={"overwrite": True}
)
