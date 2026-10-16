"""Keep the LeRobot choice discriminator when HubMixin serializes a dataclass."""
import json
from pathlib import Path


def preserve_policy_type(directory, policy_type):
    # Some HubMixin/draccus combinations overwrite the config after _save_pretrained
    # and omit the derived `type` property. The type is taken from the loaded policy,
    # never guessed from an unknown checkpoint.
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("type") not in (None, policy_type):
        raise ValueError("Serialized policy type does not match the loaded model")
    config["type"] = policy_type
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
