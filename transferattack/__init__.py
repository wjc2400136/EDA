"""Attack registry for the six methods evaluated in the EDA paper."""

import importlib


attack_zoo = {
    "l2t": (".input_transformation.l2t", "L2T"),
    "bsr": (".input_transformation.bsr", "BSR"),
    "decowa": (".input_transformation.decowa", "DeCowA"),
    "ops": (".input_transformation.ops", "OPS"),
    "sid": (".input_transformation.sid", "SID"),
    "eda": (".input_transformation.eda", "EDA"),
}


def load_attack_class(attack_name):
    """Load one of the six supported attack classes by command-line name."""
    normalized_name = attack_name.lower()
    if normalized_name not in attack_zoo:
        supported = ", ".join(attack_zoo)
        raise ValueError(
            f"Unsupported attack algorithm {attack_name!r}. "
            f"Supported methods: {supported}."
        )
    module_path, class_name = attack_zoo[normalized_name]
    module = importlib.import_module(module_path, __package__)
    return getattr(module, class_name)


__version__ = "1.0.0"
