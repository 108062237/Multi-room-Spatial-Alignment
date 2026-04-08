from .lgtnet_runner import LGTNetRunner
from .horizonnet_runner import HorizonNetRunner
from .dmhnet_runner import DMHNetRunner
from .hohonet_runner import HoHoNetRunner

MODEL_REGISTRY = {
    "lgtnet": LGTNetRunner,
    "horizonnet": HorizonNetRunner,
    "dmhnet": DMHNetRunner,
    "hohonet": HoHoNetRunner,
}