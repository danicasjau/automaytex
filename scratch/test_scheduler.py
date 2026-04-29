import sys, os
_VENV = r"D:\DANI\PROJECTS_2026\AutoTexturingMaya\mEnv\Lib\site-packages"
sys.path.insert(0, _VENV)

from diffusers import EulerDiscreteScheduler

_SDXL_SCHEDULER_CONFIG = {
    "beta_end":               0.012,
    "beta_schedule":          "scaled_linear",
    "beta_start":             0.00085,
    "clip_sample":            False,
    "interpolation_type":     "linear",
    "num_train_timesteps":    1000,
    "prediction_type":        "epsilon",
    "sample_max_value":       1.0,
    "set_alpha_to_one":       False,
    "skip_prk_steps":         True,
    "steps_offset":           1,
    "timestep_spacing":       "leading",
    "use_karras_sigmas":      False,
    "rescale_betas_zero_snr": False,
}

try:
    s = EulerDiscreteScheduler(**_SDXL_SCHEDULER_CONFIG)
    print("Success with **kwargs")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    s2 = EulerDiscreteScheduler.from_config(_SDXL_SCHEDULER_CONFIG)
    print("Success with from_config")
except Exception as e:
    import traceback
    traceback.print_exc()
