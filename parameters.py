import torch
from norse.torch.functional.lif_refrac import LIFRefracParameters
from dataclasses import dataclass


@dataclass
class EILIFParameters:
    tau_ex_syn_inv: torch.Tensor = torch.as_tensor(
        1 / (20 * 1e-3), dtype=torch.double)
    tau_ih_syn_inv: torch.Tensor = torch.as_tensor(
        1 / (50 * 1e-3), dtype=torch.double)

    tau_mem_inv: torch.Tensor = torch.as_tensor(
        1 / (20 * 1e-3), dtype=torch.double)
    tau_adaptation_inv: torch.Tensor = torch.as_tensor(
        1 / (200 * 1e-3), dtype=torch.double)

    R: torch.Tensor = torch.as_tensor(10 * 1e-3, dtype=torch.double)
    v_leak: torch.Tensor = torch.as_tensor(-65.0 * 1e-3, dtype=torch.double)
    v_th: torch.Tensor = torch.as_tensor(-50.0 * 1e-3, dtype=torch.double)
    v_reset: torch.Tensor = torch.as_tensor(-65.0 * 1e-3, dtype=torch.double)

    dale: bool = True
    ei_ratio: float = 0.8
    beta: float = 1.6
    sfa_ratio: float = 0.4

    rho: float = 1.5
    current_base_scale: float = 34
    current_base_lower: float = 1.5
    current_base_upper: float = 2.5
    current_base_mu: float = 2.
    current_base_sigma: float = 0.1

    rand_current_std: float = 0.0015
    rand_voltage_std: float = 0.0015
    rand_walk_alpha: float = 1.

    method: str = "super"
    alpha: float = 1000.


@dataclass
class EILIFRefracParameters(LIFRefracParameters):
    lif: EILIFParameters = EILIFParameters()
    rho_reset: torch.Tensor = torch.as_tensor(3.0)


@dataclass
class pret_settings:
    region: str
    scale: float
    start_ts: int
    end_ts: int