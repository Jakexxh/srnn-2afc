import torch
from torch import nn
import numpy as np
from numpy import linalg as LA
from norse.torch.functional.lif_refrac import compute_refractory_update
from norse.torch.module.lif_refrac import LIFRefracCell
from norse.torch.functional.lif_refrac import LIFRefracState
from norse.torch.functional.lif import LIFState
from norse.torch.functional.threshold import threshold
from scipy.stats import truncnorm
from typing import NamedTuple
from parameters import EILIFRefracParameters


class SFALIFState(NamedTuple):
    lif_rec_state: LIFRefracState
    sfa: torch.Tensor


class SNN_LIFCell(LIFRefracCell):

    def __init__(
            self,
            sensory_size,
            context_size,
            hidden_size,
            device,
            p: EILIFRefracParameters,
            dt: float = 0.002,
    ):
        super().__init__(sensory_size + context_size, hidden_size, p, dt)
        self.lif_p = p.lif
        self.device = device
        self.dale = self.lif_p.dale

        self.input_size = sensory_size + context_size
        self.hidden_size = hidden_size
        self.ei_ratio = self.lif_p.ei_ratio
        self.exc_size = int(hidden_size * self.lif_p.ei_ratio)
        self.sfa_size = int(hidden_size * self.lif_p.sfa_ratio)

        self.dt = torch.tensor(dt, dtype=torch.double, device=device)
        self.decay_sfa = np.exp(-dt * p.lif.tau_adaptation_inv)

        self.beta = self.lif_p.beta
        self.rho = self.lif_p.rho
        self.current_base_scale = self.lif_p.current_base_scale
        self.rand_current_std = self.lif_p.rand_current_std
        self.rand_voltage_std = self.lif_p.rand_voltage_std
        self.alpha = self.lif_p.rand_walk_alpha

        # Weights Initialization
        sinput_w = torch.empty(sensory_size, hidden_size, dtype=torch.double, device=device)
        sinput_w = nn.init.xavier_uniform_(sinput_w)
        self.sinput_weights = torch.nn.Parameter(sinput_w)

        cinputs_w = torch.empty(context_size, hidden_size, dtype=torch.double, device=device)
        cinputs_w = nn.init.xavier_uniform_(cinputs_w)
        self.cinput_weights = torch.nn.Parameter(cinputs_w)

        sd = np.sqrt(6.0 / (hidden_size + hidden_size))
        recurrent_weights_value = np.random.uniform(
            -sd, sd, (hidden_size, hidden_size))
        sr = np.max(np.absolute(LA.eigvals(recurrent_weights_value)))
        recurrent_weights_value = np.float64(
            (self.rho / sr) * recurrent_weights_value)
        self.recurrent_weights = torch.nn.Parameter(
            torch.tensor(recurrent_weights_value,
                         dtype=torch.double, device=device),
        )

        # tau
        if not self.dale and self.lif_p.tau_ex_syn_inv != self.lif_p.tau_ih_syn_inv:
            raise Exception("invalid syn tau set-up for no dale snn")
        syn_tau_inv = torch.zeros(hidden_size, dtype=torch.double, device=device)
        syn_tau_inv[:self.exc_size] = self.lif_p.tau_ex_syn_inv
        syn_tau_inv[self.exc_size:] = self.lif_p.tau_ih_syn_inv
        self.syn_tau_inv = syn_tau_inv
        self.tau_mem_inv = torch.ones(hidden_size, dtype=torch.double, device=device) * self.lif_p.tau_mem_inv

        # Base current initialization
        lower, upper = self.lif_p.current_base_lower, self.lif_p.current_base_upper
        mu, sigma = self.lif_p.current_base_mu, self.lif_p.current_base_sigma
        ei_base = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(
            (1, self.hidden_size))

        self.recurrent_base = torch.nn.Parameter(
            torch.tensor(ei_base,dtype=torch.double, device=self.device),requires_grad=False)

        # Readout mask
        read_out_m = np.ones(hidden_size)
        # mask for filtering out self-connected weights
        recurrent_weights_m = np.ones((hidden_size, hidden_size)) - np.diag(np.ones(hidden_size))

        # set up dale/no-dale mask
        if self.dale:
            # Dale rule
            dale_vec = np.ones(hidden_size)
            dale_vec[self.exc_size:] = -1 * self.ei_ratio / (1 - self.ei_ratio)
            self.recurrent_weights_dale = torch.tensor(np.float64(np.diag(dale_vec) / np.linalg.norm(
                np.matmul(np.ones((hidden_size, hidden_size)), np.diag(dale_vec)), axis=1)), dtype=torch.double,
                                                       device=device)
            read_out_m[self.exc_size:] = 0

        else:
            # rescale filter mask for the convenience in adjusting hyper-parameters
            no_dale_scale = 1 / self.hidden_size
            recurrent_weights_m = no_dale_scale * recurrent_weights_m

        self.recurrent_weights_mask = torch.tensor(recurrent_weights_m, dtype=torch.double, device=device)
        self.read_out_mask = torch.tensor(read_out_m, dtype=torch.double, device=device)

        # SFA mask
        sfa_m = np.ones(hidden_size)
        sfa_m[self.sfa_size:] = 0.
        np.random.shuffle(sfa_m)
        self.sfa_mask = torch.nn.Parameter(
            torch.tensor(sfa_m, dtype=torch.double, device=self.device),
            requires_grad=False)
        self.recurrent_batch_base = None
        self.batch_size = None

    def initial_state(self, batch_size, device, dtype=torch.double):
        self.batch_size = batch_size
        self.recurrent_batch_base = self.recurrent_base.repeat(self.batch_size, 1)

        return SFALIFState(LIFRefracState(
            lif=LIFState(
                z=torch.zeros(batch_size, self.hidden_size,
                              device=device, dtype=dtype),
                v=torch.tensor(np.random.uniform(self.lif_p.v_reset, self.lif_p.v_th, (batch_size, self.hidden_size,)),
                               device=device, dtype=dtype),
                i=torch.zeros((batch_size, self.hidden_size,),
                              device=device, dtype=dtype)
            ),
            rho=torch.zeros(batch_size, self.hidden_size,
                            device=device, dtype=dtype),
        ),
            sfa=torch.zeros(batch_size, self.hidden_size,
                            device=device, dtype=dtype),
        )

    def forward(
            self, input_tensor: torch.Tensor, state: SFALIFState
    ):
        # compute sfa updates
        new_sfa = self.decay_sfa * state.sfa + (1. - self.decay_sfa) * state.lif_rec_state.lif.z
        curr_v_th = self.lif_p.v_th + new_sfa * self.beta * self.sfa_mask

        # compute voltage updates
        # \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i*r) + \xi
        dv = self.dt * self.tau_mem_inv * \
            ((self.lif_p.v_leak - state.lif_rec_state.lif.v) + state.lif_rec_state.lif.i * self.lif_p.R)
        V_noise = torch.normal(0., self.rand_voltage_std, size=state.lif_rec_state.lif.v.size(),
                               dtype=torch.double, device=self.device)
        v_decayed = state.lif_rec_state.lif.v + dv + torch.sqrt(self.dt) * V_noise

        # compute current updates
        # \dot{i} &= -1/\tau_{\text{syn}} i
        di = -self.dt * self.syn_tau_inv * state.lif_rec_state.lif.i
        i_decayed = state.lif_rec_state.lif.i + di

        # compute new spikes
        z_new = threshold(v_decayed - curr_v_th,
                          self.lif_p.method, self.lif_p.alpha)

        # compute reset
        v_new = (1 - z_new) * v_decayed + z_new * self.lif_p.v_reset

        # compute current jumps
        s_input, c_input = torch.split(
            input_tensor.double(), split_size_or_sections=[4, 1], dim=1)

        # sensory input
        s_i = torch.matmul(s_input, self.sinput_weights)
        # cue input
        c_i = torch.matmul(c_input, self.cinput_weights)

        # recurrent input
        if self.dale:
            rec_i = torch.matmul(state.lif_rec_state.lif.z, torch.matmul((torch.abs(
                self.recurrent_weights) * self.recurrent_weights_mask),
                self.recurrent_weights_dale).T)
        else:
            rec_i = torch.matmul(state.lif_rec_state.lif.z,
                                 self.recurrent_weights * self.recurrent_weights_mask)

        # simulate base current
        I_noise = torch.normal(0., self.rand_current_std, size=state.lif_rec_state.lif.i.size(),
                               dtype=torch.double, device=self.device)
        self.recurrent_batch_base = self.alpha * self.recurrent_batch_base + I_noise

        i_new = (
            i_decayed
            + s_i
            + c_i
            + rec_i
            + self.recurrent_batch_base * self.current_base_scale
        )

        v_new, z_new, rho_new = compute_refractory_update(
            state.lif_rec_state, z_new, v_new, self.p)
        output_z = z_new * self.read_out_mask
        return output_z, SFALIFState(LIFRefracState(LIFState(z_new, v_new, i_new), rho_new), new_sfa)
