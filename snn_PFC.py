import torch
from lif_refrac_cell import SNN_LIFCell
from norse.torch.module.leaky_integrator import LIFeedForwardCell
from norse.torch.functional.lif import LIFState
from norse.torch.functional.leaky_integrator import LIState
from parameters import pret_settings
from typing import NamedTuple


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class SNN(torch.nn.Module):
    def __init__(self, p, hidden_size, output_size=2, dt=0.001, ei_ratio=0.8, device='cpu'):
        super(SNN, self).__init__()

        self.sensory_size = 4
        self.context_size = 1
        self.ei_ratio = ei_ratio

        self.l1 = SNN_LIFCell(
            self.sensory_size,
            self.context_size,
            hidden_size,
            device,
            p=p,
            dt=dt,
        )

        self.fc_out = torch.nn.Linear(
            hidden_size, output_size, bias=False).double()
        self.out = LIFeedForwardCell(shape=(output_size,), dt=dt)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recording = None

    def forward(self, x, pret_setting:pret_settings):
        seq_length, batch_size, _ = x.shape
        s1 = self.l1.initial_state(batch_size, x.device, dtype=torch.double)
        so = self.out.initial_state(batch_size, x.device, dtype=torch.double)

        voltages = []
        self.recording = SNNState(
            LIFState(
                z=torch.zeros(seq_length, batch_size,
                              self.hidden_size, dtype=torch.double),
                v=torch.zeros(seq_length, batch_size,
                              self.hidden_size, dtype=torch.double),
                i=torch.zeros(seq_length, batch_size,
                              self.hidden_size, dtype=torch.double)
            ),
            LIState(
                v=torch.zeros(seq_length, batch_size,
                              self.output_size, dtype=torch.double),
                i=torch.zeros(seq_length, batch_size,
                              self.output_size, dtype=torch.double)
            )
        )
        w_rec = self.l1.recurrent_weights.data

        if_pret = pret_setting is not None and not self.training
        w_rec_scale = None

        if if_pret:
            # do customized perturbation work
            pass

        for ts in range(seq_length):
            inputs = x[ts, :, :]

            z, s1 = self.l1(inputs, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)

            self.recording.lif0.z[ts, :] = s1.lif_rec_state.lif.z
            self.recording.lif0.v[ts, :] = s1.lif_rec_state.lif.v
            self.recording.lif0.i[ts, :] = s1.lif_rec_state.lif.i
            self.recording.readout.v[ts, :] = so.v
            self.recording.readout.i[ts, :] = so.i

            voltages += [vo]

        return voltages
