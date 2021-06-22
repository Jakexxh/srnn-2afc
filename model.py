import torch
from parameters import pret_settings

# def softmax_decode(x, loss_step):
#     x = torch.sum(torch.stack(x[-loss_step:]).permute(1,2,0),dim=2)
#     log_p_y = torch.nn.functional.log_softmax(x, dim=1)
#     return log_p_y


def msq_decode(x, loss_step):
    return torch.transpose(torch.stack(x[-loss_step:]), 1, 0, )


class Model(torch.nn.Module):
    def __init__(self, snn, loss_step, decoder='mse'):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = msq_decode
        if decoder == 'mse':
            self.decoder = msq_decode
        else:
            pass
        #     self.decoder = softmax_decode

        self.loss_step = loss_step

    def forward(self, x,pret_setting:pret_settings=None):
        x = self.snn(x,pret_setting)
        p_y = self.decoder(x, self.loss_step)
        return p_y
