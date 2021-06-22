import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.functional.lif import lif_step, lif_feed_forward_step, lif_current_encoder, LIFParameters
from scipy.signal import chirp

RAND_SEED = 1


class DataGenerator:

    def __init__(self, batch_size, trial_setting, p, dt=0.001, cueing_context_num=1, 
                 context_num=2, sensory_num=2, if_base_line=False, mix_coh=1.0):

        self.p = p
        self.dt = dt
        self.batch_size = batch_size
        self.cc_num = cueing_context_num
        self.c_num = context_num
        self.s_num = sensory_num
        self.trial_num = self.cc_num * self.c_num * (self.s_num ** 2)
        self.trial_setting = trial_setting
        self.if_base_line=if_base_line
        self.mix_coh = mix_coh
        self.rng = RandomState(seed=RAND_SEED)

        self.seq_len = self.get_seq_len()
        self.trial_shape = [self.c_num * self.s_num + 1, self.seq_len]

        self.cue_start = int(self.trial_setting['fixation'] // self.dt)
        self.cue_time = int(self.trial_setting['cueing'] // self.dt)

        self.sensory_start = int(
            (self.trial_setting['fixation'] + self.trial_setting['cueing'] + self.trial_setting[
                'cueing_delay']) // self.dt)
        self.sensory_time = int(self.trial_setting['stimulus'] // self.dt)

        self.contexts = []
        self.sensory_intpus = {'v': {}, 'a': {}}
        self.sensory = []
        self.trials = []

    def get_seq_len(self):
        l = 0
        for key in self.trial_setting.keys():
            l += self.trial_setting[key]
        return int((l * 1000) // (self.dt * 1000))

    def generate_context(self):
        time_units = int(self.trial_setting['cueing'] // self.dt)

        self.contexts.append([])
        time_series = np.arange(0, time_units, 1)

        freq = .5
        centr = 3.
        scale = 1.

        # cueing context 0
        input_current00 = np.sin(
            np.pi/time_units * time_series)*(centr + scale * np.sin(freq * time_series))
        input_current01 = np.tile(np.sin(np.pi/time_units * time_series[:int(time_units//2)])*(
            centr + scale * np.sin(freq * time_series[:int(time_units//2)])), 2)[::-1]

        if self.mix_coh > 1.0 or self.mix_coh < 0.0:
            raise Exception("Invalid coherence value")


        if self.mix_coh < 1.0:
            spike0 = torch.flatten(self.generate_spike(self.mix_coh*input_current00+(1-self.mix_coh)*input_current01,time_units))
            spike1 = torch.flatten(self.generate_spike(self.mix_coh*input_current01+(1-self.mix_coh)*input_current00,time_units))
        else:
            spike0 = torch.flatten(self.generate_spike(input_current00,time_units))
            spike1 = torch.flatten(self.generate_spike(input_current01,time_units))


        self.contexts[0].append(spike0)
        self.contexts[0].append(spike1)

        if self.cc_num == 1:
            return

        self.contexts.append([])
        input_current10 = np.cos(
            np.pi/time_units/2 * time_series)*(centr + scale * np.cos(freq * time_series))
        input_current11 = np.tile(np.cos(np.pi/time_units/2 * time_series[:int(time_units//2)])*(
            centr + scale * np.cos(freq * time_series[:int(time_units//2)])), 2)
        
        if self.mix_coh:
            pass

        spike10 = torch.flatten(self.generate_spike(input_current10,time_units))
        spike11 = torch.flatten(self.generate_spike(input_current11,time_units))

        self.contexts[1].append(spike10)
        self.contexts[1].append(spike11)


    def generate_spike(self, input_current, T):
        spikes = []
        v = torch.zeros(1)

        for index, _ in enumerate(range(T)):
            z, v = lif_current_encoder(input_current[index], v, self.p)
            if z == 1.:
                spikes.append(1.)
            else:
                spikes.append(0)

        return torch.tensor(spikes)

    def generate_sensory(self):
        time_units = int(self.trial_setting['stimulus'] // self.dt)
        time_series = np.arange(0, time_units, 1)

        a_scale = 2.
        a_freq = 0.2
        centr1 = 2.

        a_fire = centr1+np.sin(a_freq * time_series)*a_scale
        self.sensory_intpus['a']['fire'] = torch.flatten(
            self.generate_spike(a_fire, time_units))
        a_regular = centr1+np.sin(a_freq * time_series+np.pi)*a_scale
        self.sensory_intpus['a']['regular'] = torch.flatten(
            self.generate_spike(a_regular, time_units))

        v_scale = 1.
        v_freq = 0.1
        centr2 = 2.5

        v_fire = centr2+np.cos(v_freq*2 * time_series)*v_scale
        self.sensory_intpus['v']['fire'] = torch.flatten(
            self.generate_spike(v_fire, time_units))
        v_regular = centr2 + np.cos(v_freq * time_series)*v_scale
        self.sensory_intpus['v']['regular'] = torch.flatten(
            self.generate_spike(v_regular, time_units))

    def generate_all_trials(self):
        self.generate_context()
        self.generate_sensory()

        for cc in range(self.cc_num):
            for c in range(self.c_num):
                for aud in range(2):
                    for light in range(2):
                        curr_trial = torch.zeros(*self.trial_shape)

                        # Target
                        if c == 0:
                            ground_truth = aud
                        else:
                            ground_truth = light

                        # Context Input
                        curr_trial[4, self.cue_start:self.cue_start +
                                   self.cue_time] = self.contexts[cc][c]

                        # Sensory Input
                        if aud == 0:
                            curr_trial[0, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['a']['fire']
                            curr_trial[1, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['a']['regular']
                        else:
                            curr_trial[0, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['a']['regular']
                            curr_trial[1, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['a']['fire']

                        if light == 0:
                            curr_trial[2, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['v']['fire']
                            curr_trial[3, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['v']['regular']
                        else:
                            curr_trial[2, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['v']['regular']
                            curr_trial[3, self.sensory_start:self.sensory_start +
                                       self.sensory_time] = self.sensory_intpus['v']['fire']

                        desc = {'cc': cc, 'c': c, 'aud': aud,
                                'light': light, 'truth': ground_truth}
                        self.trials.append(
                            (curr_trial.transpose(1, 0), ground_truth, desc))

    def get_batch(self, if_oneHot=True, if_rand=True):
        trial_batch = []
        target_batch = []
        desc_batch = []
        for index in range(self.batch_size):
            if if_rand:
                choice = self.rng.choice(self.trial_num)
            else:
                choice = int(index % self.trial_num)

            trial, tg, desc = self.trials[choice]
            trial_batch.append(torch.unsqueeze(trial, dim=0))
            if if_oneHot:
                tg_vector = np.zeros(2)
                tg_vector[tg] = 1.
                target_batch.append(tg_vector)
            else:
                target_batch.append(tg)
            desc_batch.append(desc)
        
        trial_batch = torch.cat(trial_batch)
        trial_batch = trial_batch.transpose(1, 0)
        if self.if_base_line: trial_batch = torch.zeros(trial_batch.size())
        target_batch = torch.tensor(target_batch)

        return trial_batch, target_batch, desc_batch

    def plot_trials(self):
        for index in range(len(self.trials)):
            width_in_inches = 12
            height_in_inches = 6
            dots_per_inch = 70
            time_series = np.arange(self.seq_len)
            fig, ax = plt.subplots(1, figsize=(
                width_in_inches, height_in_inches))

            inputs = self.trials[index][0].to_sparse().coalesce()
            t = inputs.indices()[0]
            n = inputs.indices()[1]
            ax.scatter(t, n, marker='|', color='black')
            ax.set_xlabel('Time [ms]')

            ax.set_yticks([0, 1, 2, 3, 4])
            ax.set_yticklabels(['Aud-Right', 'Aud-Left', 'Vis-Right', 'Vis-Left', 'Cueing'])
            plt.gca().axes.get_xaxis().set_visible(False)

            plt.autoscale(False)
            plt.title(self.trials[index][2])
            plt.savefig(str(index) + '.png')
            # plt.show()


## TEST ##
if __name__ == '__main__':
    trial_setting = {
        'fixation': 0.0,
        'cueing': 0.1,
        'cueing_delay': 0.1,
        'stimulus': 0.1,
        'decision': 0.00}

    dg = DataGenerator(8, trial_setting, LIFParameters(),
                       cueing_context_num=1, dt=0.002)
    dg.generate_all_trials()
    dg.plot_trials()

