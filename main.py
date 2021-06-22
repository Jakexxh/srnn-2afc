import datetime
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from norse.torch.functional.lif import LIFParameters, default_bio_parameters
from model import Model
from snn_PFC import SNN
from data_generator import DataGenerator
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from absl import flags
from absl import app
import os
import copy

from parameters import EILIFRefracParameters, EILIFParameters
from parameters import pret_settings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

FLAGS = flags.FLAGS


### Model Schema ###
flags.DEFINE_integer("cueing_context_num", 1, 
        "cueing context num, default is 1 (2 two-alternative forced choice)")
flags.DEFINE_float("dt", 0.002, "time step length (second)")
# network
flags.DEFINE_integer("hidden_size", 500, "hidden size for srnn.")
flags.DEFINE_bool("dale", True, "if use Dale's princple")
flags.DEFINE_float("ei_ratio", 0.8, "ei ratio (the ratio of excitatory neurons)")
flags.DEFINE_float("beta", 1.6, "beta for sfa updating")
flags.DEFINE_float("sfa_ratio", 0.25, "ratio of sfa neurons in srnn")
flags.DEFINE_float("rho", 1.5, "eescaling constant for W_rece")
# base current
flags.DEFINE_float("current_base_scale", 0.041, "scale value for base current init")
flags.DEFINE_float("current_base_lower", 1.8, "low threshold for base current init")
flags.DEFINE_float("current_base_upper", 2.4, "up threshold for base current init")
flags.DEFINE_float("current_base_mu", 2., "mean for base current init under normal distribution")
flags.DEFINE_float("current_base_sigma", 0.1, "sigma for base current init under normal distribution")
# noise
flags.DEFINE_float("rand_current_std", 0.002, "std for white noise in current")
flags.DEFINE_float("rand_voltage_std", 0.004, "std for white noise in voltage")
flags.DEFINE_float("rand_walk_alpha", 1., "updating constant for random walk noise")
# cell
flags.DEFINE_float("rho_reset", 3., "refractory time step")
flags.DEFINE_float("tau_ex_syn_inv", 1 / (35 * 1e-3), "inv tau of syn for excitatory cell")
flags.DEFINE_float("tau_ih_syn_inv", 1 / (40 * 1e-3), "inv tau of syn for inhibitory cell")
flags.DEFINE_float("tau_mem_inv", 1 / (20 * 1e-3), "inv tau for mem potential")
flags.DEFINE_float("tau_adaptation_inv", 1 / (400 * 1e-3), "inv tau for sfa")
flags.DEFINE_float("R", 10 * 1e-3, "syn resistance")
flags.DEFINE_float("v_leak", -65.0 * 1e-3, "voltage at leaking state")
flags.DEFINE_float("v_th", -50.0 * 1e-3, "spiking threshold")
flags.DEFINE_float("v_reset", -65.0 * 1e-3, "resting state")
# task
flags.DEFINE_float("fixation", 0.2, "fixation time")
flags.DEFINE_float("cueing", 0.1, "cueing time")
flags.DEFINE_float("cueing_delay", 0.4, "delay time after cueing.")
flags.DEFINE_float("stimulus", 0.1, "stimulus time")
flags.DEFINE_float("decision", 0., "extra decision time")


#### Training ####
flags.DEFINE_bool("train", True, "if train model")
flags.DEFINE_bool("early_stop", False, "if early stop while training")
flags.DEFINE_integer("epochs", 100, "num of training episodes to do.")
flags.DEFINE_integer("batch_num", 200, "num of batches in training to do.")
flags.DEFINE_integer("batch_size", 64, "num of examples in one minibatch.")
flags.DEFINE_enum(
    "optimizer", "adam", ["adam", "sgd"], "optimizer to use for training.")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate to use.")
flags.DEFINE_string("sg_gradient", "super", "surrogate gradient algorithm")
flags.DEFINE_float("alpha", 1000., "alpha for surrogate gradient")
# loss
flags.DEFINE_enum(
    "loss_fun", "mse", ["mse", "nll"], "loss function (nll is deprecated)")
flags.DEFINE_integer(
    "loss_step", 5, "the final steps for accumulating loss.")
# regularizations
flags.DEFINE_bool("l2_reg", False, "if add L2 regularization.")
flags.DEFINE_float("l2_reg_value", 1e-3, "scale consant for L2 reg")
flags.DEFINE_bool("fr_reg", True, "if add firing rate regularization.")
flags.DEFINE_float("fr_reg_value", 1e-9, "scale consant of fr reg")
# clipping while training
flags.DEFINE_bool("clip_grad", False, "if clip gradient during backpropagation")
flags.DEFINE_float("grad_clip_value", 1.5, "gradient to clip at.")
flags.DEFINE_bool("clip_weights", False, "if clip weights during backpropagation")
flags.DEFINE_float("weights_clip_value", 1.0, "weights to clip at.")
# log and checkpoint
flags.DEFINE_integer(
    "log_interval", 50, "in which intervals to display learning progress."
)
flags.DEFINE_integer("model_save_interval", 5,
                     "Save model every so many epochs.")
flags.DEFINE_boolean("save_model", True, "Save the model after training.")


#### Testing ####
flags.DEFINE_bool("only_test", False, "if just test model")


#### Saving & Loading ####
flags.DEFINE_bool("load", False, "if load checkpoint to model")
flags.DEFINE_string("load_path","","Load path")
flags.DEFINE_boolean("do_plot", True, "if do intermediate plots about neurons in tensorboard")
flags.DEFINE_bool("save_recording", False, "If save recordings of srnn")


def train(model, loss_fun, data_loader, epoch, optimizer, device, writer):
    model.train()
    losses = []
    step = FLAGS.batch_num * epoch

    for batch_i in range(FLAGS.batch_num):
        if FLAGS.loss_fun == "mse":
            trials, tgs, _ = data_loader.get_batch()
            tgs = tgs.unsqueeze(1).repeat((1, FLAGS.loss_step, 1))
            _, ground_truth = torch.max(tgs[:, -1], 1)
        else:
            trials, tgs, _ = data_loader.get_batch(if_oneHot=False)
            ground_truth = tgs

        trials, tgs, ground_truth = trials.to(
            device), tgs.to(device), ground_truth.to(device)

        optimizer.zero_grad()
        output = model(trials)
        loss = loss_fun(output, tgs)

        if FLAGS.l2_reg:
            wrec_w = model.snn.l1.recurrent_weights
            l2_loss = FLAGS.l2_reg_value / (FLAGS.hidden_size ** 2) * torch.nn.functional.mse_loss(
                wrec_w, target=torch.zeros_like(wrec_w), reduction='sum')
            loss += l2_loss.to(device)

        if FLAGS.fr_reg:
            bin_size = 5
            fr = torch.mean(model.snn.recording.lif0.z.permute(2, 0, 1),
                            dim=-1) 
            fr = torch.sum(torch.reshape(torch.mean(fr, axis=0),
                                          (-1, bin_size)), axis=1) / (FLAGS.dt * bin_size)
            fr_loss = FLAGS.fr_reg_value * torch.nn.functional.mse_loss(
                fr, target=torch.zeros_like(fr), reduction='sum')
            loss += fr_loss.to(device)

        loss.backward()

        if FLAGS.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), FLAGS.grad_clip_value)
        optimizer.step()

        if FLAGS.clip_weights:
            w_rec = model.snn.l1.recurrent_weights.data
            w_rec = w_rec.clamp(-FLAGS.weights_clip_value,
                                FLAGS.weights_clip_value)
            model.snn.l1.recurrent_weights.data = w_rec

        step += 1

        if batch_i % FLAGS.log_interval == 0:
            logging.info(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    FLAGS.epochs,
                    batch_i + FLAGS.log_interval,
                    FLAGS.batch_num,
                    100.0 * (batch_i + FLAGS.log_interval) / FLAGS.batch_num,
                    loss.item(),
                )
            )
            if FLAGS.loss_fun == "mse":
                pred = torch.argmax(output[:, -1], 1)
            else:
                pred = torch.argmax(output, 1)

            accuracy = (ground_truth == pred.squeeze()).float().mean()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, loss_fun, data_loader, epoch, device, writer,
         test_size=10, record_size=10, path_prefix="", pret_setting=None):

    model.eval()
    test_loss = []
    correct = 0
    step = FLAGS.batch_num * (1 + epoch)
    with torch.no_grad():

        for test_step in range(test_size):
            if FLAGS.loss_fun == "mse":
                trials, tgs, _ = data_loader.get_batch()
                tgs = tgs.unsqueeze(1).repeat((1, FLAGS.loss_step, 1))
                _, ground_truth = torch.max(tgs[:, -1], 1)
            else:
                trials, tgs, _ = data_loader.get_batch(if_oneHot=False)
                ground_truth = tgs

            trials, tgs, ground_truth = trials.to(device), tgs.to(device), ground_truth.to(device)

            output = model(trials, pret_setting)
            test_loss.append(loss_fun(output, tgs).item())
            if FLAGS.loss_fun == "mse":
                pred = torch.argmax(output[:, -1], 1)
            else:
                pred = torch.argmax(output, 1)

            correct += pred.eq(ground_truth.view_as(pred)).sum().item()

        if FLAGS.do_plot:
            if FLAGS.loss_fun == "mse":
                trials, tgs, _ = data_loader.get_batch(if_rand=False)
                tgs = tgs.unsqueeze(1).repeat((1, FLAGS.loss_step, 1))
            else:
                trials, tgs, _ = data_loader.get_batch(
                    if_oneHot=False, if_rand=False)

            trials, tgs = trials.to(device), tgs.to(device)
            _ = model(trials, pret_setting)

            width_in_inches = 12
            height_in_inches = 10

            fig, ax = plt.subplots(5, figsize=(
                width_in_inches, height_in_inches))

            ax[0].plot(model.snn.recording.lif0.z[:, 0,
                       :].squeeze(1).detach().cpu().numpy())
            ax[1].plot(model.snn.recording.lif0.v[:, 0,
                       :].squeeze(1).detach().cpu().numpy())
            ax[2].plot(model.snn.recording.lif0.i[:, 0,
                       :].squeeze(1).detach().cpu().numpy())

            ax[3].plot(model.snn.recording.readout.v[:, 0,
                       :].squeeze(1).detach().cpu().numpy())
            ax[4].plot(model.snn.recording.readout.i[:, 0,
                       :].squeeze(1).detach().cpu().numpy())

            ax[0].set_title('lif0.z')
            ax[1].set_title('lif0.v')
            ax[2].set_title('lif0.i')

            ax[3].set_title('readout.v')
            ax[4].set_title('readout.i')
            writer.add_figure("Voltages/output", fig, step)

        if FLAGS.save_recording:
            test_desc_list = []
            test_trials_list = []
            test_targets_list = []
            test_preds_list = []

            test_lif0_z_list = []
            test_lif0_v_list = []
            test_lif0_i_list = []

            test_readout_v_list = []
            test_readout_i_list = []

            for _ in range(record_size):
                if FLAGS.loss_fun == "mse":
                    trials, tgs, desc = data_loader.get_batch(if_rand=False)
                    tgs = tgs.unsqueeze(1).repeat((1, FLAGS.loss_step, 1))
                    _, ground_truth = torch.max(tgs[:, -1], 1)
                else:
                    trials, tgs, desc = data_loader.get_batch(
                        if_oneHot=False, if_rand=False)
                    ground_truth = tgs

                trials, tgs, ground_truth = trials.to(
                    device), tgs.to(device), ground_truth.to(device)
                output = model(trials, pret_setting)

                if FLAGS.loss_fun == "mse":
                    pred = torch.argmax(output[:, -1], 1)
                else:
                    pred = torch.argmax(output, 1)

                test_trials_list.append(trials.cpu().detach().numpy())
                test_targets_list.append(ground_truth.cpu().detach().numpy())
                test_desc_list.append(desc)
                test_preds_list.append(pred.cpu().detach().numpy())

                test_lif0_z_list.append(
                    model.snn.recording.lif0.z.cpu().detach().numpy())
                test_lif0_v_list.append(
                    model.snn.recording.lif0.v.cpu().detach().numpy())
                test_lif0_i_list.append(
                    model.snn.recording.lif0.i.cpu().detach().numpy())

                test_readout_v_list.append(
                    model.snn.recording.readout.v.cpu().detach().numpy())
                test_readout_i_list.append(
                    model.snn.recording.readout.i.cpu().detach().numpy())

            np.save(path_prefix + "test_trials.npy",
                    np.hstack(test_trials_list))
            np.save(path_prefix + "test_targets.npy",
                    np.hstack(test_targets_list))
            np.save(path_prefix + "test_preds.npy",
                    np.hstack(test_preds_list))

            np.save(path_prefix + "test_lif0_z.npy",
                    np.hstack(test_lif0_z_list))
            np.save(path_prefix + "test_lif0_v.npy",
                    np.hstack(test_lif0_v_list))
            np.save(path_prefix + "test_lif0_i.npy",
                    np.hstack(test_lif0_i_list))
            np.save(path_prefix + "test_readout_v.npy",
                    np.hstack(test_readout_v_list))
            np.save(path_prefix + "test_readout_i.npy",
                    np.hstack(test_readout_i_list))

            with open(path_prefix + 'test_desc.json', 'w') as fout:
                json.dump(
                    ([val for sublist in test_desc_list for val in sublist]), fout)

    test_loss = np.mean(test_loss)
    data_size = test_size * FLAGS.batch_size
    accuracy = 100.0 * correct / data_size
    logging.info(
        f"\nTest set: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{data_size} ({accuracy:.0f}%)\n"
    )

    if writer:
        writer.add_scalar("Loss/test", test_loss, step)
        writer.add_scalar("Accuracy/test", accuracy, step)

        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            writer.add_histogram(tag, value.data.cpu().numpy(), step)
            if value.grad is not None:
                writer.add_histogram(
                    tag + "/grad", value.grad.data.cpu().numpy(), step)

    return test_loss, accuracy


def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
        },
        path,
    )


def load(path, model, optimizer):
    if torch.cuda.is_available():
        def map_location(storage, loc):
            return storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(path + "/snn_PFC-final.pt",
                            map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint['epoch']


def main(argv):
    trial_setting = {
        'fixation': FLAGS.fixation,
        'cueing': FLAGS.cueing,
        'cueing_delay': FLAGS.cueing_delay,
        'stimulus': FLAGS.stimulus,
        'decision': FLAGS.decision}

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if FLAGS.load:
        path = FLAGS.load_path
    else:
        path = f"snn_PFC_log/{date}"

    if not FLAGS.load:
        os.makedirs(path, exist_ok=True)

    p_json = flags.FLAGS.flags_by_module_dict()
    p_json = {k: {v.name: v.value for v in vs} for k, vs in p_json.items()}

    os.chdir(path)
    if not os.path.exists('parameters.json'):
        with open('parameters.json', 'w') as fout:
            json.dump(p_json, fout)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        writer = SummaryWriter()
    except ImportError:
        writer = None

    data_loader = DataGenerator(FLAGS.batch_size, trial_setting=trial_setting, p=LIFParameters(
    ), cueing_context_num=FLAGS.cueing_context_num, dt=FLAGS.dt)
    data_loader.generate_all_trials()

    model_parameters = EILIFRefracParameters(
        lif=EILIFParameters(
            tau_ex_syn_inv=torch.as_tensor(FLAGS.tau_ex_syn_inv, dtype=torch.double),
            tau_ih_syn_inv=torch.as_tensor(FLAGS.tau_ih_syn_inv, dtype=torch.double),
            tau_mem_inv=torch.as_tensor(FLAGS.tau_mem_inv, dtype=torch.double),
            tau_adaptation_inv=torch.as_tensor(FLAGS.tau_adaptation_inv, dtype=torch.double),
            R=torch.as_tensor(FLAGS.R, dtype=torch.double),
            v_leak=torch.as_tensor(FLAGS.v_leak, dtype=torch.double),
            v_th=torch.as_tensor(FLAGS.v_th, dtype=torch.double),
            v_reset=torch.as_tensor(FLAGS.v_reset, dtype=torch.double),

            dale=FLAGS.dale,
            ei_ratio=FLAGS.ei_ratio,
            beta=FLAGS.beta,
            sfa_ratio=FLAGS.sfa_ratio,
            rho=FLAGS.rho,
            current_base_scale=FLAGS.current_base_scale,
            current_base_lower=FLAGS.current_base_lower,
            current_base_upper=FLAGS.current_base_upper,
            current_base_mu=FLAGS.current_base_mu,
            current_base_sigma=FLAGS.current_base_sigma,
            rand_current_std=FLAGS.rand_current_std,
            rand_voltage_std=FLAGS.rand_voltage_std,
            rand_walk_alpha=FLAGS.rand_walk_alpha,

            method=FLAGS.sg_gradient,
            alpha=FLAGS.alpha,
        ),

        rho_reset=torch.as_tensor(FLAGS.rho_reset),

    )

    model = Model(
        snn=SNN(
            p=model_parameters,
            hidden_size=FLAGS.hidden_size,
            output_size=2,
            dt=FLAGS.dt,
            device=device
        ),
        decoder=FLAGS.loss_fun,
        loss_step=FLAGS.loss_step
    )

    optimizer = None
    if FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=FLAGS.learning_rate)

    loss_fun = None
    if FLAGS.loss_fun == "mse":
        loss_fun = torch.nn.functional.mse_loss
    elif FLAGS.loss_fun == "nll":
        loss_fun = torch.nn.functional.nll_loss

    model.to(device)

    epoch = 0
    if FLAGS.load:
        model, optimizer, epoch = load(path, model, optimizer)

    if FLAGS.only_test:
        print('Testing model')
        test(model, loss_fun, data_loader , epoch + 1, device, writer, record_size=20, path_prefix='base_')
        w_rec = model.snn.l1.recurrent_weights.data
        np.save("wrec.npy", w_rec.cpu())

        return

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    accuracy = 0.
    max_accuracy = 0.
    early_stopping = EarlyStopping(patience=10)

    for _ in range(FLAGS.epochs):
        training_loss, mean_loss = train(
            model, loss_fun, data_loader, epoch, optimizer, device, writer
        )
        test_loss, accuracy = test(
            model, loss_fun, data_loader, epoch, device, writer)

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        max_accuracy = np.max(np.array(accuracies))

        if FLAGS.early_stop and early_stopping.step(test_loss):
            epoch += 1
            break

        if (epoch % FLAGS.model_save_interval == 0) and FLAGS.save_model:
            model_path = f"snn_PFC-{epoch}.pt"
            save(
                model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=accuracy > max_accuracy,
            )

        epoch += 1

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "snn_PFC-final.pt"
    save(
        model_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        is_best=accuracy > max_accuracy,
    )
    if writer:
        writer.close()


if __name__ == '__main__':
    app.run(main)
