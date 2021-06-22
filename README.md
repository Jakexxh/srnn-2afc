# Spiking recurrent neural networks represent task-relevant neural sequences in rule-dependent computation


## Requirements

This code is written in Python 3 and requires

* norse==0.03

This SRNN implementation was built upon the Python package Norse (https://github.com/norse/norse), based on PyTorch and deep learning-compatible spiking neural network components, including network architectures and optimization algorithms.

* matplotlib

* dataclasses

* tensorboard


## Examples

Example task specifications, including all hyper-parameters in the paper, can be found in `main.py` and `parameters.py`.

Training networks with default set-up

```
python main.py
```

Testing networks based on the checkpoint. The model loads the checkpoint file `snn_PFC-final.pt` in the dir `path_to_ckpt_dir`. The trained checkpoint is already shared in [Google Drive](https://drive.google.com/file/d/1uqPcKgggxp0ExwxK1j16OHgSwcPEvlcR/view?usp=sharing)

```
python main.py --only_test True --load True --load_path path_to_ckpt
```

If you want to save the time series data in the SRNN, including voltages, currents and spikes, please set parameter `save_recording` to True.


## Visualization 
### Running process
When training starts, a dir, named as the timestamp, will be generated automatically. Accuracy and loss metrics are saved inside as a tensorboard log file, please use Tensorboard to open them.

### Neural data analysis
Please check `ResultsVisualization.ipynb` and `results_visualization.py` for details.

## Note
The visulaliztion outputs vary for different models, but the results should be similar to those in the paper. If not, please adjust your parameters for training and also try to train multiple models to obatin the optimal results. 

## License
MIT

## Citation
* Xue, Xiaohe, Michael M. Halassa, and Zhe S. Chen. "Spiking recurrent neural networks represent task-relevant neural sequences in rule-dependent computation." bioRxiv (2021).
