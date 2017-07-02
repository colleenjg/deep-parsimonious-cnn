# Learning Deep Parsimonious Representations

This repository is an independent extension of the NIPS'16 paper:

Renjie Liao, Alexander Schwing, Richard S. Zemel, Raquel Urtasun. Learning Deep Parsimonious Representations. Neural Information Processing System, 2016. https://github.com/lrjconan/deep_parsimonious

The code here applies distillation to networks trained using the methods described in the paper above to generate smaller networks of comparable accuracy. Furhter, distillation is combined with clustering regularization as described in the paper to generate a "hybrid" training method.

## Using the code
### Train a model
  Determine model parameters and saving parameters in `exp_config.py`
  Run `run_train_model.py` passing `<exp_id>` (e.g., CIFAR10_baseline) as args.

### Train a distilled model
  Determine cumbersome model (baseline or clustering) to use in `exp_config.py`
  Determine distilled model parameters in `exp_config.py`
  Run `run_distill_model.py` passing `<distilled_model_id>` `<cumbersome_model_id>` and paramaters `<lambda>` `<temperature>` as args.

### Test a model
  Determine model to test in `exp_config.py`
  run `run_test_model.py` passing `<exp_id>` (e.g., CIFAR10_baseline) as args.

### Evaluate clustering
  Deterine model to test in `exp_config.py`
  run `eval_clustering.py` passing `<exp_id>` (e.g., CIFAR10_baseline) as args.

### Record raw and shuffled CIFAR10 data for t-SNE
  Determine directory to save in in `ecord_raw_data.py`
  run `record_raw_data.py` passing `<exp_id>` (e.g., CIFAR10_baseline) as args.

### Record activations for t-SNE
  Determine model to test in `exp_config.py`
  Determine directory to save in in `record_for_tsne.py`
  Uncomment a line if running on a model distilled from a clustered model.
  Run `record_for_tsne.py` passing `<exp_id>` (e.g., CIFAR10_baseline) as args.

### Run t-SNE and plot
  Determine filename and directories to save in (2x, values and plots) in `run_tsne_and_plot.py`
  Run `run_tsne_and_plot.py`

### Generate multipanelled plots
  Determine directories from which to get t-SNE dataFrames.
  Run `generate_plots.py`
  Manually save plot
