# Neural Networks Playbook

An interactive, single-page playbook that takes you from zero to expert in neural networks, in 13 levels. Every level has explanations, the maths, code, a self-check, and a small instrument to play with. Everything runs in the browser, with no build step and no backend.

**Live page:** https://aminghuf.github.io/neural-networks-playbook/ (once GitHub Pages is enabled; see below)

## What's inside

| Level | Topic | Interactive piece |
| --- | --- | --- |
| — | Live trainer | A network learning spirals, moons, circle or XOR in the browser, with "break it on purpose" presets |
| 0 | The big picture | — |
| 1 | Math toolkit | Matrix shape checker |
| 2 | The single neuron | Neuron sliders with a live activation curve |
| 3 | Multi-layer networks | XOR network you can switch on and off; parameter counter |
| 4 | Loss, gradient descent, backprop | Learning-rate explorer; line-by-line backprop stepper |
| 5 | NumPy from scratch | Full code (see `examples/mlp_numpy.py`) |
| 6 | Training in practice | Underfitting/overfitting diagnoser; debugging recipe |
| 7 | PyTorch | Training-loop template |
| 8 | CNNs | Live convolution grid you can draw on; layer-size calculator |
| 9 | RNNs, LSTMs, attention | Softmax temperature demo |
| 10 | Transformers and LLMs | Causal attention heatmap; model size and memory estimator |
| 11 | Expert frontier | LoRA calculator |
| 12 | Production / MLOps | Serving, GPUs on Kubernetes, monitoring |

It also has a project ladder, a cheat sheet and a 12-week study plan. Progress ticks are saved in your browser's `localStorage`.

## Run it

Open `index.html` in a browser. That's it.

## Examples

Standalone scripts from the playbook, in `examples/`:

| File | What it does | Verified |
| --- | --- | --- |
| `mlp_numpy.py` | MLP with backprop, Adam and a gradient check, trained on two spirals | Gradient check ~6e-11; 100% test accuracy |
| `backprop_worked_example.py` | The Level 4 worked example, with a numerical gradient check | Matches the playbook's numbers |
| `causal_attention.py` | Single-head causal self-attention in NumPy | Lower-triangular weights; rows sum to 1 |
| `pytorch_mnist.py` | PyTorch MLP training loop on MNIST | Not executed here (PyTorch unavailable); standard API |

```bash
pip install numpy
python examples/mlp_numpy.py
```

## Publishing on GitHub Pages

Settings → Pages → Build and deployment → Source: "Deploy from a branch", Branch: `main`, folder `/ (root)`.
