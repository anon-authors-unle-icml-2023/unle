This repository contains the code to reproduce the results of the ICML 2023 submission: "Maximum Likelihood Learning of Unnormalized Models for Simulation-Based Inference"


Requirements:

- Conda
- Optionally (recommended) A GPU, along with cuda driver libraries.

To reproduce the experiments, please first set up an environment (which will be named `unle`) containing all the necessary dependencies by running `conda env create -f environment.yml` (for a GPU environment) or `conda env create -f environment-cpu.yml` for a CPU environment.
We strongly recommend that you use a GPU when running the experiments; using a GPU yields considerable speedups for training and inference. 
All the experiment submission / visualisation scripts take the form of a `jupyter` notebook. **No `jupyter` notebook engine is not provided** as part of the environment. You can either install 
`jupyter-notebook`/`jupyterlab` in this environment directly (by running the bash command `conda install -n unle jupyterlab`), or register the
`python` executable of the `unle` environment to an external `jupyterlab` engine. In the
latter case, the aforementioned CUDA environment variables need to be specified in the `share/jupyter/kernels/unle/kernel.json` (this file being relative to the jupyterlab environment root folder).
Here is an example `kernel.json` file that does so. You will need to change the placeholder paths indicated using </the/following/convention>:

```json
{
 "argv": [
  "</path/to/unle/env>/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "unle",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
   "PATH":"</path/to/unle/env>/bin:$PATH",
   "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
   "XLA_PYTHON_CLIENT_ALLOCATOR":"platform"
  }
}
```

The results used to plot the figure of the submission are provided in the following [google drive location](https://drive.google.com/drive/folders/1f3MCjNZUE5BhIYcEYc9U4rsHlPMyMQOJ?usp=sharing)
an external google drive location. To plot the results, either re-run the experiments, or download the contents present at the google drive link and 

- move `iclr_experiments_3` into `experiments/experiment-3/`.
