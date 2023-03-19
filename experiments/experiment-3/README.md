We ran TSNPE using the official code from https://github.com/mackelab/tsnpe_neurips/, with the simulator adjusted to return 15 summary statistics instead of 18, to enable comparison with our paper and with ([1]). Similar to [2], we ran TSNPE for 8 rounds with rejection sampling, and then for another 10 with sampling importance resampling (SIR). We kept all hyperparameters the same as in [2] except for the number of training samples. To facilitate comparison with SUNLE, we used 50,000 training samples in the first rounds of rejection sampling and SIR, and 10,000 samples in all other rounds.

[1] Variational methods for simulation-based inference, Glöckler et al.
[2] Truncated proposals for scalable and hassle-free simulation-based inference, Deistler et al.
