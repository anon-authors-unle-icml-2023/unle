import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn
import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple, List

from sbibm.tasks.task import Task
from sbibm.algorithms.sbi.utils import wrap_simulator_fn, SimulatorWrapper



from sbivibm.utils import wrap_posterior, wrap_prior, automatic_transform
from .filter import get_filter, build_classifier, train_classifier, init_classification_data, append_new_classification_data, set_surrogate_likelihood

import pickle
import os



def wrap_prior_gpu(p):
    return type(p)(low=p.base_dist.low.cuda(), high=p.base_dist.high.cuda())


class GPUSimulatorWrapper(SimulatorWrapper):
    def __init__(self, sim_wrapper):
        self.sim_wrapper = sim_wrapper

    def __call__(self, parameters, *args, **kwargs):
        x = self.sim_wrapper(parameters.cpu(), *args, **kwargs)
        return x.cuda()


def wrap_simulator_gpu(s):
    return GPUSimulatorWrapper(s)


class PotentialFn:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, theta):
        return - self.p.log_prob(theta, track_gradients=True)


def build_classifier(input_dim, hidden_dim=50):
    from torch import nn
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim).cuda(),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim,hidden_dim).cuda(),
        nn.Linear(hidden_dim, hidden_dim).cuda(),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim,hidden_dim).cuda(),
        nn.Linear(hidden_dim, hidden_dim).cuda(),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim,hidden_dim).cuda(),
        nn.Linear(hidden_dim, hidden_dim).cuda(),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim,hidden_dim).cuda(),
        nn.Linear(hidden_dim, 1).cuda(),
        nn.Sigmoid())


def train_classifier(classifier, data, epochs=10):
    from torch import optim, nn
    classifier.train()
    loss_fn = nn.BCELoss(reduce=False)
    optim = torch.optim.Adam(classifier.parameters())
    # class weight
    w = data.dataset.tensors[1].sum()/data.dataset.tensors[1].shape[0]
    weight = torch.tensor([w,1-w]).cuda()
    for i in range(epochs):
        for x,y in data:
            optim.zero_grad()
            y_pred = classifier(x)
            weight_ = weight[y.view(-1).long()].view_as(y)
            loss = torch.mean(loss_fn(y_pred.squeeze(), y)*weight_)
            loss.backward()
            optim.step()
        if (i % int(epochs/5)) == 0:
            print(loss.detach())
    classifier.eval()
    return classifier


# def init_classification_data(samples, y, batch_size=1000):
#     from torch.utils.data import TensorDataset, DataLoader
#     y = y.float()
#     data = TensorDataset(samples, y)
#     train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
#     return train_loader
# 
# 
# def append_new_classification_data(dataloader, samples, y, batch_size=1000):
#     from torch.utils.data import TensorDataset, DataLoader
#     y = y.float()
#     old_samples, old_y = dataloader.dataset.tensors
#     samples = torch.vstack([samples, old_samples])
#     y = torch.hstack([y, old_y])
#     data = TensorDataset(samples, y)
#     train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
#     return train_loader

def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_simulations_list: List[int]=None,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 1000,
    automatic_transforms_enabled: bool = True,
    mcmc_method: str = "slice_np_vectorized",
    mcmc_parameters: Dict[str, Any] = { "num_chains": 100, "thin": 10, "warmup_steps": 100, "init_strategy": "sir", "sir_batch_size": 1000, "sir_num_batches": 100, },
    show_progress_bars: bool = True,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    simulation_filter: str = "identity",
    cache_inf: Optional[str] = None,
    **kwargs,
) -> Tuple[list,torch.Tensor, int]:
    """Runs (S)NLE from `sbi` with handling of a simulation filter
Args:
    task: Task instance
    num_observation: Observation number to load, alternative to `observation`
    observation: Observation, alternative to `num_observation`
    num_samples: Number of samples to generate from posterior
    num_simulations: Simulation budget
    num_rounds: Number of rounds
    neural_net: Neural network to use, one of maf / mdn / made / nsf
    hidden_features: Number of hidden features in network
    simulation_batch_size: Batch size for simulator
    training_batch_size: Batch size for training network
    automatic_transforms_enabled: Whether to enable automatic transforms
    mcmc_method: MCMC method
    mcmc_parameters: MCMC parameters
    z_score_x: Whether to z-score x
    z_score_theta: Whether to z-score theta
Returns:
    Samples from posterior, number of simulator calls, log probability of true params if computable
"""
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)
    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    elif num_simulations_list is not None and num_simulations_list != "None":
        log.info(f"Running SNLE with non uniform simulations sizes")
        num_simulations_list = list(num_simulations_list)
        num_simulations_per_round = min(num_simulations_list)
        assert sum(num_simulations_list) == num_simulations
        assert len(num_simulations_list) == num_rounds
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)


    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
            
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

    simulator = task.get_simulator(max_calls=num_simulations, sim_type="sequential")
    # simulator = task.get_simulator(max_calls=num_simulations)

    sim_filter = get_filter(simulation_filter)
    if simulation_filter != "identity":
        theta = prior.sample((1,)).cuda()
        classifier = build_classifier(theta.shape[-1])


    transforms = automatic_transform(task)

    transforms.inv.base_transform.parts[1] = type(transforms.inv.base_transform.parts[1])(
        loc=transforms.inv.base_transform.parts[1].loc,
        scale=transforms.inv.base_transform.parts[1].scale,
    )

    if automatic_transforms_enabled:
        prior = wrap_prior(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    prior = wrap_prior_gpu(prior)
    simulator = wrap_simulator_gpu(simulator)

    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE(
        density_estimator=density_estimator_fun, prior=prior,
        device="gpu"
    )

    proposal = prior
    posteriors = []
    for r in range(num_rounds):

        if num_simulations_list is not None and num_simulations_list != "None":
            num_simulations_per_round = num_simulations_list[r]

        if task.name == "pyloric" and r==0:
            theta, x = task.get_precomputed_dataset()
            theta = theta[:num_simulations_per_round].cuda()
            x = x[:num_simulations_per_round].cuda()
        else:
            if r==0:
                theta = proposal.sample((num_simulations_per_round,))
            else:
                # import hamiltorch
                # hamiltorch.set_random_seed(123)
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # def make_posterior_log_prob(p):
                #     def posterior(theta):
                #         return p.log_prob(theta, track_gradients=True)
                #     return posterior
                # theta = hamiltorch.sample(
                #     log_prob_func=make_posterior_log_prob(proposal), params_init=prior.sample(),  num_samples=num_simulations_per_round, step_size=0.001, num_steps_per_sample=1
                # )
                # theta = torch.stack([t for t in theta]).cuda()
                from pyro.infer.mcmc import HMC, NUTS
                from pyro.infer.mcmc.api import MCMC

                kernel = HMC(
                    potential_fn=PotentialFn(proposal), trajectory_length=1
                )
                mcmc = MCMC(
                    kernel, num_samples=1000, 
                    num_chains=80,
                    initial_params={'theta': theta[idx][:80]},
                    # mp_context="spawn"
                )
                t = mcmc.run()

                __import__('pdb').set_trace()

                theta = proposal.sample((num_simulations_per_round,))[:num_simulations_per_round]
            log.info(f"Simulating {num_simulations_per_round} samples")
            x = simulator(theta)

        
        idx = sim_filter(theta, x, observation).cuda()
        log.info(f"Filtered out {idx.sum()} values")
        if simulation_filter != "identity":
            if r==0:
                classification_data = init_classification_data(theta, idx)
            else:
                classification_data = append_new_classification_data(classification_data, theta, idx)

            # XXX! 10> 200
            classifier = train_classifier(classifier, classification_data,epochs=10)

        if simulation_filter != "identity":
            density_estimator = inference_method.append_simulations(
            theta[idx], x[idx], from_round=r
            ).train(
                training_batch_size=training_batch_size,
                retrain_from_scratch_each_round=False,
                discard_prior_samples=False,
                show_train_summary=True,
                # max_num_epochs=4
            )
        else:
            density_estimator = inference_method.append_simulations(
                theta, x, from_round=r
            ).train(
                training_batch_size=training_batch_size,
                retrain_from_scratch_each_round=False,
                discard_prior_samples=False,
                show_train_summary=True,
                # max_num_epochs=4
            )


        posterior = inference_method.build_posterior(
            density_estimator, mcmc_method=mcmc_method, mcmc_parameters=mcmc_parameters
        )
        # Copy hyperparameters, e.g., mcmc_init_samples for "latest_sample" strategy.
        if r > 0:
            posterior.copy_hyperparameters_from(posteriors[-1])

        if simulation_filter != "identity":
            set_surrogate_likelihood(posterior, classifier)

        proposal = posterior.set_default_x(observation.cuda())
        posteriors.append(posterior)



        if cache_inf is not None:
            # Save inference object...
            inference_method._summary_writer = None
            inference_method._build_neural_net = None
            save = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            with open(save + os.sep + str(cache_inf) + ".pkl", "wb") as handle:
                print(save + os.sep + str(cache_inf) + ".pkl")
                pickle.dump(inference_method, handle)
            torch.save(posterior, save + os.sep + str(cache_inf) + "_posterior.pkl")
            if simulation_filter != "identity":
                torch.save(classifier, save + os.sep + str(cache_inf) + "_classifier.pkl")
            inference_method._summary_writer = inference_method._default_summary_writer()




    if automatic_transforms_enabled:
        for post in posteriors:
            post = wrap_posterior(post, transforms)


    samples = posteriors[-1].sample((num_samples,)).detach()[:num_samples]
    return posteriors, samples, num_simulations


            
            

		

