from ES import create_problem, s3801454_s3699463_ES
import numpy as np
from typing import List, Tuple
from smbo import SequentialModelBasedOptimization
import ConfigSpace
from tqdm import trange, tqdm
import time
from datetime import datetime

np.random.seed(42)
F18, _logger = create_problem(18)

def evaluate(mu: int, lambda_: int, crossover_prob:float) -> float:
    scores = []
    for _ in range(20):
        s = s3801454_s3699463_ES(F18, int(mu), int(lambda_), crossover_prob)
        scores.append(s)
        F18.reset()  
    _logger.close()  
    
    return sum(scores) / len(scores)

def sample_configurations(n_configurations: int):
    cs = ConfigSpace.ConfigurationSpace('ES', seed=42)

    mu = ConfigSpace.UniformIntegerHyperparameter(name='mu', lower=10, upper=200, default_value=15)
    lambda_ = ConfigSpace.UniformIntegerHyperparameter(name='lambda_', lower=2, upper=700, default_value=105)
    pc = ConfigSpace.UniformFloatHyperparameter(name='pc', lower=0.1, upper=1, log=True, default_value=0.5)
    
    cs.add_hyperparameters([mu, lambda_, pc])

    return np.array([(configuration['mu'],configuration['lambda_'], configuration['pc'])
                    for configuration in cs.sample_configuration(n_configurations)])

def sample_initial_configurations(n:int) -> List[Tuple[np.array, float]]:
    configs = sample_configurations(n)
    
    return [((mu, lambda_, pc), evaluate(mu, lambda_, pc)) for mu, lambda_, pc in tqdm(configs)]

if __name__ == '__main__':
    smbo = SequentialModelBasedOptimization(random_state=42)
    print('Initializing SMBO...')
    smbo.initialize(sample_initial_configurations(24))
    print('Initialized SMBO')

    for _ in trange(128, desc='SMBO Iteration'):
        smbo.fit_model()
        theta_new = smbo.select_configuration(sample_configurations(64))
        performance = evaluate(theta_new[0], theta_new[1], theta_new[2])
        smbo.update_runs((theta_new, performance))
        print(f'Current best: {smbo.theta_inc_performance}')


    print()
    print(f'Score: {smbo.theta_inc_performance}')
    mu, lambda_, pc = smbo.theta_inc
    print(f'Params: {(mu, lambda_, pc)}')
    print()
    print("Testing params")
    score = evaluate(mu, lambda_, round(pc, 3))
    print(f'Average Score: {score}')

    with open('smbo.csv', '+a') as file:
        file.write(f'{smbo.theta_inc_performance}, {(mu, lambda_, pc)}, {score}, {datetime.fromtimestamp(time.time())}\n')
