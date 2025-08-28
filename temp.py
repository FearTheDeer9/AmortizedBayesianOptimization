import pickle
import jax
import jax.numpy as jnp 
with open("experiments/joint-training/checkpoints/production_12hour/policy_phase_13.pkl", "rb") as f:
    data = pickle.load(f)
    print(data)