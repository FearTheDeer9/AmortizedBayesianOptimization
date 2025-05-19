# Multi-Stage Training Process

Our training process involves multiple distinct stages to effectively train both the surrogate and acquisition models.

## Stage 1: Surrogate Model Pretraining

The surrogate model is pretrained to amortize inference across different causal structures:

1. **Data Collection**: Generate expert demonstrations from PARENT_SCALE
2. **Supervised Training**: Train the surrogate model to predict posteriors from observed data
3. **Evaluation**: Test the model's ability to generalize to unseen causal structures

### Training Loop

```python
def pretrain_surrogate(expert_demonstrations, learning_rate, num_epochs):
    surrogate_params = initialize_surrogate()
    
    for epoch in range(num_epochs):
        for batch in batched(expert_demonstrations):
            # Extract observations, interventions, and true posteriors
            observations = batch["observations"]
            interventions = batch["interventions"]
            true_posteriors = batch["posteriors"]
            
            # Create buffer from demonstrations
            buffer = create_buffer(observations, interventions)
            
            # Update surrogate parameters
            surrogate_params = update_surrogate(
                params=surrogate_params,
                buffer=buffer,
                learning_rate=learning_rate,
                target_posteriors=true_posteriors
            )
    
    return surrogate_params
```

## Stage 2: Acquisition Model Training with Verifiable Rewards

The acquisition model is trained using GRPO with verifiable rewards:

1. **Initial Setup**: Use the pretrained surrogate model
2. **GRPO Training**: Train the acquisition model using:
   - Structure discovery rewards (based on information gain)
   - Optimization rewards (based on target improvement)
   - Parent intervention rewards (based on ground truth during training)
3. **Imitation Learning**: Optionally incorporate expert demonstrations as additional training signal

### Training Loop

```python
def train_acquisition(surrogate_params, expert_demonstrations, learning_rate, num_steps):
    acquisition_params = initialize_acquisition()
    
    for step in range(num_steps):
        # Sample a random SCM and generate initial buffer
        scm, buffer = sample_problem()
        
        # Generate a group of candidate interventions
        state = get_current_state(buffer, surrogate_params)
        interventions = generate_group_interventions(state, acquisition_params)
        
        # Apply interventions and collect rewards
        rewards = []
        for intervention in interventions:
            # Apply intervention
            new_scm = apply_intervention(scm, intervention)
            outcome = sample_from_scm(new_scm)
            new_buffer = buffer.add_intervention(intervention, outcome)
            
            # Compute verifiable rewards
            reward = compute_verifiable_reward(
                state, intervention, outcome, new_buffer, surrogate_params
            )
            rewards.append(reward)
            
        # Update acquisition model using GRPO
        acquisition_params = update_acquisition_grpo(
            params=acquisition_params,
            state=state,
            interventions=interventions,
            rewards=rewards,
            learning_rate=learning_rate
        )
    
    return acquisition_params
```

## Stage 3: End-to-End Training

Finally, we train the complete system end-to-end:

1. **Initialize**: Start with pretrained surrogate and acquisition models
2. **Alternate Updates**: Alternate between updating:
   - Surrogate model based on new observations
   - Acquisition model based on verifiable rewards
3. **Progressive Difficulty**: Gradually increase problem complexity

### Training Loop

```python
def train_end_to_end(surrogate_params, acquisition_params, num_steps):
    for step in range(num_steps):
        # Sample a problem of appropriate difficulty
        difficulty = adjust_difficulty(step)
        scm, buffer = sample_problem(difficulty)
        
        # Update surrogate with current buffer
        surrogate_params = update_surrogate(
            params=surrogate_params,
            buffer=buffer,
            learning_rate=surrogate_lr
        )
        
        # Select intervention using acquisition model
        posterior = compute_posterior(buffer, surrogate_params)
        state = State(posterior=posterior, buffer=buffer)
        intervention = select_intervention(state, acquisition_params)
        
        # Apply intervention and observe outcome
        new_scm = apply_intervention(scm, intervention)
        outcome = sample_from_scm(new_scm)
        new_buffer = buffer.add_intervention(intervention, outcome)
        
        # Compute new posterior
        new_posterior = compute_posterior(new_buffer, surrogate_params)
        new_state = State(posterior=new_posterior, buffer=new_buffer)
        
        # Compute reward
        reward = compute_verifiable_reward(
            state, intervention, outcome, new_buffer, surrogate_params
        )
        
        # Update acquisition model
        acquisition_params = update_acquisition(
            params=acquisition_params,
            state=state,
            intervention=intervention,
            reward=reward,
            next_state=new_state,
            learning_rate=acquisition_lr
        )
    
    return surrogate_params, acquisition_params
```

## Deployment Process

During deployment, the system operates without expert guidance:

1. **Initial Observations**: Collect observational data
2. **Surrogate Inference**: Compute posterior over parent sets
3. **Acquisition Selection**: Select optimal intervention
4. **Intervention Execution**: Apply intervention and observe outcome
5. **Update**: Update buffer and repeat from step 2

This multi-stage training process ensures both components are properly trained for their respective tasks before being integrated into a complete system.