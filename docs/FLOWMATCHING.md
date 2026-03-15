# Flow Matching

Flow Matching (FM) is a generative modeling paradigm that learns a continuous velocity field that transforms samples drawn from a simple source distribution (e.g., Gaussian noise) into samples from the target distribution (e.g., robot actions). Instead of learning to denoise noisy samples (as in diffusion models), flow matching directly regresses the velocity at each point in time that moves a sample along a probability path.

ref. https://arxiv.org/abs/2210.02747
