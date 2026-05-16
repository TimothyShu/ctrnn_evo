"""
Mutation operators — Milestone 2.

Operators to implement:
  - perturb_weights   : Gaussian perturbation of weight magnitudes
  - perturb_tau       : Gaussian perturbation with type-range clamping
  - perturb_bias      : Gaussian perturbation of biases
  - perturb_position  : Gaussian perturbation of spatial positions (clamped to [0,1]^2)
  - type_flip         : Change neuron type, re-clamp tau, optionally reinitialise weights
  - add_node          : Activate a free slot, initialise its fields
  - remove_node       : Mask off an active hidden slot
  - add_edge          : Set an entry in edge_mask and initialise weight
  - remove_edge       : Clear an entry in edge_mask

All operators must:
  - Return a genome that passes validate_genome()
  - Never mutate I/O neuron slots structurally
  - Be vmappable over a population batch
"""
