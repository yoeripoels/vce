"""Computing the explanation alignment cost (eac).
"""

'''
make each evaluation a class instance. first ->
  1. create data start/end point from source process (synthetic.structure)
  2. return this data so it can be used to generate explanations
  3. from generated explanation, compute eac given the information known in this class object about
      --> compute_exp() and find_closest_explanation_preprocess() in explainmetric.py
'''

# also, re-implement plot_solution


def create_explanation_simple(all_lines, line_idx_a, line_idx_b, return_shape=True):
    pass
