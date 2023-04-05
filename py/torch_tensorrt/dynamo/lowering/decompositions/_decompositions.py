from torch._inductor.decomposition import decompositions as inductor_decompositions

default_decompositions = inductor_decompositions.copy()
