import numpy as np
import pymc3 as pm



def generate_smoothly_transformed_dataset(cltv_grid, pdf_grid, tau, size):
    transformed_pdf_grid = pdf_grid * np.exp(tau * cltv_grid)
    # pymc3 will normalise this distribution itself
    transformed_dist = pm.Interpolated.dist(cltv_grid, transformed_pdf_grid)
    scaling_factor = (((transformed_pdf_grid * cltv_grid).sum() / transformed_pdf_grid.sum())
                      / ((pdf_grid * cltv_grid).sum() / pdf_grid.sum()))
    return transformed_dist.random(size=size), scaling_factor



def generate_smoothly_transformed_datasets(cltv_grid, pdf_grid, taus, size):
    taus = np.ravel(taus)
    return [generate_smoothly_transformed_dataset(cltv_grid, pdf_grid, tau, size) for tau in taus]



def generate_scaled_then_shifted_dataset(ref_dist, scale, loc, size, ref_mean=None):
    try:
        reference_mean = ref_dist.mean()
    except AttributeError:
        assert ref_mean, "ref_mean must be provided if ref_dist doesn't provide a `mean` method"
        assert np.isreal(ref_mean), "ref_mean must be a real number"
        reference_mean = ref_mean
    scaling_factor = (reference_mean * scale + loc) / reference_mean
    return (ref_dist.random(size=size) * scale + loc, scaling_factor)



def generate_scaled_then_shifted_datasets(ref_dist, scales, locs, size, ref_mean=None):
    scales, locs = np.ravel(scales), np.ravel(locs)
    assert len(scales) == len(locs), "scales and locs must be the same length"
    return [generate_scaled_then_shifted_dataset(ref_dist, s, l, size, ref_mean) for s, l in zip(scales, locs)]



def generate_mixture_of_scaled_dataset(ref_dist, mix1_scale, mix2_scale, mix1_proportion, size, ref_mean=None):
    mix1_N = int(size * mix1_proportion)
    mix2_N = size - mix1_N
    mix1_dataset = generate_scaled_then_shifted_dataset(ref_dist, mix1_scale, loc=0.0, size=mix1_N, ref_mean=ref_mean)
    mix2_dataset = generate_scaled_then_shifted_dataset(ref_dist, mix2_scale, loc=0.0, size=mix2_N, ref_mean=ref_mean)
    scaling_factor = (mix1_proportion * mix1_scale) + ((1-mix1_proportion) * mix2_scale)
    return np.concatenate([mix1_dataset[0], mix2_dataset[0]]), scaling_factor
    
    
    
def generate_mixture_of_scaled_datasets(ref_dist, mix1_scales, mix2_scales, mix1_proportions, size, ref_mean=None):
    mix1_scales, mix2_scales, mix1_proportions = np.ravel(mix1_scales), np.ravel(mix2_scales), np.ravel(mix1_proportions)
    assert len(mix1_scales) == len(mix2_scales) == len(mix1_proportions), "inputs must be the same length"
    return [generate_mixture_of_scaled_dataset(ref_dist, m1s, m2s, m1p, size, ref_mean)
            for m1s, m2s, m1p in zip(mix1_scales, mix2_scales, mix1_proportions)]
