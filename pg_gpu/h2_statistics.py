"""
GPU-accelerated H₂ statistics for h2py inference.

Provides a drop-in replacement for ``h2py.parsing.compute_h2_statistics()``.
The output format is identical to h2py, which itself mimics moments.LD.
"""

import cupy as cp
import numpy as np

from .accessible import AccessibleMask, resolve_accessible_mask
from .genotype_kernels import _PopDataGeno, _GenoPopFlat
from .genotype_matrix import GenotypeMatrix
from .haplotype_kernels import _launch, _HapPopFlat
from .haplotype_matrix import HaplotypeMatrix
from .ld_pipeline import (
    estimate_ld_chunk_size as _estimate_ld_chunk_size,
    iter_pairs_within_distance as _iter_pairs_within_distance,
    compute_counts_for_pairs as _compute_counts_for_pairs,
    compute_genotype_counts_for_pairs as _compute_genotype_counts_for_pairs,
    estimate_ld_chunk_size as _estimate_ld_chunk_size,
    het_names as _het_names,
    PopData as _PopData,
)
from .moments_ld import (
    _interpolate_genetic_distances,
    _max_bp_for_r_dist,
    _compute_heterozygosity
)


def compute_h2_statistics(
    vcf_file=None,
    pop_file=None,
    pops=None,
    pop_assignment=None,
    bed_file=None,
    chromosome=None,
    interval=None,
    rec_map_file=None,
    r_bins=None,
    bp_bins=None,
    min_bp=None,
    mut_map_file=None,  # TODO
    u_bar=None,     # TODO
    use_haplotypes=False,
    report=True,
    compute_denoms=True,
    stats_to_compute=None,
    ac_filter=True,
    genotype_matrix=None,
    haplotype_matrix=None,
):
    """GPU-accelerated drop-in for ``h2py.parsing.compute_h2_statistics()``.

    Accepts same arguments...

    Parameters
    ----------
    vcf_file : str, optional
    bed_file : str, optional
        Path to BED file defining accessible regions. Inacessible sites are
        ignored when computing statistics. Required if ``compute_denoms`` is
        ``True``.
    chromosome : str, optional
    rec_map_file : str, optional
        Path to recombination map file: tab-delimited with cols Pos, Map(cM).
    pop_file, pop_assignment :
    r_bins : array-like, optional
    bp_bins : array-like, optional
    interval : tuple, optional
    use_genotypes : bool, default True
    stats_to_compute : list, optional
    compute_denoms : bool, default True
    ac_filter : bool, default True
    report : bool, default True
    haplotype_matrix : HaplotypeMatrix, optional
    genotype_matrix : GenotypeMatrix, optional

    Returns
    -------
    sums : dict
        Keys 'bins', 'sums', 'stats', 'pops', 'denoms' (h2py format).

    Usage
    -----
        from pg_gpu.h2 import compute_h2_statistics
        h2_stats = compute_ld_statistics(
            vcf_file="data.vcf.gz",
            bed_file="accessible.bed.gz",
            rec_map_file="rec_map.txt",
            pop_file="pops.txt",
            pops=["popA", "popB"],
            r_bins=[0, 1e-6, 2e-6, 5e-6],
        )
    """
    # Flip `use_haplotypes`
    use_genotypes = True
    if use_haplotypes:
        use_genotypes = False

    if pops is None:
        pops = ['pop0', 'pop1']
    num_pops = len(pops)
    if num_pops < 1 or num_pops > 4:
        raise ValueError("1-4 populations supported")
    if r_bins is None and bp_bins is None:
        raise ValueError("Either r_bins or bp_bins must be provided")

    if use_genotypes:
        # Genotype (diploid) path
        if genotype_matrix is not None:
            gm = genotype_matrix
            if gm.device != 'GPU':
                gm.transfer_to_gpu()
        elif haplotype_matrix is not None:
            gm = GenotypeMatrix.from_haplotype_matrix(haplotype_matrix)
            if gm.device != 'GPU':
                gm.transfer_to_gpu()
        else:
            if vcf_file is None:
                raise ValueError("vcf_file or genotype_matrix required")
            if pop_file is None:
                raise ValueError("pop_file is required when loading from VCF")
            if report:
                print(f"Loading {vcf_file} (genotypes) ...")
            gm = GenotypeMatrix.from_vcf(vcf_file)
            gm.load_pop_file(pop_file, pops=pops)
            if ac_filter:
                gm = gm.apply_biallelic_filter()
            if bed_file is not None or interval is not None:
                mask = _get_mask_with_interval(bed_file, interval, hm.chrom_end)
                gm.set_accessible_mask(mask)
            gm.transfer_to_gpu()
        mat = gm
        if report:
            print(f"  {gm.num_individuals} individuals,"
                  f"{gm.num_variants:,} variants")
    else:
        # Haplotype (phased) path
        if haplotype_matrix is not None:
            hm = haplotype_matrix
            if not isinstance(hm.haplotypes, cp.ndarray):
                hm.transfer_to_gpu()
        else:
            if vcf_file is None:
                raise ValueError("vcf_file or haplotype_matrix is required")
            if pop_file is None:
                raise ValueError("pop_file is required when loading from VCF")
            if report:
                print(f"Loading {vcf_file} ...")
            hm = HaplotypeMatrix.from_vcf(vcf_file)
            hm.load_pop_file(pop_file, pops=pops)
            if ac_filter:
                hm = hm.apply_biallelic_filter()
            if bed_file is not None or interval is not None:
                mask = _get_mask_with_interval(bed_file, interval, hm.chrom_end)
                hm.set_accessible_mask(mask)
            hm.transfer_to_gpu()
        mat = hm
        if report:
            print(f"  {hm.num_haplotypes} hap, {hm.num_variants:,} variants")

    # Determine bins and distance metric for pair binning
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file required with r_bins")
        bins = np.asarray(r_bins, dtype=np.float64)
        if hasattr(mat.positions, 'get'):
            pos_cpu = mat.positions.get()
        else:
            pos_cpu = np.asarray(mat.positions)
        gen_dists = _interpolate_genetic_distances(pos_cpu, rec_map_file)
        gen_dists_gpu = cp.asarray(gen_dists)
        max_bp_dist = _max_bp_for_r_dist(pos_cpu, gen_dists, float(bins[-1]))
    else:
        bins = np.asarray(bp_bins, dtype=np.float64)
        gen_dists_gpu = None
        max_bp_dist = float(bins[-1])

    n_bins = len(bins) - 1
    if report:
        print(f"  Computing H2 ({n_bins} bins, {num_pops} pops) ...")

    h2_stat_names = _h2_names(num_pops)
    het_stat_names = _het_names(num_pops)

    h2_sums = _compute_h2_sums(mat, pops, bins, gen_dists_gpu, max_bp_dist,
                               use_genotypes=use_genotypes)
    het = _compute_heterozygosity(mat, pops, use_genotypes=use_genotypes)

    if report:
        print("  Done computing H2 statistics.")

    if compute_denoms:
        all_pos = _get_accessible_bed_positions(
            bed_file, interval, mat.chrom_end)
        pos_gpu = cp.asarray(all_pos)
        n_pos = len(all_pos)
        if r_bins is not None:
            coords = _interpolate_genetic_distances(all_pos, rec_map_file)
            coords_gpu = cp.asarray(coords)
        else:
            coords_gpu = cp.asarray(all_pos)
        if report:
            print(f"  Computing denominators ({n_pos} pos) ...")
        bins_gpu = cp.asarray(bins)
        denoms = _compute_h2_denoms(coords_gpu, bins_gpu, max_bp_dist, pos_gpu)
        if report:
            print(f"  Done computing denominators.")
    else:
        denoms = None

    if report:
        print("  Done.")

    bin_tuples = [(float(bins[i]), float(bins[i + 1])) for i in range(n_bins)]
    sums_list = [h2_sums[i] for i in range(n_bins)]
    sums_list.append(np.array([het[h] for h in het_stat_names]))

    sums = {'bins': bin_tuples, 'sums': sums_list, 'denoms': denoms,
            'stats': (h2_stat_names, het_stat_names), 'pops': pops}
    return sums


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _compute_h2_sums(
    mat,
    pops,
    bins,
    gen_dists_gpu,
    max_bp_dist,
    use_genotypes=True
    ):
    """Compute H₂ statistic sums in bins on GPU.

    Handles both haplotype and genotype modes.
    """
    # TODO
    num_pops = len(pops)

    if use_genotypes:
        geno = mat.genotypes
        xp = cp if isinstance(geno, cp.ndarray) else np
        alt_sum = xp.zeros(mat.num_variants, dtype=xp.int64)
        n_valid_filter = xp.zeros(mat.num_variants, dtype=xp.int64)
        seen = set()
        for pop in pops:
            for idx in mat.sample_sets[pop]:
                if idx in seen:
                    continue
                seen.add(idx)
                row = geno[idx, :]
                v = row >= 0
                alt_sum += xp.where(v, row, 0).astype(xp.int64)
                n_valid_filter += v.astype(xp.int64)
        max_alt = 2 * n_valid_filter
        keep = (alt_sum > 0) & (alt_sum < max_alt) & (n_valid_filter >= 2)
        keep_idx = xp.where(keep)[0]
        pos = mat.positions[keep_idx]
        data_matrix = mat.genotypes[:, keep_idx]
        count_fn = _compute_genotype_counts_for_pairs
        stat_fn = None
    else:
        pos = mat.positions
        data_matrix = mat.haplotypes
        count_fn = _compute_counts_for_pairs
        stat_fn = None

    if not isinstance(pos, cp.ndarray):
        pos = cp.array(pos)

    n_bins = len(bins) - 1
    bins_gpu = cp.asarray(bins)
    pop_indices = [mat.sample_sets[p] for p in pops]
    max_samp = max(len(pi) for pi in pop_indices)
    chunk_size = _estimate_ld_chunk_size(max_samp, num_pops=num_pops)

    h2_stat_names = _h2_names(num_pops)
    n_h2 = len(h2_stat_names)

    # TODO what is this?
    if use_genotypes:
        gen_dists_lookup = gen_dists_gpu[keep_idx]
    else:
        gen_dists_lookup = gen_dists_gpu

    bin_sums = cp.zeros((n_bins, n_h2), dtype=cp.float64)
    stat_specs = _generate_h2_stat_specs(num_pops)

    for ci, cj in _iter_pairs_within_distance(pos, max_bp_dist, chunk_size):
        if gen_dists_lookup is not None:
            distances = cp.abs(gen_dists_lookup[cj] - gen_dists_lookup[ci])
        else:
            distances = pos[cj] - pos[ci]
        # cb = cp.digitize(distances, bins_gpu) - 1
        cb = cp.searchsorted(bins_gpu, distances) - 1
        del distances

        counts_list = []
        n_valid_list = []
        for pidx in pop_indices:
            c, nv = count_fn(data_matrix, ci, cj, pidx)
            counts_list.append(c)
            n_valid_list.append(nv)

        if use_genotypes:
            stats = compute_multi_pop_h2_geno(
                counts_list, n_valid_list, stat_specs)
        else:
            stats = compute_multi_pop_h2_hap(
                counts_list, n_valid_list, stat_specs)

        valid = (cb >= 0) & (cb < n_bins)
        vb = cb[valid]
        vs = stats[valid]
        flat_idx = vb[:, None] * n_h2 + cp.arange(n_h2)[None, :]
        cp.add.at(bin_sums.ravel(), flat_idx.ravel(), vs.ravel())

        del counts_list, n_valid_list, stats, cb

    # Return `bin_sums` to host memory
    return bin_sums.get()


def _estimate_h2_chunk_size(max_samp, num_pops=None):
    # TODO
    return


def _compute_h2_denoms(coords, bins, max_bp_dist, pos):
    """Compute denominators for LD statistics on GPU. These are counts of
    pairs of accessible sites, binned by recombination distance.
    """
    denoms = cp.zeros(len(bins), dtype=np.float64)
    # Inclusive indices of the first right locus to be counted in bin 0, for
    # each left locus in `coords`.
    lower_indices = cp.maximum(cp.searchsorted(coords, coords + bins[0]),
                               cp.arange(1, len(coords) + 1))
    for ii, bin_end in enumerate(bins[1:]):
        # Non-inclusive indices of the last right locus to be counted in bin
        # `ii`, for each left locus in `coords`.
        upper_indices = cp.minimum(cp.searchsorted(coords, coords + bin_end),
                                   cp.searchsorted(pos, pos + max_bp_dist))
        denoms[ii] = cp.sum(upper_indices - lower_indices)
        lower_indices = upper_indices
    # The last element of `denoms` is the numerator of H
    denoms[-1] = len(coords)
    # Return to memory
    return denoms.get()


def _get_accessible_bed_positions(bed_file, interval, chrom_end):
    """Generate an array of accessible positions within a genomic interval."""
    mask = _get_mask_with_interval(bed_file, interval, chrom_end)
    return _get_accessible_positions(mask)


def _get_mask_with_interval(bed_file, interval, chrom_end):
    """Intersect accessible BED intervals with an overriding ``interval``, if
    both are given.

    If only one of ``bed_file`` or ``interval`` is given, get a mask
    accounting for it.`
    """
    if bed_file is not None:
        chrom_start = 1
        mask = resolve_accessible_mask(bed_file, chrom_start, chrom_end).mask
    else:
        mask = AccessibleMask(np.ones(chrom_end, dtype=bool))
    if interval is not None:
        mask[:interval[0]] = False
        mask[interval[1]:] = False
    accessible_mask = resolve_accessible_mask(mask, None, None)
    return accessible_mask


def _get_accessible_positions(accessible_mask):
    """Get a 1-indexed array of accessible positions."""
    return np.where(accessible_mask.mask)[0] + accessible_mask.offset + 1


def _generate_h2_stat_specs(num_pops):
    """Generate a list of tuples ('h2', (i, j)), where i, j index pops."""
    specs = []
    names = _h2_names(num_pops)
    for name in names:
        parts = name.split("_")
        pop_nums = tuple(int(p) for p in parts[1:])
        specs.append(("h2", pop_nums))
    return specs


def _h2_names(num_pops):
    """Generate H₂ statistic names."""
    names = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            names.append(f"H2_{ii}_{jj}")
    return names


# -----------------------------------------------------------------------------
# CUDA kernels for H2.
# -----------------------------------------------------------------------------


_H2_WITHIN_HAP_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1, const double*c2, const double*c3, const double*c4,
       const double*nn, const int*I, double*out, const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t];
    double a=c1[i],b=c2[i],c=c3[i],d=c4[i],n=nn[i];
    double num=2.*a*d+2*b*c;
    double den=n*(n-1.);
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_H2_BETWEEN_HAP_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*c1, const double*c2, const double*c3, const double*c4,
       const double*nn, const int*I, const int*J, double*out, const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x; if(t>=M)return;
    int i=I[t],j=J[t];
    double ai=c1[i],bi=c2[i],ci=c3[i],di=c4[i],ni=nn[i];
    double aj=c1[j],bj=c2[j],cj=c3[j],dj=c4[j],nj=nn[j];
    double num=ai*dj+aj*di+bi*cj+bj*ci;
    double den=ni*nj;
    out[t]=(den>0.)?num/den:0.;
}''', "k", options=("-std=c++11",))

_H2_WITHIN_GENO_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*g,const double*nn,const int* I,double*out,const int M){
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=M)return;
    int i=I[t];
    const double*row=g+9*i;
    double n5=row[4],n=nn[i];
    if(n<1.0){out[t]=0.;return;}
    // This estimator is merely the frequency of double heterozygotes
    out[t]=n5/n;
}''', "k", options=("-std=c++11",)) 

_H2_BETWEEN_GENO_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const double*X11,const double*X10,const double*X01,
       const double*nn,const int*I,const int*J,double*out,const int N){
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=N)return;
    int i=I[t],j=J[t];
    double ni=nn[i],nj=nn[j];
    double X11i=X11[i],X10i=X10[i],X01i=X01[i];
    double X11j=X11[j],X10j=X10[j],X01j=X01[j];
    double X00i=ni-X11i-X10i-X01i;
    double X00j=nj-X11j-X10j-X01j;
    double num=X11i*X00j+X11j*X00i+X10i*X01j+X10j*X01i;
    double den=ni*nj;
    out[t]=(den>0.0)?num/den:0.0;
}''', "k", options=("-std=c++11",))


# -----------------------------------------------------------------------------
#  Batch dispatch functions
# -----------------------------------------------------------------------------


def compute_multi_pop_h2_hap(counts_per_pop, n_valid_per_pop, stat_specs):
    """Compute all specified H₂ statistics using haplotype counts."""
    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)
    pops = [_PopData(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]
    n_pops = len(pops)
    F = _HapPopFlat(pops)
    h2_calls = [pidx for stat, pidx in stat_specs if stat == "h2"]
    within_calls = [(idx, c) for idx, c in enumerate(h2_calls) if c[0] == c[1]]
    between_calls = [(idx, c) for idx, c in enumerate(h2_calls) if c[0] != c[1]]

    pair_range = cp.arange(n_pairs, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * n_pairs + pair_range[None, :]).ravel()

    result = cp.zeros((n_pairs, n_stats), dtype=cp.float64)

    if within_calls:
        idxs = [i for i, _ in within_calls]
        calls = [c for _, c in within_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        M = len(calls) * n_pairs
        out = cp.empty(M, dtype=cp.float64)
        args = (F.c1, F.c2, F.c3, F.c4, F.n, fI, out, M)
        _launch(_H2_WITHIN_HAP_KERN, args, M)
        for fi, res_idx in enumerate(idxs):
            result[:, res_idx] = out[fi * n_pairs:(fi + 1) * n_pairs]

    if between_calls:
        idxs = [i for i, _ in between_calls]
        calls = [c for _, c in between_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
        M = len(calls) * n_pairs
        out = cp.empty(M, dtype=cp.float64)
        args = (F.c1, F.c2, F.c3, F.c4, F.n, fI, fJ, out, M)
        _launch(_H2_BETWEEN_HAP_KERN, args, M)
        for fi, res_idx in enumerate(idxs):
            result[:, res_idx] = out[fi * n_pairs:(fi + 1) * n_pairs]

    return result


def compute_multi_pop_h2_geno(counts_per_pop, n_valid_per_pop, stat_specs):
    """Compute all specified H₂ statistics using genotype counts."""
    n_pairs = counts_per_pop[0].shape[0]
    n_stats = len(stat_specs)
    pops = [_PopDataGeno(counts_per_pop[p], n_valid_per_pop[p])
            for p in range(len(counts_per_pop))]
    F = _GenoPopFlat(pops)
    h2_calls = [pidx for stat, pidx in stat_specs if stat == "h2"]
    within_calls = [(idx, c) for idx, c in enumerate(h2_calls) if c[0] == c[1]]
    between_calls = [(idx, c) for idx, c in enumerate(h2_calls) if c[0] != c[1]]

    pair_range = cp.arange(n_pairs, dtype=cp.int32)

    def expand(pop_arr):
        return (pop_arr[:, None] * n_pairs + pair_range[None, :]).ravel()

    result = cp.zeros((n_pairs, n_stats), dtype=cp.float64)

    if within_calls:
        idxs = [i for i, _ in within_calls]
        calls = [c for _, c in between_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        M = len(calls) * n_pairs
        out = cp.empty(M, dtype=cp.float64)
        args = (F.g_moments, F.n, fI, out, M)
        _launch(_H2_WITHIN_GENO_KERN, args, M)
        for fi, res_idx in enumerate(idxs):
            result[:, res_idx] = out[fi * n_pairs:(fi + 1) * n_pairs]

    # Between-population H2 exploits quantities precomputed for pi2_ijij
    if between_calls:
        idxs = [i for i, _ in between_calls]
        calls = [c for _, c in between_calls]
        fI = expand(cp.array([c[0] for c in calls], dtype=cp.int32))
        fJ = expand(cp.array([c[1] for c in calls], dtype=cp.int32))
        M = len(calls) * n_pairs
        out = cp.empty(M, dtype=cp.float64)
        args = (F.X11, F.X10, F.X01, F.n, fI, fJ, out, M)
        _launch(_H2_BETWEEN_GENO_KERN, args, M)
        for fi, res_idx in enumerate(idxs):
            result[:, res_idx] = out[fi * n_pairs:(fi + 1) * n_pairs]

    return result

