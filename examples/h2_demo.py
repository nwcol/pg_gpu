"""
Copied from moments_integration_demo.py...

This is just a local example for debugging
"""

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import msprime
import demes
import demesdraw
import moments
import moments.LD
import sys
from collections import OrderedDict
from math import ceil

import h2py
import pg_gpu


# ── Settings ───────────────────────────────────────────────────

NUM_REPS = 100
SEQ_LEN = 5_000_000
MUT_RATE = 1.5e-8
REC_RATE = 1.5e-8
SAMPLE_SIZE = 3  # diploids per population
SAMPLE_POPS = ["deme0", "deme1", "deme2"]
DATA_DIR = "examples/data/moments_3pop_integration_demo"
R_BINS = np.logspace(-6, -3, 20)


# ── Model definition and simulation ────────────────────────────

def simulate_data(demographic_model, vcf_dir):
    """
    Simulate replicate regions with msprime and write VCFs, alongside
    samples-to-deme map and flat recombination map. Return paths to
    VCF files, recombination map, and samples file.
    """
    os.makedirs(vcf_dir, exist_ok=True)
    vcf_paths = [os.path.join(vcf_dir, f"rep_{i}.vcf.gz") for i in range(NUM_REPS)]
    map_path = os.path.join(vcf_dir, "flat_map.txt")
    samples_path = os.path.join(vcf_dir, "samples.txt")

    tree_sequences = msprime.sim_ancestry(
        {pop: SAMPLE_SIZE for pop in SAMPLE_POPS},
        demography=msprime.Demography.from_demes(demographic_model),
        sequence_length=SEQ_LEN,
        recombination_rate=REC_RATE,
        num_replicates=NUM_REPS,
        random_seed=1024,
    )
    for i, (ts, vcf) in enumerate(zip(tree_sequences, vcf_paths)):
        ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=i * 10 + 1)
        population_names = [pop.metadata["name"] for pop in ts.populations()]
        individual_populations = [population_names[ind.population] for ind in ts.individuals()]
        individual_names = [f"{pop}{i}" for i, pop in enumerate(individual_populations)]
        ts.write_vcf(gzip.open(vcf, "wt"), individual_names=individual_names, position_transform="legacy")

    # write samples file
    with open(samples_path, "w") as handle:
        handle.write("sample\tpop\n")
        for name, pop in zip(individual_names, individual_populations):
            handle.write(f"{name}\t{pop}\n")

    # write flat recombination map
    with open(map_path, "w") as handle:
        handle.write("pos\tMap(cM)\n")
        handle.write("0\t0\n")
        handle.write(f"{SEQ_LEN}\t{REC_RATE * SEQ_LEN * 100}\n")

    return vcf_paths, map_path, samples_path


generative_model = """
# YAML of the generative model to simulate and subsequently fit.
# See the demes docs for details on the specification:
# https://popsim-consortium.github.io/demes-docs/latest/introduction.html
time_units: generations
generation_time: 1
demes:
    - name: anc
      epochs:
        - {end_time: 15000.0, start_size: 10000.0}
    - name: trunk
      ancestors: [anc]
      epochs:
        - {end_time: 5000.0, start_size: 20000.0}
    - name: deme0
      ancestors: [anc]
      epochs:
        - {end_time: 0, start_size: 5000.0, end_size: 50000.0}
    - name: deme1
      ancestors: [trunk]
      epochs:
        - {end_time: 0, start_size: 20000.0}
    - name: deme2
      ancestors: [trunk]
      epochs:
        - {end_time: 0, start_size: 20000.0}
migrations:
    - {source: deme0, dest: deme2, rate: 0.0001}
"""


# ── Usage ──────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(DATA_DIR, exist_ok=True)

    true_yaml_path = os.path.join(DATA_DIR, "true_model.yaml")
    with open(true_yaml_path, "w") as handle:
        handle.write(generative_model)



    vcf_path = os.path.join(DATA_DIR, "data")
    vcf_paths, map_path, samples_path = simulate_data(demes.load(true_yaml_path), vcf_path)

    # ── pg_gpu drop-in replacement !!! ─────────────────────────────
    h2_sums = {
        vcf: pg_gpu.h2_statistics.compute_h2_statistics(
            vcf, rec_map_file=map_path, pop_file=samples_path,
            pops=SAMPLE_POPS, r_bins=R_BINS, report=True
        ) for vcf in vcf_paths
    }
    h2_stats = h2py.parsing.bootstrap_data(h2_sums)
    model = h2py.H2stats.from_demes(
        true_yaml_path,
        sampled_demes=SAMPLE_POPS,
        r_bins=R_BINS,
        u=MUT_RATE
    )
    h2py.plotting.plot_h2_curves_comp(
        model,
        h2_stats["means"],
        h2_stats["varcovs"],
        r_bins=h2_stats["bins"]
    )



