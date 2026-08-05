# GPU access paths

The project requires an NVIDIA CUDA GPU. FPGA instances and graphics-only hosts are not substitutes for the target benchmark.

## Preferred: Brown Oscar

The Brown exploratory account provides four CPU cores and two standard GPUs for 48 hours, but no persistent data directory. This project requests one GPU and four CPU cores so both jobs fit within that allocation. Submit the committed Slurm scripts:

```bash
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh
export HF_TOKEN="<read-only-token>"
sbatch scripts/slurm_baseline.sh
sbatch scripts/slurm_specdec.sh
```

The target 8B model plus 1B draft should be attempted first on a 24 GB Ampere GPU. The 3B draft may require more VRAM. Store the repository, Hugging Face cache, and raw results under `~/scratch`; copy important results off Oscar before the 48-hour exploratory allocation expires.

## Alternative: standalone Linux CUDA host

This path supports an approved Azure NVIDIA VM or another Ubuntu CUDA machine:

```bash
git clone https://github.com/JoJojo1256/AIinferenceproj.git
cd AIinferenceproj
bash env/setup_linux_gpu.sh
export HF_TOKEN="<read-only-token>"
bash scripts/run_gpu.sh baseline --workload qa
bash scripts/run_gpu.sh speculative --workload qa --speculation-length 4
```

The preflight exits before model download when CUDA is unavailable or the GPU has less than 20 GiB of VRAM.

## Azure requirements

Azure NP-series VMs are FPGA-backed and **cannot** run this CUDA/PyTorch project. Request an NVIDIA quota family instead:

- Preferred for comfortable headroom: one A100 VM, such as `Standard_NC24ads_A100_v4`.
- Lower-cost 24 GB option: one full A10 VM, such as `Standard_NV36ads_A10_v5`.

Before provisioning, confirm:

1. The intended use is permitted by the subscription owner.
2. You have Contributor access to a dedicated resource group.
3. The region has non-zero quota for the exact NVIDIA VM family.
4. The hourly price and shutdown/deallocation plan are approved.

Always use `az vm deallocate` when an Azure benchmark session ends; shutting down only inside Linux may continue billing allocated compute.
