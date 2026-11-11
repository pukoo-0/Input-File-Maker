"""
jobscript.py — cluster submission scripts (SLURM or PBS) and plain run scripts.

make_script(cfg, commands, program) -> (filename, text, submit_command, warnings)
  cfg: scheduler, job_name, nodes, ntasks_per_node, cpus_per_task, time, partition, account,
       memory, modules (text, one per line), setup (extra lines), launcher ("mpirun" | "srun")
  commands: list of "PROGRAM ARGS" strings where $RUN marks an MPI launch
"""

DEFAULTS = dict(scheduler="slurm", job_name="dft", nodes=1, ntasks_per_node=4, cpus_per_task=1,
                time="24:00:00", partition="", account="", memory="", email="",
                modules="", setup="", launcher="mpirun")


def _valid_time(t):
    parts = str(t).split(":")
    return len(parts) == 3 and all(p.isdigit() for p in parts)


def make_script(cfg, commands, program):
    c = {**DEFAULTS, **cfg}
    warnings = []
    ntasks = int(c["nodes"]) * int(c["ntasks_per_node"])
    if not _valid_time(c["time"]):
        warnings.append("Write the time limit as HH:MM:SS, for example 24:00:00.")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(c["job_name"])) or "dft"
    lines = ["#!/bin/bash"]
    if c["scheduler"] == "slurm":
        lines += [f"#SBATCH --job-name={name}", f"#SBATCH --nodes={int(c['nodes'])}",
                  f"#SBATCH --ntasks-per-node={int(c['ntasks_per_node'])}",
                  f"#SBATCH --cpus-per-task={int(c['cpus_per_task'])}", f"#SBATCH --time={c['time']}",
                  f"#SBATCH --output={name}.%j.log"]
        if c["partition"]:
            lines.append(f"#SBATCH --partition={c['partition']}")
        if c["account"]:
            lines.append(f"#SBATCH --account={c['account']}")
        if c["memory"]:
            lines.append(f"#SBATCH --mem={c['memory']}")
        if c["email"]:
            lines += [f"#SBATCH --mail-user={c['email']}", "#SBATCH --mail-type=END,FAIL"]
        filename = f"{name}.slurm"
        submit = f"sbatch {filename}"
    elif c["scheduler"] == "pbs":
        lines += [f"#PBS -N {name}", f"#PBS -l nodes={int(c['nodes'])}:ppn={int(c['ntasks_per_node'])}",
                  f"#PBS -l walltime={c['time']}", "#PBS -j oe"]
        if c["partition"]:
            lines.append(f"#PBS -q {c['partition']}")
        if c["account"]:
            lines.append(f"#PBS -A {c['account']}")
        if c["memory"]:
            lines.append(f"#PBS -l mem={c['memory']}")
        if c["email"]:
            lines += [f"#PBS -M {c['email']}", "#PBS -m ae"]
        lines += ["", "cd $PBS_O_WORKDIR"]
        filename = f"{name}.pbs"
        submit = f"qsub {filename}"
    else:
        filename = f"run_{name}.sh"
        submit = f"bash {filename}"
    lines.append("")
    mods = [m.strip() for m in str(c["modules"]).splitlines() if m.strip()]
    if mods:
        lines += [m if m.startswith(("module", "source", "export", "conda", "spack")) else f"module load {m}"
                  for m in mods]
        lines.append("")
    if str(c["setup"]).strip():
        lines += [str(c["setup"]).strip(), ""]
    if int(c["cpus_per_task"]) > 1:
        lines.append(f"export OMP_NUM_THREADS={int(c['cpus_per_task'])}")
    else:
        lines.append("export OMP_NUM_THREADS=1")
    if program == "qe":
        if c["scheduler"] == "slurm" and c["launcher"] == "srun":
            lines.append('RUN="srun"')
        else:
            lines.append(f'RUN="mpirun -np {ntasks}"')
    lines.append("")
    lines += commands
    lines.append("")
    if program == "orca" and c["launcher"] == "srun":
        warnings.append("ORCA starts its own MPI processes, so it must not be launched with srun. The script "
                        "runs ORCA directly.")
    return filename, "\n".join(lines), submit, warnings
