from pathlib import Path
from typing import Literal


def _resolve_slurm_file_name(sbatch_script_filepath: str, job_id: str, kind: Literal["output", "error"]) -> str:
    # must be an absolute path
    assert sbatch_script_filepath.startswith("/")

    with open(sbatch_script_filepath, "r") as f:
        sbatch_file_contents = f.readlines()

        # output file directive
        output_file_directives = [
            l for l in sbatch_file_contents if l.startswith(f"#SBATCH --{kind}")
        ]
        assert len(output_file_directives) == 1
        output_file_directive = output_file_directives[0]
        out_file = output_file_directive.split("=")[1].strip()

        # job name directive
        job_name_directives = [
            l for l in sbatch_file_contents if l.startswith("#SBATCH --job-name")
        ]
        assert len(job_name_directives) <= 1
        job_name = (
            job_name_directives[0]
            if job_name_directives
            else Path(sbatch_script_filepath).name
        )

    supported_slurm_replacement_symbols = ("%j", "%x")

    import re

    replacement_symbols = re.findall(r"%[a-zA-Z]", out_file)
    if set(replacement_symbols).difference(set(supported_slurm_replacement_symbols)):
        raise ValueError(
            f"slurm file name {out_file} contains unsupported replacement symbols: {replacement_symbols}"
        )
    else:
        resolved_out_file = out_file
        for symbol in supported_slurm_replacement_symbols:
            if symbol == "%x":
                resolved_out_file = resolved_out_file.replace(symbol, job_name)
            elif symbol == "%j":
                resolved_out_file = resolved_out_file.replace(symbol, job_id)
            else:
                raise ValueError
    return resolved_out_file
