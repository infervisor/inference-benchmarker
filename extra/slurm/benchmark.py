import os
import subprocess

CPUS_PER_GPU = 11
MEM_PER_CPU_GB = 11


def main():
    models = [
        ('meta-llama/Meta-Llama-3.1-8B-Instruct', 1),
        ('meta-llama/Meta-Llama-3.1-70B-Instruct', 4),
        ('mistralai/Mixtral-8x7B-Instruct-v0.1', 2),
    ]
    engines = ['tgi', 'vllm']
    for model in models:
        print(f"Submitting job for {model[0]}")
        gpus = model[1]
        cpus_per_task = gpus * CPUS_PER_GPU
        mem_per_cpu = gpus * MEM_PER_CPU_GB
        for engine in engines:
            job_name = f'bench_{engine}_{model[0].replace("/", "_")}'
            args = ['sbatch', '--cpus-per-task', str(cpus_per_task), '--mem-per-cpu', str(mem_per_cpu) + 'G', '--gpus',
                    str(gpus), '--nodes', '1',
                    '--job-name', job_name, f'{engine}.slurm']
            token = os.environ.get('HF_TOKEN', '')
            path = os.environ.get('PATH', '')
            process = subprocess.run(args, capture_output=True,
                                     env={'MODEL': model[0], 'HF_TOKEN': token, 'PATH': path})
            print(process.stdout.decode())
            print(process.stderr.decode())
            if process.returncode != 0:
                print(f'Error while submitting :: {args}')
                exit(1)


if __name__ == '__main__':
    main()
