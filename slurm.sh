sinfo
scancel
squeue
scontrol show node rosetta | grep -E 'CPUTot|RealMemory|Gres|AllocTRES'
scontrol show node voyager | grep -E 'CPUTot|RealMemory|Gres|AllocTRES'

srun -A rahul -q rahul -p rahul -c 32 -w voyager --mem=32G --gres=gpu:2 --pty bash --login