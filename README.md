# ultrasound-benchmark
Devito vs kwave benchmark

## Requirements

In order to run this benchmark [Devito](https://github.com/devitocodes/devito) and [kwave](https://www.kwve.com/) need to be installed with the proper CPU and GPU compiler. More specifically, this benchmark relies on a proper `gcc>8` or `icc` compiler for the CPU benchmark and `nvc` compiler and related librairies for the GPU benchmark. WE recommend the Nvidia [HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-213-downloads) version 21.3 that contains cuda `11` and cuda `10.2` (needed for kwave).

### Devito

Devito can be installed with `pip` viw `pip install -U devito` that will install the latest (v4.4) version od Devito.

### Kwave

For this benchmark, in order to make it fair, we rely on the C++ sources of Kwave rather thant the Matlab interface. The guide to download and compile these sources can be found at http://www.k-wave.org/documentation/example_cpp_running_simulations.php. 


## Run

Once devito and kwave installed, you will need to either alias the kwave compile executable to `kwavecpu` and `kwavegpu` or modify the script to replace these alias by the path to the executable. You can then run the script via `python bench.py` which will run all cases and save the result as a pandas dataframe.
