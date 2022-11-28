python3 plots/plot_error_size.py --dataset MiniImageNet --no_meta_cp
python3 plots/plot_error_size.py --dataset FewRel --no_meta_cp
python3 plots/plot_error_size.py --dataset Heart --n_ways 2 --n_datasets 250 --n_shots 1500 --no_meta_cp
python3 plots/plot_error_size_per_cal_var_delta.py --dataset MiniImageNet
python3 plots/plot_error_size_per_cal_var_delta.py --dataset Heart --n_ways 2 --n_datasets 250 --n_shots 1500
