Traceback (most recent call last):
  File "/apps/python/3.9.4/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/apps/python/3.9.4/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/nh_run_data_dir.py", line 194, in <module>
    _main()
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/nh_run_data_dir.py", line 46, in _main
    start_run(config_file=Path(args["config_file"]), gpu=args["gpu"], data_dir=args["data_dir"])
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/nh_run_data_dir.py", line 86, in start_run
    start_training(config)
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/training/train.py", line 16, in start_training
    trainer = BaseTrainer(cfg=cfg)
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/training/basetrainer.py", line 63, in __init__
    self._create_folder_structure()
  File "/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/neuralhydrology/training/basetrainer.py", line 386, in _create_folder_structure
    raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")
RuntimeError: There is already a folder at /datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/workflows/camelaus_lstm/petrichore/runs/spatial_twofold_0_e_3001_124642
