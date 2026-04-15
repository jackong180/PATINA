import os
import shutil
from datetime import datetime

import yaml


def _serialize_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(v) for v in value]
    return str(value)


class ExperimentLogger:
    def __init__(self, config, args, project_root):
        self.config = config
        self.args = args
        self.project_root = os.path.abspath(project_root)

    def prepare(self):
        outputs_root = self._resolve_outputs_root()
        exp_name = self.args.exp_name

        if self.args.run_dir:
            run_dir = os.path.abspath(self.args.run_dir)
            run_id = os.path.basename(run_dir.rstrip(os.sep))
            is_new_run = not os.path.exists(run_dir)
        else:
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{self.config.SEED}"
            run_dir = os.path.join(outputs_root, exp_name, run_id)
            is_new_run = True

        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        logs_dir = os.path.join(run_dir, "logs")
        visualizations_dir = os.path.join(run_dir, "visualizations")
        src_backup_dir = os.path.join(run_dir, "src_backup")
        primary_config_dump_path = os.path.join(run_dir, "config.yaml")
        if is_new_run or not os.path.exists(primary_config_dump_path):
            config_dump_path = primary_config_dump_path
        else:
            mode_name = "train" if self.config.MODE == 1 else "test"
            config_dump_path = os.path.join(run_dir, f"config_{mode_name}.yaml")

        for path in [run_dir, checkpoints_dir, logs_dir, visualizations_dir, src_backup_dir]:
            os.makedirs(path, exist_ok=True)

        self.config.OUTPUTS_ROOT = outputs_root
        self.config.EXP_NAME = exp_name
        self.config.RUN_ID = run_id
        self.config.RUN_DIR = run_dir
        self.config.CHECKPOINTS_DIR = checkpoints_dir
        self.config.LOGS_DIR = logs_dir
        self.config.VISUALIZATIONS_DIR = visualizations_dir
        self.config.SRC_BACKUP_DIR = src_backup_dir
        self.config.CONFIG_DUMP_PATH = config_dump_path
        self.config.DEFAULT_RESULTS_DIR = visualizations_dir
        if getattr(self.config, 'RESULTS', None) in (None, ''):
            self.config.RESULTS = visualizations_dir
        elif self.config.MODE == 2 and getattr(self.config, 'RESUME_FROM', None):
            eval_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.config.RESULTS = os.path.join(os.path.abspath(self.config.RESULTS), eval_stamp)
            os.makedirs(self.config.RESULTS, exist_ok=True)

        if is_new_run and not self.args.skip_src_backup:
            self._backup_source()

        self._dump_config()
        print(f"Experiment outputs: {run_dir}")
        return self

    def _resolve_outputs_root(self):
        if os.path.isabs(self.args.outputs_dir):
            return self.args.outputs_dir
        return os.path.abspath(os.path.join(self.project_root, self.args.outputs_dir))

    def _backup_source(self):
        src_backup_dir = self.config.SRC_BACKUP_DIR
        src_dir = os.path.join(self.project_root, "src")
        src_backup_src_dir = os.path.join(src_backup_dir, "src")

        if os.path.isdir(src_dir) and not os.path.exists(src_backup_src_dir):
            shutil.copytree(
                src_dir,
                src_backup_src_dir,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )

        files_to_copy = [
            ("main.py", "main.py"),
            ("requirements.txt", "requirements.txt"),
            ("pyproject.toml", "pyproject.toml"),
            ("config.yml.example", "config.yml.example"),
        ]

        config_source = getattr(self.config, "CONFIG_PATH", None)
        if config_source:
            files_to_copy.append((config_source, "config_source.yml"))

        for source, dest_name in files_to_copy:
            source_path = source
            if not os.path.isabs(source_path):
                source_path = os.path.join(self.project_root, source_path)
            if os.path.exists(source_path):
                shutil.copy2(source_path, os.path.join(src_backup_dir, dest_name))

    def _dump_config(self):
        payload = self.config.to_dict()
        payload["ARGS"] = _serialize_value(vars(self.args))
        with open(self.config.CONFIG_DUMP_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
