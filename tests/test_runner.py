"""Offline contract tests; uses a stub CLI only for subprocess orchestration."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpqa_runner", ROOT / "run_gpqa.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpqa portable ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "工具 with spaces"
        self.root.mkdir()
        shutil.copyfile(ROOT / "settings.json", self.root / "settings.json")
        shutil.copytree(ROOT / "prompts", self.root / "prompts")

    def settings(self, *arguments):
        args = runner.parser(self.root).parse_args(arguments)
        with patch.dict(os.environ, {}, clear=True):
            return runner.load_settings(args, self.root)

    def test_bundled_data_integrity(self):
        manifest = runner.read_json(ROOT / "data" / "manifest.json")
        for name, expected in manifest["files"].items():
            data = (ROOT / "data" / "gpqa" / name).read_bytes()
            self.assertEqual(len(data), expected["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])
        self.assertEqual(runner.validate_data(ROOT, runner.SUBSETS),
                         {"diamond": 198, "main": 448, "extended": 546})

    def test_defaults_and_full_config(self):
        settings = self.settings()
        path = runner.generate_configs(self.root, settings)
        cfg = runpy.run_path(str(path))
        self.assertEqual(cfg["models"][0]["max_out_len"], 512)
        self.assertEqual(cfg["models"][0]["type"], runner.MODEL_TYPE)
        self.assertEqual(cfg["datasets"][0]["name"], "gpqa_diamond.csv")
        self.assertEqual(cfg["datasets"][0]["path"], str(self.root / "data" / "gpqa"))
        self.assertEqual(cfg["summarizer"], {"attr": "accuracy"})
        standalone = runpy.run_path(str(self.root / "configs/models/vllm_api/vllm_api_general_chat.py"))
        self.assertEqual(standalone["models"], cfg["models"])

    def test_settings_cli_and_nested_override_precedence(self):
        override = self.root / "server.json"
        override.write_text(json.dumps({"model": {"host_port": 9000,
                           "generation_kwargs": {"temperature": 0.2, "top_p": 0.9}}}))
        settings = self.settings("--settings", str(override), "--host-port", "8989",
                                 "--temperature", "0.5", "--generation-kwargs", '{"temperature":0.3}',
                                 "--set", "generation_kwargs.temperature=0.6",
                                 "--set", "generation_kwargs.chat_template_kwargs.enable_thinking=false")
        self.assertEqual(settings["model"]["host_port"], 8989)
        self.assertEqual(settings["model"]["generation_kwargs"], {
            "temperature": 0.6, "ignore_eos": False, "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(runner.read_json(self.root / "settings.json")["model"]["host_port"], 8080)

    def test_environment_key_and_cli_priority(self):
        with patch.dict(os.environ, {"AIS_BENCH_API_KEY": "env-value"}):
            settings = runner.load_settings(runner.parser().parse_args([]), self.root)
            self.assertEqual(settings["model"]["api_key"], "env-value")
            settings = runner.load_settings(runner.parser().parse_args(["--api-key", "cli-value"]), self.root)
            self.assertEqual(settings["model"]["api_key"], "cli-value")

    def test_booleans_and_json_file(self):
        params = self.root / "generation.json"
        params.write_text('{"top_p": 0.95, "stop": ["END"]}', encoding="utf-8")
        settings = self.settings("--stream", "--no-ignore-eos", "--trust-remote-code",
                                 "--generation-kwargs", "@" + str(params))
        self.assertTrue(settings["model"]["stream"])
        self.assertTrue(settings["model"]["trust_remote_code"])
        self.assertFalse(settings["model"]["generation_kwargs"]["ignore_eos"])
        self.assertEqual(settings["model"]["generation_kwargs"]["stop"], ["END"])

    def test_url_normalization(self):
        for value in ("http://localhost:8989", "http://localhost:8989/v1/",
                      "http://localhost:8989/v1/chat/completions"):
            self.assertEqual(runner.normalize_url(value), "http://localhost:8989/")
        self.assertEqual(runner.normalize_url("https://host/proxy/v1"), "https://host/proxy/")
        self.assertEqual(runner.normalize_url("localhost:8989"), "http://localhost:8989/")
        self.assertEqual(runner.normalize_url("host/proxy/v1", enable_ssl=True), "https://host/proxy/")
        self.assertEqual(runner.normalize_url("http://[::1]:8989/v1"), "http://[::1]:8989/")
        with self.assertRaises(ValueError):
            runner.normalize_url("file:///tmp/api")

    def test_invalid_parameters_fail_before_generation(self):
        for arguments in (("--host-port", "0"), ("--batch-size", "-1"), ("--max-out-len", "0"),
                          ("--request-rate", "nan"), ("--num-prompts", "0"),
                          ("--set", "generation_kwargs=[]"), ("--set", "batch_size=true"),
                          ("--set", "generation_kwargs.ignore_eos=1"), ("--set", "attr=local"),
                          ("--set", "generation_kwargs.top_p=NaN"), ("--abbr", "../outside")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.settings(*arguments)
        self.assertFalse((self.root / "configs").exists())

    def test_all_subsets_and_str_prompt(self):
        settings = self.settings("--subset", "all", "--prompt", "str")
        datasets = runner.dataset_configs(self.root, settings["dataset"])
        self.assertEqual([d["name"] for d in datasets], [f"gpqa_{s}.csv" for s in runner.SUBSETS])
        self.assertIsInstance(datasets[0]["infer_cfg"]["prompt_template"]["template"], str)
        self.assertEqual(datasets[0]["eval_cfg"]["pred_postprocessor"]["options"], "ABCD")

    def test_missing_empty_and_corrupt_csv(self):
        with self.assertRaisesRegex(ValueError, "Missing bundled dataset"):
            runner.validate_data(self.root, ["diamond"])
        folder = self.root / "data" / "gpqa"
        folder.mkdir(parents=True)
        (folder / "gpqa_diamond.csv").write_text("bad,header\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid GPQA CSV header"):
            runner.validate_data(self.root, ["diamond"])
        header = (ROOT / "data/gpqa/gpqa_diamond.csv").read_text(encoding="utf-8").splitlines()[0]
        (folder / "gpqa_diamond.csv").write_text(header + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Empty GPQA dataset"):
            runner.validate_data(self.root, ["diamond"])
        (folder / "gpqa_diamond.csv").write_text(header + "\nshort,row\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid GPQA question"):
            runner.validate_data(self.root, ["diamond"])

    def test_model_strings_are_safe_python_literals(self):
        value = "model'\\path\nwith unicode 模型"
        settings = self.settings("--model", value)
        cfg = runpy.run_path(str(runner.generate_configs(self.root, settings)))
        self.assertEqual(cfg["models"][0]["model"], value)

    def test_moved_tool_runs_from_unrelated_directory(self):
        shutil.copyfile(ROOT / "run_gpqa.py", self.root / "run_gpqa.py")
        shutil.copytree(ROOT / "data", self.root / "data")
        result = subprocess.run([sys.executable, str(self.root / "run_gpqa.py"), "--generate-only",
                                 "--host-port", "8989", "--subset", "main", "diamond"],
                                cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = runpy.run_path(str(self.root / "configs/gpqa_benchmark.py"))
        for dataset in cfg["datasets"]:
            self.assertEqual(Path(dataset["path"]), self.root / "data/gpqa")
            self.assertTrue((Path(dataset["path"]) / dataset["name"]).is_file())
        self.assertEqual(cfg["models"][0]["host_port"], 8989)
        self.assertFalse((Path(self.temp.name) / "configs").exists())

    def test_cli_execution_cwd_passthrough_and_exit_code(self):
        shutil.copyfile(ROOT / "run_gpqa.py", self.root / "run_gpqa.py")
        shutil.copytree(ROOT / "data", self.root / "data")
        fake_install = Path(self.temp.name) / "fake site packages"
        package = fake_install / "ais_bench/benchmark/cli"
        package.mkdir(parents=True)
        for directory in (package, package.parent, package.parent.parent):
            (directory / "__init__.py").write_text("")
        (package / "main.py").write_text(
            "import json, os, pathlib, runpy, sys\n"
            "cfg = runpy.run_path(sys.argv[1])\n"
            "pathlib.Path('invocation.json').write_text(json.dumps({'cwd': os.getcwd(), "
            "'argv': sys.argv[1:], 'dataset': cfg['datasets'][0]['path']}))\n"
            "sys.exit(17)\n", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(fake_install)
        result = subprocess.run([sys.executable, str(self.root / "run_gpqa.py"), "--num-prompts", "2",
                                 "--", "--num-warmups", "0"], cwd=self.temp.name, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        observed = runner.read_json(self.root / "invocation.json")
        self.assertEqual(Path(observed["cwd"]), self.root)
        self.assertEqual(observed["argv"][-2:], ["--num-warmups", "0"])
        self.assertIn(str(self.root / "outputs"), observed["argv"])
        self.assertIn("--num-prompts", observed["argv"])
        self.assertFalse(list(fake_install.rglob("__pycache__")))

    def test_relative_input_files_follow_tool_not_cwd(self):
        shutil.copyfile(ROOT / "run_gpqa.py", self.root / "run_gpqa.py")
        shutil.copytree(ROOT / "data", self.root / "data")
        profiles = self.root / "profiles"
        profiles.mkdir()
        (profiles / "server.json").write_text('{"model":{"host_port":8989}}', encoding="utf-8")
        (profiles / "generation.json").write_text('{"top_p":0.85}', encoding="utf-8")
        (self.root / "tokenizer").mkdir()
        # Identical filenames in the caller's directory must not take precedence.
        other_profiles = Path(self.temp.name) / "profiles"
        other_profiles.mkdir()
        (other_profiles / "server.json").write_text('{"model":{"host_port":1111}}', encoding="utf-8")
        (other_profiles / "generation.json").write_text('{"top_p":0.1}', encoding="utf-8")
        result = subprocess.run([
            sys.executable, str(self.root / "run_gpqa.py"), "--generate-only",
            "--settings", "profiles/server.json", "--generation-kwargs", "@profiles/generation.json",
            "--path", "tokenizer",
        ], cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = runpy.run_path(str(self.root / "configs/gpqa_benchmark.py"))
        self.assertEqual(cfg["models"][0]["host_port"], 8989)
        self.assertEqual(cfg["models"][0]["generation_kwargs"]["top_p"], 0.85)
        self.assertEqual(Path(cfg["models"][0]["path"]), self.root / "tokenizer")

    def test_copy_with_old_configs_regenerates_all_local_paths(self):
        shutil.copyfile(ROOT / "run_gpqa.py", self.root / "run_gpqa.py")
        shutil.copytree(ROOT / "data", self.root / "data")
        (self.root / "tokenizer").mkdir()
        saved = runner.read_json(self.root / "settings.json")
        saved["model"]["path"] = "tokenizer"
        (self.root / "settings.json").write_text(json.dumps(saved), encoding="utf-8")
        runner.generate_configs(self.root, self.settings())
        second_root = Path(self.temp.name).resolve() / "再次复制 renamed tool"
        shutil.copytree(self.root, second_root)
        result = subprocess.run([sys.executable, str(second_root / "run_gpqa.py"), "--generate-only"],
                                cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for relative in ("configs/gpqa_benchmark.py", "configs/datasets/gpqa/gpqa_local.py",
                         "configs/models/vllm_api/vllm_api_general_chat.py"):
            cfg = runpy.run_path(str(second_root / relative))
            if "gpqa_datasets" in cfg:
                for dataset in cfg["gpqa_datasets"]:
                    self.assertEqual(Path(dataset["path"]), second_root / "data/gpqa")
                    self.assertTrue((Path(dataset["path"]) / dataset["name"]).is_file())
            if "models" in cfg:
                self.assertEqual(Path(cfg["models"][0]["path"]), second_root / "tokenizer")

    def test_missing_ais_bench_reports_actionable_error(self):
        with patch.object(runner, "ROOT", ROOT), patch.object(runner.importlib.util, "find_spec", return_value=None), \
                patch.object(runner, "generate_configs", return_value=ROOT / "configs/gpqa_benchmark.py"), \
                patch("builtins.print") as output:
            self.assertEqual(runner.main([]), 2)
            self.assertTrue(any("Activate the environment" in str(call) for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
