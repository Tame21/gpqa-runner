"""GSM8K data integrity, dataset selection and portable CLI orchestration."""

import hashlib
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
sys.path.insert(0, str(ROOT))
import run_gpqa as runner


class GSM8KTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gsm8k portable ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "工具 with spaces"
        self.root.mkdir()
        for name in ("settings.json", "run_gpqa.py", "acc_len.py"):
            shutil.copyfile(ROOT / name, self.root / name)
        shutil.copytree(ROOT / "prompts", self.root / "prompts")

    def settings(self, *arguments):
        with patch.dict(os.environ, {}, clear=True):
            return runner.load_settings(runner.parser(self.root).parse_args(arguments), self.root)

    def test_bundled_data_integrity(self):
        folder = ROOT / "data/gsm8k"
        manifest = runner.read_json(folder / "manifest.json")
        for name, expected in manifest["files"].items():
            data = (folder / name).read_bytes()
            self.assertEqual(len(data), expected["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected["sha256"])
            if "records" in expected:
                self.assertEqual(len(data.splitlines()), expected["records"])
        self.assertEqual(manifest["files"]["train.jsonl"]["records"], 7473)
        self.assertEqual(manifest["files"]["test.jsonl"]["records"], 1319)
        self.assertEqual(runner.validate_data(ROOT, ["test"], "gsm8k"), {"test": 1319})

    def test_dataset_switching_and_legacy_defaults(self):
        original = (self.root / "settings.json").read_bytes()
        self.assertEqual(self.settings()["dataset"]["subsets"], ["diamond"])
        for extra in ((), ("--subset", "test"), ("--subset", "all")):
            self.assertEqual(self.settings("--dataset", "gsm8k", *extra)["dataset"],
                             {"name": "gsm8k", "subsets": ["test"], "prompt": "cot"})
        self.assertEqual((self.root / "settings.json").read_bytes(), original)
        saved = json.loads(original)
        del saved["dataset"]["name"]
        (self.root / "settings.json").write_text(json.dumps(saved), encoding="utf-8")
        self.assertEqual(self.settings()["dataset"]["name"], "gpqa")

    def test_settings_file_switch_resets_only_inherited_subsets(self):
        override = self.root / "gsm8k.json"
        override.write_text('{"dataset":{"name":"gsm8k"}}', encoding="utf-8")
        self.assertEqual(self.settings("--settings", "gsm8k.json")["dataset"]["subsets"], ["test"])
        self.assertEqual(self.settings("--settings", "gsm8k.json", "--dataset", "gpqa")
                         ["dataset"]["subsets"], ["diamond"])
        self.assertEqual(self.settings("--settings", "gsm8k.json", "--dataset", "gpqa", "--subset", "main")
                         ["dataset"]["subsets"], ["main"])
        override.write_text('{"dataset":{"name":"gsm8k","subsets":["diamond"]}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset.subsets for gsm8k"):
            self.settings("--settings", "gsm8k.json")

    def test_invalid_dataset_and_cross_dataset_subsets(self):
        for arguments in (("--dataset", "gsm8k", "--subset", "diamond"),
                          ("--dataset", "gpqa", "--subset", "test"),
                          ("--dataset", "gsm8k", "--subset", "all", "main")):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "dataset.subsets"):
                self.settings(*arguments)
        for name in ("missing", [], None):
            (self.root / "override.json").write_text(json.dumps({"dataset": {"name": name}}), encoding="utf-8")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "dataset.name"):
                self.settings("--settings", "override.json")

    def test_generated_configs_use_gsm8k_reader_and_numeric_scoring(self):
        for mode in ("cot", "str"):
            with self.subTest(mode=mode):
                settings = self.settings("--dataset", "gsm8k", "--prompt", mode, "--temperature", "0.3")
                path = runner.generate_configs(self.root, settings)
                self.assertEqual(path.name, "gsm8k_benchmark.py")
                cfg = runpy.run_path(str(path))
                self.assertEqual(len(cfg["datasets"]), 1)
                dataset = cfg["datasets"][0]
                self.assertEqual(dataset["type"], "ais_bench.benchmark.datasets.GSM8KDataset")
                self.assertEqual(dataset["reader_cfg"],
                                 {"input_columns": ["question"], "output_column": "answer", "test_split": "test"})
                self.assertEqual(dataset["eval_cfg"], {
                    "evaluator": {"type": "ais_bench.benchmark.datasets.Gsm8kEvaluator"},
                    "pred_postprocessor": {"type": "ais_bench.benchmark.datasets.gsm8k_postprocess"},
                    "dataset_postprocessor": {"type": "ais_bench.benchmark.datasets.gsm8k_dataset_postprocess"},
                })
                template = dataset["infer_cfg"]["prompt_template"]["template"]
                prompt = template["round"][0]["prompt"] if mode == "cot" else template
                self.assertIn("What is 2 + 2?", prompt.format(question="What is 2 + 2?"))
                standalone = runpy.run_path(str(self.root / "configs/datasets/gsm8k/gsm8k_local.py"))
                self.assertEqual(standalone["gsm8k_datasets"], cfg["datasets"])
                self.assertEqual(cfg["models"][0]["generation_kwargs"]["temperature"], 0.3)
                self.assertEqual(cfg["models"][0]["generation_kwargs"]["top_k"], 20)

    def test_gsm8k_prompt_rejects_answer_leakage(self):
        (self.root / "prompts/gsm8k/cot.txt").write_text("{question}\n{answer}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "placeholders"):
            runner.dataset_configs(self.root, self.settings("--dataset", "gsm8k")["dataset"])

    def test_missing_empty_corrupt_and_invalid_answer_data(self):
        folder = self.root / "data/gsm8k"
        folder.mkdir(parents=True)
        good = json.dumps({"question": "What is 2 + 2?", "answer": "2 + 2 = 4\n#### 4"}) + "\n"
        (folder / "test.jsonl").write_text(good, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train.jsonl"):
            runner.validate_data(self.root, ["test"], "gsm8k")
        (folder / "train.jsonl").write_text(good, encoding="utf-8")
        self.assertEqual(runner.validate_data(self.root, ["test"], "gsm8k"), {"test": 1})
        for split in ("train", "test"):
            path = folder / f"{split}.jsonl"
            for content in ("", "not json\n", "[]\n", "null\n", "{}\n", "\n",
                            '{"question":"q","answer":"4"}\n',
                            '{"question":"q","answer":"#### nope"}\n',
                            '{"question":1,"answer":"#### 4"}\n'):
                with self.subTest(split=split, content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "GSM8K"):
                        runner.validate_data(self.root, ["test"], "gsm8k")
            path.write_text(good, encoding="utf-8")

    def test_cli_after_move_uses_gsm8k_without_gpqa_data(self):
        shutil.copytree(ROOT / "data/gsm8k", self.root / "data/gsm8k")
        runner.generate_configs(self.root, self.settings("--dataset", "gsm8k"))
        moved = Path(self.temp.name) / "moved 工具"
        shutil.copytree(self.root, moved)
        moved = moved.resolve()
        fake_install = Path(self.temp.name) / "fake site packages"
        package = fake_install / "ais_bench/benchmark/cli"
        package.mkdir(parents=True)
        for directory in (package, package.parent, package.parent.parent):
            (directory / "__init__.py").write_text("")
        (package / "main.py").write_text(
            "import json, pathlib, runpy, sys\n"
            "cfg = runpy.run_path(sys.argv[1])\n"
            "dataset = cfg['datasets'][0]\n"
            "rows = [json.loads(line) for line in (pathlib.Path(dataset['path']) / 'test.jsonl')"
            ".read_text(encoding='utf-8').splitlines()]\n"
            "pathlib.Path('invocation.json').write_text(json.dumps({'argv':sys.argv[1:],"
            "'dataset':dataset, 'count':len(rows)}), encoding='utf-8')\n",
            encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(fake_install), PYTHONUTF8="1")
        result = subprocess.run([sys.executable, str(moved / "run_gpqa.py"), "--dataset", "gsm8k",
                                 "--num-prompts", "5", "--no-acc-len", "--", "--num-warmups", "0"],
                                cwd=self.temp.name, env=env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Local GSM8K: test=1319", result.stdout)
        observed = runner.read_json(moved / "invocation.json")
        self.assertEqual(Path(observed["argv"][0]), moved / "configs/gsm8k_benchmark.py")
        self.assertEqual(Path(observed["dataset"]["path"]), moved / "data/gsm8k")
        self.assertEqual(observed["count"], 1319)
        self.assertEqual(observed["argv"][-4:], ["--num-prompts", "5", "--num-warmups", "0"])
        self.assertFalse((Path(self.temp.name) / "configs").exists())


if __name__ == "__main__":
    unittest.main()
