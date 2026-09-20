"""Acceptance-length math, Prometheus parsing and post-benchmark sequencing."""

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acc_len
import run_gpqa as runner


METRICS = '''# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{model_name="test model",engine="0"} 100
vllm:spec_decode_num_drafts_created{engine="0"} 1730000000
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0",engine="0"} 80
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1",engine="0"} 40
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="2",engine="0"} 20
vllm:spec_decode_num_accepted_tokens_per_pos_created{position="0"} 1730000000
vllm:spec_decode_num_draft_tokens_total 300
'''


class MetricsTests(unittest.TestCase):
    def test_formula_matches_provided_script_and_ignores_created(self):
        result = acc_len.calculate_acc_len(METRICS, 3)
        self.assertEqual(result["num_drafts"], 100)
        self.assertEqual(result["num_accepted_tokens_per_pos"], [80, 40, 20])
        self.assertEqual(result["acceptance_per_pos"], [0.8, 0.4, 0.2])
        self.assertAlmostEqual(result["acc_len"], 2.4)
        self.assertEqual(result["status"], "ok")

    def test_auto_detection_multi_engine_and_scientific_notation(self):
        more = '''vllm:spec_decode_num_drafts_total{engine="1"} 1e2 1234
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 40 # {trace_id="a"} 0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="1"} 20
'''
        result = acc_len.calculate_acc_len(METRICS + more)
        self.assertEqual(result["num_drafts"], 200)
        self.assertEqual(result["num_accepted_tokens_per_pos"], [120, 60, 20])
        self.assertEqual(result["acc_len"], 2.0)

    def test_escaped_labels_sparse_positions_and_bare_counter_names(self):
        text = 'vllm:spec_decode_num_drafts 10\n'
        text += r'vllm:spec_decode_num_accepted_tokens_per_pos{model_name="a, position=\"99\" {x}", position="1"} 4'
        drafts, accepted = acc_len.analyse_metrics(text, 3)
        self.assertEqual((drafts, accepted), (10, [0, 4, 0]))

    def test_zero_drafts_has_no_fabricated_length(self):
        result = acc_len.calculate_acc_len('vllm:spec_decode_num_drafts_total 0\n', 3)
        self.assertEqual(result["status"], "no_draft")
        self.assertIsNone(result["acc_len"])
        self.assertIsNone(result["acceptance_per_pos"])

    def test_missing_invalid_or_out_of_range_metrics(self):
        invalid = [
            ('other_metric 1\n', None),
            ('vllm:spec_decode_num_drafts_total 5\n', None),
            (METRICS, 2), (METRICS, 0), (METRICS, True),
            (METRICS.replace('position="1"', 'position="-1"'), 3),
            (METRICS.replace('position="1"', 'rank="1"'), 3),
            (METRICS.replace('} 100', '} NaN'), 3),
            (METRICS.replace('} 100', '} Inf'), 3),
            (METRICS.replace('} 100', '} -1'), 3),
        ]
        for text, count in invalid:
            with self.subTest(count=count, text=text[:40]), self.assertRaises(ValueError):
                acc_len.calculate_acc_len(text, count)

    def test_endpoint_tracks_service_address(self):
        self.assertEqual(acc_len.metrics_url({"host_ip": "10.0.0.5", "host_port": 8989}),
                         'http://10.0.0.5:8989/metrics')
        self.assertEqual(acc_len.metrics_url({"host_ip": "::", "host_port": 9000, "enable_ssl": True}),
                         'https://[::1]:9000/metrics')
        self.assertEqual(acc_len.metrics_url({"host_ip": "0.0.0.0"}), 'http://127.0.0.1:8080/metrics')
        self.assertEqual(acc_len.metrics_url({"url": "https://host/proxy/"}), 'https://host/proxy/metrics')
        self.assertEqual(acc_len.metrics_url({}, 'http://metrics:9001/metrics'), 'http://metrics:9001/metrics')

    def test_fetch_uses_timeout_and_returns_stats(self):
        with patch.object(acc_len, 'urlopen', return_value=io.BytesIO(METRICS.encode())) as fetch:
            result = acc_len.collect_acc_len('http://localhost:9000/metrics', 3, 1.5)
        self.assertAlmostEqual(result['acc_len'], 2.4)
        self.assertEqual(fetch.call_args.args[0].full_url, 'http://localhost:9000/metrics')
        self.assertEqual(fetch.call_args.kwargs['timeout'], 1.5)

    def test_report_persists_success_and_network_errors(self):
        for response in (None, URLError('connection refused'), HTTPError('http://localhost', 503, 'unavailable', {}, None)):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as folder, \
                    patch.object(acc_len, 'collect_acc_len', side_effect=response,
                                 return_value=acc_len.calculate_acc_len(METRICS)), patch('builtins.print'):
                report = acc_len.report_acc_len({}, {}, Path(folder))
                paths = list(Path(folder).glob('acc_len_*.json'))
                self.assertEqual(len(paths), 1)
                self.assertEqual(json.loads(paths[0].read_text(encoding='utf-8')), report)
                self.assertEqual(report['scope'], 'server_cumulative')
                if response:
                    self.assertEqual(report['status'], 'unavailable')
                    self.assertIsNone(report['acc_len'])
                else:
                    self.assertAlmostEqual(report['acc_len'], 2.4)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        shutil.copyfile(ROOT / 'settings.json', self.root / 'settings.json')

    def run_main(self, arguments=(), exit_code=0):
        events = []
        def benchmark(*args, **kwargs):
            events.append('benchmark')
            return SimpleNamespace(returncode=exit_code)
        def metrics(*args, **kwargs):
            events.append('metrics')
            return {'status': 'unavailable', 'acc_len': None}
        with patch.object(runner, 'ROOT', self.root), \
                patch.object(runner, 'validate_data', return_value={'diamond': 198}), \
                patch.object(runner, 'generate_configs', return_value=self.root / 'configs/gpqa_benchmark.py'), \
                patch.object(runner.importlib.util, 'find_spec', return_value=object()), \
                patch.object(runner.subprocess, 'run', side_effect=benchmark), \
                patch.object(runner, 'report_acc_len', side_effect=metrics) as report, \
                patch('builtins.print'):
            code = runner.main(list(arguments))
        return code, events, report

    def test_success_runs_metrics_after_benchmark_and_preserves_success(self):
        code, events, report = self.run_main(['--host-port', '8989', '--num-spec', '3'])
        self.assertEqual(code, 0)
        self.assertEqual(events, ['benchmark', 'metrics'])
        self.assertEqual(report.call_args.args[0]['host_port'], 8989)
        self.assertEqual(report.call_args.args[1]['num_speculative_tokens'], 3)
        self.assertEqual(report.call_args.args[2], self.root / 'outputs/acc_len')

    def test_failed_benchmark_keeps_exit_code_without_metrics(self):
        code, events, _ = self.run_main(exit_code=17)
        self.assertEqual((code, events), (17, ['benchmark']))

    def test_generation_disable_offline_and_forwarded_dry_run_skip_metrics(self):
        for arguments in (['--generate-only'], ['--no-acc-len'], ['--mode', 'eval'], ['--mode', 'perf'],
                          ['--', '--dry-run'], ['--', '--search'], ['--', '-m', 'eval'],
                          ['--', '--mode=perf']):
            with self.subTest(arguments=arguments):
                code, events, _ = self.run_main(arguments)
                self.assertEqual(code, 0)
                self.assertNotIn('metrics', events)
                if arguments == ['--generate-only']:
                    self.assertEqual(events, [])

    def test_settings_options_and_validation(self):
        args = runner.parser().parse_args(['--metrics-url', 'https://metrics/metrics', '--metrics-timeout', '2'])
        settings = runner.load_settings(args, self.root)
        self.assertEqual(settings['acc_len']['metrics_url'], 'https://metrics/metrics')
        self.assertEqual(settings['acc_len']['timeout'], 2)
        for arguments in (['--num-spec', '0'], ['--metrics-timeout', '0'], ['--metrics-timeout', 'nan'],
                          ['--metrics-url', 'file:///metrics']):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                runner.load_settings(runner.parser().parse_args(arguments), self.root)

    def test_old_settings_without_acc_len_section_remain_supported(self):
        path = self.root / 'settings.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        del saved['acc_len']
        path.write_text(json.dumps(saved), encoding='utf-8')
        settings = runner.load_settings(runner.parser().parse_args([]), self.root)
        self.assertTrue(settings['acc_len']['enabled'])


if __name__ == '__main__':
    unittest.main()
