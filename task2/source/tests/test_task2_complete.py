import hashlib
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch

from scripts.task2_finish import disconnect_after_verified
from scripts.task2_stage import checked_copy, restore_archive
from task2_ppo.cache_geometry import geometry_summary, reconstruct_cached
from task2_ppo.evaluation_complete import (activate_adapter, evaluate_condition,
                                          evaluate_prompt, summarize_evaluation)
from task2_ppo.session_budget import SessionBudget, SessionBudgetReached, UnlimitedTestBudget
from test_task2_continuation import batch, models
from common.data import load_yaml


class Tokenizer:
    eos_token_id, pad_token_id = 6, 0
    def __call__(self, text, return_tensors=None, **kwargs):
        if return_tensors:
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        return {"input_ids": [int(t) for t in text.split()]}
    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i != 6)
    def apply_chat_template(self, *args, **kwargs):
        return "test prompt"


def metric_row(index=0):
    return {"schedule": {"row_index": index, "prompt_id": str(index), "source_index": index,
                         "generation_seed": 106304 + index},
            "response_length": 3, "terminated_with_eos": True, "truncated": False,
            "raw_reward": .7, "effective_reward": .7, "reference_kl": -.1,
            "sampled_entropy": 2., "categorical_entropy": 3., "response": "tiny test"}


class CompleteTaskTests(unittest.TestCase):
    def test_prompt_and_token_weighted_metrics_are_distinct_and_signed(self):
        a, b = metric_row(), metric_row(1)
        a.update(response_length=1, reference_kl=-.1)
        b.update(response_length=3, reference_kl=.3)
        result = summarize_evaluation([a, b])
        self.assertAlmostEqual(result["reference_kl_mean"], .1)
        self.assertAlmostEqual(result["reference_kl_token_weighted"], .2)
        self.assertEqual(result["response_length_std_population"], 1.)
        self.assertEqual(result["categorical_entropy_mean"], 3.)

    def test_evaluation_uses_recorded_seed_and_raw_reward(self):
        actor, _, _, _ = models()
        draws = []
        def generated(*args, **kwargs):
            draws.append(int(torch.randint(0, 100000, (1,))))
            return batch(eos=False)
        row = {"prompt_id": "0", "source_index": 0, "messages": [{"role": "user", "content": "test"}]}
        entry = metric_row()["schedule"]
        cfg = load_yaml("configs/ppo.yaml")
        with mock.patch("task2_ppo.evaluation_complete.batch_generate", side_effect=generated), mock.patch(
                "task2_ppo.evaluation_complete.score_reward_pairs", return_value=torch.tensor([2.])):
            first = evaluate_prompt(actor, Tokenizer(), None, row, entry, cfg)
            second = evaluate_prompt(actor, Tokenizer(), None, row, entry, cfg)
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(first["raw_reward"], 2.)
        self.assertEqual(first["effective_reward"], 1.)
        self.assertEqual(first["response_token_ids"], second["response_token_ids"])

    def test_evaluation_resumes_without_regenerating_persisted_prompts(self):
        cfg = load_yaml("configs/ppo.yaml")
        entries = [metric_row(i)["schedule"] for i in range(3)]
        counter = {"calls": 0}
        class Budget:
            def check(self):
                if counter["calls"] >= 2:
                    raise SessionBudgetReached()
        def evaluate(*args):
            entry = args[4]
            counter["calls"] += 1
            return metric_row(entry["row_index"])
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
                "task2_ppo.evaluation_complete.evaluate_prompt", side_effect=evaluate):
            root = Path(temporary)
            args = (None, None, None, [{}, {}, {}], {"evaluation": entries}, cfg, "standard", {"test": True}, root / "local", root / "drive")
            with self.assertRaises(SessionBudgetReached):
                evaluate_condition(*args, Budget())
            summary = evaluate_condition(*args, UnlimitedTestBudget())
            self.assertEqual(counter["calls"], 3)
            self.assertEqual(summary["metrics"]["prompts"], 3)
            self.assertEqual(len(list((root / "drive/standard").glob("prompt_*.zip"))), 3)

    def test_fixed_cache_affected_fraction_differs_from_active_clipping(self):
        row = {"new_logprobs": [math.log(1.5), math.log(.5), math.log(.5), math.log(1.5)],
               "old_logprobs": [0.] * 4, "advantages": [1., -1., 1., -1.]}
        result = geometry_summary([row])["results"]
        self.assertEqual(result[1]["affected_token_fraction"], 1.)
        self.assertEqual(result[1]["active_clipped_branch_fraction"], .5)
        self.assertAlmostEqual(result[1]["clipped_surrogate"], -.15, places=6)
        self.assertEqual(result[2]["affected_token_fraction"], 0.)

    def test_cache_reconstruction_preserves_exact_length_and_eos(self):
        cached = {"response": "4 5", "response_tokens": 3, "terminated_with_eos": True,
                  "prompt_id": "test", "source_index": 7}
        row = {"prompt_id": "test", "source_index": 7, "messages": []}
        b = reconstruct_cached(Tokenizer(), cached, row, load_yaml("configs/ppo.yaml"))
        self.assertEqual(b["prompt_width"], 3)
        self.assertEqual(b["response_ids"].tolist(), [[4, 5, 6]])
        cached["response_tokens"] = 4
        with self.assertRaisesRegex(RuntimeError, "no trimming/padding"):
            reconstruct_cached(Tokenizer(), cached, row, load_yaml("configs/ppo.yaml"))

    def test_adapter_switch_keeps_evaluation_frozen(self):
        actor, _, _, _ = models()
        actor.add_adapter("midpoint", actor.peft_config["default"])
        activate_adapter(actor, "midpoint", "unused")
        self.assertFalse(actor.training)
        self.assertTrue(all(not p.requires_grad for p in actor.parameters()))

    def test_session_limit_includes_elapsed_setup_and_rejects_unbounded_values(self):
        with mock.patch("time.time", return_value=10000.), mock.patch("time.monotonic", return_value=500.):
            budget = SessionBudget(120, session_start_epoch=10000. - 121 * 60)
            with self.assertRaises(SessionBudgetReached):
                budget.check()
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                SessionBudget(value)

    def test_automatic_disconnect_requires_verified_backup_and_successful_flush(self):
        disconnect, flush = mock.Mock(), mock.Mock()
        with self.assertRaises(RuntimeError):
            disconnect_after_verified({"approved": True, "automatic_disconnect": True}, False, flush, disconnect)
        disconnect.assert_not_called()
        flush.side_effect = OSError("flush failed")
        with self.assertRaises(OSError):
            disconnect_after_verified({"approved": True, "automatic_disconnect": True}, True, flush, disconnect)
        disconnect.assert_not_called()
        flush.side_effect = None
        disconnect_after_verified({"approved": True, "automatic_disconnect": True}, True, flush, disconnect)
        disconnect.assert_called_once()

    def test_staged_archives_verify_bytes_and_refuse_modified_local_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"immutable course fixture"
            archive = root / "assets.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("data/test.jsonl", payload)
            spec = {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            hashes = {"data/test.jsonl": hashlib.sha256(payload).hexdigest()}
            restore_archive(spec, root / "repo", hashes)
            (root / "repo/data/test.jsonl").write_text("modified")
            with self.assertRaises(RuntimeError):
                restore_archive(spec, root / "repo", hashes)

    def test_streamed_copy_rejects_incorrect_public_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").write_bytes(b"data")
            with self.assertRaises(RuntimeError):
                checked_copy(root / "source", root / "target", "0" * 64)
            self.assertFalse((root / "target").exists())

    def test_finish_recovers_committed_but_not_uploaded_prompt_and_saves_log(self):
        from scripts.task2_finish import finish_session
        from task2_ppo.storage import atomic_json, commit_directory, restore_published
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / 'repo/results/task2_ppo/evaluation/standard'
            local.mkdir(parents=True)
            pending = local / '.draft'
            pending.mkdir()
            atomic_json(pending / 'record.json', {'completed_fixture': True})
            commit_directory(pending, local / 'prompt_0000')
            logs = root / 'repo/results/task2_ppo/gpu_sessions'
            draft = logs / '.draft_session'
            draft.mkdir(parents=True)
            (draft / 'console.log').write_text('saved fixture log')
            result = finish_session(root / 'repo', root / 'drive', draft, 'session', {'process_stopped': True})
            self.assertTrue(result['verified'])
            self.assertTrue((root / 'drive/evaluation/standard/prompt_0000.zip').is_file())
            restore_published(root / 'drive/gpu_sessions/session.zip', root / 'restored_session')
            metadata = json.loads((root / 'restored_session/session.json').read_text())
            self.assertEqual(metadata['persistence']['interrupted_uploads_recovered'], 1)
            self.assertEqual((root / 'restored_session/console.log').read_text(), 'saved fixture log')



class PipelineTests(unittest.TestCase):
    def test_all_endpoints_freeze_before_seven_complete_evaluations(self):
        from task2_ppo import pipeline
        from task2_ppo.storage import atomic_json, commit_directory
        cfg = load_yaml('configs/ppo.yaml')
        approval = json.loads(Path('docs/task2_approval.json').read_text())
        resource = {'approved': True, 'session_minutes': 120, 'automatic_disconnect': True}
        audit = {'environment': {'cpu_test': True}, 'approvals': [approval], 'source': {},
                 'chunk2_files': {}, 'schedule_sha256': 'test', 'public_models': {},
                 'assets': {'verified_files': {cfg['cached_rollouts']: 'fixture'}}}
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'docs').mkdir()
            atomic_json(root / 'docs/task2_approval.json', approval)
            atomic_json(root / 'docs/task2_chunk3_files.json', {})
            atomic_json(root / 'docs/task2_resource_approval.json', resource)
            torch.save([{}] * 32, root / 'cache.pt')
            cfg['cached_rollouts'] = 'cache.pt'
            audit['assets']['verified_files'] = {'cache.pt': 'fixture'}
            def path(name): return root / name
            def run(config, spec, train, schedule, folders, observed, *args, **kwargs):
                calls.append(('train', spec['name'], spec['updates']))
                run_cfg = {**config, 'updates': spec['updates'], 'clip_epsilon': spec['clip_epsilon'], 'kl_beta': spec['kl_beta']}
                contract = {'run': spec, 'release_config': run_cfg, 'approvals': observed['approvals'],
                    'source': observed['source'], 'chunk2_files': observed['chunk2_files'], 'assets': observed['assets']['verified_files'],
                    'models': observed['public_models'], 'schedule_sha256': observed['schedule_sha256']}
                destination = path('outputs/task2_ppo') / spec['name']
                destination.mkdir(parents=True)
                draft = destination / '.draft'
                draft.mkdir()
                atomic_json(draft / 'summary.json', {'completed_updates': spec['updates'], 'contract': contract})
                atomic_json(draft / 'policy/adapter_config.json', {'fixture': True})
                commit_directory(draft, destination / 'final')
            def evaluate(model, tok, rm, rows, schedule, config, name, *args):
                self.assertTrue((root / 'drive/frozen_candidates.json').is_file())
                self.assertEqual(sum(c[0] == 'train' for c in calls), 6)
                calls.append(('eval', name, len(rows)))
            def cache(model, tok, rows, *args): calls.append(('cache', 'fixed', len(rows)))
            with mock.patch.multiple(pipeline,
                    repo_path=path, validate_job=mock.Mock(return_value=(cfg, [], {}, {'policy': 'fixture'}, audit)),
                    read_jsonl=mock.Mock(return_value=[{}] * 200),
                    load_tokenizer_local=mock.Mock(return_value=None),
                    load_reward_local=mock.Mock(return_value=(None, {})),
                    load_evaluation_policy=mock.Mock(return_value=(None, {})),
                    activate_adapter=mock.Mock(), generation_protocol=mock.Mock(return_value={}),
                    evaluate_condition=evaluate, score_cache=cache), mock.patch.object(
                    pipeline.continuation, 'run_one', side_effect=run):
                self.assertEqual(pipeline.run_pipeline('models', root / 'drive', 120), 0)
            self.assertEqual(sum(c[2] for c in calls if c[0] == 'train'), 60)
            self.assertEqual([c[2] for c in calls if c[0] == 'eval'], [200] * 7)
            self.assertEqual(calls[-1], ('cache', 'fixed', 32))

    def test_inherited_decoding_defaults_are_recorded_and_changes_rejected(self):
        from transformers import GenerationConfig
        from types import SimpleNamespace
        from task2_ppo.generation_audit import generation_protocol
        cfg = load_yaml('configs/ppo.yaml')
        model = SimpleNamespace(generation_config=GenerationConfig(top_k=20, repetition_penalty=1.1,
                                do_sample=True, use_cache=True), config=SimpleNamespace(use_cache=False))
        report = generation_protocol(model, Tokenizer(), 768, cfg)
        self.assertEqual(report['course_generate_overrides']['top_p'], .9)
        self.assertEqual(report['inherited_generation_config']['top_k'], 20)
        self.assertTrue(report['inherited_generation_config']['use_cache'])
        model.generation_config.top_k = 50
        with self.assertRaises(RuntimeError):
            generation_protocol(model, Tokenizer(), 768, cfg)

if __name__ == "__main__":
    unittest.main(verbosity=2)
