"""Frozen text-only LoRA target selection on a synthetic module tree."""
import unittest

from training import train


class FakeLinear:
    linear = True


class FakeNonLinear:
    linear = False


def named_modules_fixture(skip_layers=(), wrong_prefix=False):
    rows = [('model.visual.blocks.0.attn.q_proj', FakeLinear()),
            ('model.language_model.layers.3.mlp.gate_proj', FakeLinear()),
            ('model.language_model.layers.3.self_attn.o_proj', FakeLinear())]
    for layer in range(32):
        if layer in skip_layers:
            continue
        for projection in ('q_proj', 'v_proj'):
            if wrong_prefix:
                name = 'model.layers.%d.self_attn.%s' % (layer, projection)
            else:
                name = 'model.language_model.layers.%d.self_attn.%s' % (layer, projection)
            rows.append((name, FakeLinear()))
    return rows


class TrainAdapterTests(unittest.TestCase):
    def test_selects_exactly_the_sixteen_frozen_modules(self):
        targets = train.select_adapter_targets(named_modules_fixture())
        expected = ['model.language_model.layers.%d.self_attn.%s' % (layer, projection)
                    for layer in train.LORA_LAYER_INDICES
                    for projection in train.LORA_PROJECTIONS]
        self.assertEqual(targets, expected)
        self.assertEqual(len(targets), 16)
        self.assertTrue(all('layers.0.' not in name for name in targets))
        self.assertTrue(all('.self_attn.' in name for name in targets))

    def test_missing_module_is_a_refusal_not_a_smaller_adapter(self):
        with self.assertRaises(train.TrainGateError) as caught:
            train.select_adapter_targets(named_modules_fixture(skip_layers=(15,)))
        message = str(caught.exception)
        self.assertIn('matched 14 of 16', message)
        self.assertIn('layer 15', message)

    def test_unexpected_naming_is_refused(self):
        with self.assertRaises(train.TrainGateError) as caught:
            train.select_adapter_targets(named_modules_fixture(wrong_prefix=True))
        self.assertIn('matched 0 of 16', str(caught.exception))

    def test_non_linear_modules_are_not_targets(self):
        modules = []
        for name, module in named_modules_fixture():
            if name == 'model.language_model.layers.7.self_attn.q_proj':
                modules.append((name, FakeNonLinear()))
            else:
                modules.append((name, module))
        with self.assertRaises(train.TrainGateError) as caught:
            train.select_adapter_targets(
                modules, is_linear=lambda module: getattr(module, 'linear', True))
        self.assertIn('layer 7 q_proj', str(caught.exception))

    def test_lora_config_is_the_frozen_configuration(self):
        targets = ['module_%d' % index for index in range(16)]
        kwargs = train.lora_config_kwargs(targets)
        self.assertEqual(kwargs['r'], 8)
        self.assertEqual(kwargs['lora_alpha'], 16)
        self.assertEqual(kwargs['lora_dropout'], 0.0)
        self.assertEqual(kwargs['bias'], 'none')
        self.assertEqual(kwargs['task_type'], 'CAUSAL_LM')
        self.assertEqual(kwargs['target_modules'], targets)
        with self.assertRaises(train.TrainGateError):
            train.lora_config_kwargs(targets[:15])
