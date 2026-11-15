"""CPU contract checks without importing model/CUDA extensions."""
import copy
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor
from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook

file = Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/core/optimizer/grouped_adamw.py'
spec = importlib.util.spec_from_file_location('grouped_optimizer_under_test', file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        self.head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)

class ConstructorTest(unittest.TestCase):
    def test_updates_lr_and_grouped_checkpoint_resume(self):
        torch.manual_seed(42)
        original = Model()
        grouped = copy.deepcopy(original)
        cfg = dict(type='AdamW', lr=4e-4, weight_decay=.01, fused=False, foreach=False)
        paramwise = dict(custom_keys={'img_backbone':dict(lr_mult=.1)}, bias_decay_mult=0.)
        a = DefaultOptimizerConstructor(cfg, paramwise)(original)
        cfg_b = dict(cfg, foreach=True)
        b = module.GroupedAdamWOptimizerConstructor(cfg_b, paramwise)(grouped)
        self.assertLess(len(b.param_groups), len(a.param_groups))
        def options(opt, model):
            by_id = {id(p):n for n,p in model.named_parameters()}
            return {by_id[id(p)]:{k:v for k,v in g.items() if k not in ('params','foreach')}
                    for g in opt.param_groups for p in g['params']}
        self.assertEqual(options(a, original), options(b, grouped))
        self.assertEqual({id(p) for p in grouped.parameters()},
                         {id(p) for g in b.param_groups for p in g['params']})
        runners=[SimpleNamespace(optimizer=opt,max_iters=42192,iter=0) for opt in (a,b)]
        hooks=[CosineAnnealingLrUpdaterHook(by_epoch=False,warmup='linear',warmup_iters=500,
                                          warmup_ratio=1./3,min_lr_ratio=1e-3) for _ in runners]
        for hook,runner in zip(hooks,runners):hook.before_run(runner)
        for iteration in (0,1,499,500,501,1000,42191):
            for hook,runner in zip(hooks,runners):
                runner.iter=iteration;hook.before_train_iter(runner)
            self.assertEqual(options(a,original),options(b,grouped))
        for attempt in range(6):
            for hook,runner in zip(hooks,runners):
                runner.iter=attempt;hook.before_train_iter(runner)
            for ordinal,((name,p),(other,q)) in enumerate(zip(original.named_parameters(),grouped.named_parameters())):
                self.assertEqual(name,other)
                grad = None if not p.requires_grad or attempt==2 or (name=='head.1.bias' and attempt<3) else torch.full_like(p,.001*(ordinal+1)*(attempt+1))
                p.grad = grad
                q.grad = None if grad is None else grad.clone()
            a.step();b.step()
            for p,q in zip(original.parameters(),grouped.parameters()):
                torch.testing.assert_close(p,q,atol=1e-8,rtol=2e-6)
                self.assertEqual(a.state[p].keys(), b.state[q].keys())
                for key in a.state[p]:torch.testing.assert_close(a.state[p][key],b.state[q][key],atol=1e-8,rtol=2e-6)
        resumed_model = copy.deepcopy(grouped)
        resumed = module.GroupedAdamWOptimizerConstructor(cfg_b,paramwise)(resumed_model)
        resumed.load_state_dict(copy.deepcopy(b.state_dict()))
        self.assertEqual(len(resumed.param_groups),len(b.param_groups))
        for ordinal,(p,q) in enumerate(zip(grouped.parameters(),resumed_model.parameters())):
            self.assertEqual(b.state[p].keys(),resumed.state[q].keys())
            for key in b.state[p]:torch.testing.assert_close(b.state[p][key],resumed.state[q][key],atol=0,rtol=0)
            p.grad = torch.full_like(p,.002*(ordinal+1)) if p.requires_grad else None
            q.grad = torch.full_like(q,.002*(ordinal+1)) if q.requires_grad else None
        b.step();resumed.step()
        for p,q in zip(grouped.parameters(),resumed_model.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
        self.assertTrue(torch.equal(grouped.frozen,torch.ones_like(grouped.frozen)))

if __name__=='__main__':unittest.main()
