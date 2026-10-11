import math
import unittest
import torch
from task3_grpo.grpo import group_relative_advantages as advantage, grpo_policy_loss as loss, mask_truncated_sequences
from task3_grpo.training_core import length_gradient_records, categorical_entropy


class ObjectiveTests(unittest.TestCase):
    def test_constant_groups_are_zero(self):
        torch.testing.assert_close(advantage(torch.tensor([1.,1.,10.,10.]),torch.tensor([0,0,1,1])),torch.zeros(4))
    def test_other_groups_do_not_change_sign(self):
        a = advantage(torch.tensor([0.,2.]),torch.tensor([7,7]))
        b = advantage(torch.tensor([0.,2.,100.,102.]),torch.tensor([7,7,8,8]))[:2]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_noncontiguous_permutation(self):
        r,g = torch.tensor([1.,3.,10.,12.]),torch.tensor([11,11,-5,-5])
        p = torch.tensor([2,0,3,1])
        torch.testing.assert_close(advantage(r[p],g[p]),advantage(r,g)[p])
    def test_population_std_and_epsilon_addition(self):
        r = torch.tensor([0.,2e-6])
        torch.testing.assert_close(advantage(r,torch.zeros(2)),torch.tensor([-.5,.5]))
    def test_singleton_group(self):
        torch.testing.assert_close(advantage(torch.tensor([2.]),torch.tensor([99])),torch.zeros(1))
    def test_advantages_detached_float32(self):
        a = advantage(torch.tensor([0.,2.],dtype=torch.float64,requires_grad=True),torch.tensor([0,0]))
        self.assertFalse(a.requires_grad)
        self.assertEqual(a.dtype,torch.float32)
    def test_nonfinite_rewards_rejected(self):
        with self.assertRaises(ValueError):
            advantage(torch.tensor([float('nan')]),torch.tensor([0]))
    def test_mismatched_groups_rejected(self):
        with self.assertRaises(ValueError):
            advantage(torch.ones(2),torch.ones(1))
    def test_nonfinite_group_ids_rejected(self):
        with self.assertRaises(ValueError):
            advantage(torch.ones(2),torch.tensor([0.,float('nan')]))
    def test_nonfinite_epsilon_rejected(self):
        with self.assertRaises(ValueError):
            advantage(torch.ones(2),torch.ones(2),float('nan'))
    def test_positive_clipping_uses_minimum(self):
        n = torch.tensor([[math.log(1.5)]],requires_grad=True)
        value,_ = loss(n,torch.zeros_like(n),torch.ones(1),torch.ones_like(n),n.detach(),.2,0.)
        self.assertAlmostEqual(float(value.detach()),-1.2,places=6)
        value.backward()
        self.assertEqual(float(n.grad),0.)
    def test_negative_clipping_uses_minimum(self):
        n = torch.tensor([[math.log(.5)]],requires_grad=True)
        value,_ = loss(n,torch.zeros_like(n),-torch.ones(1),torch.ones_like(n),n.detach(),.2,0.)
        self.assertAlmostEqual(float(value.detach()),.8,places=6)
        value.backward()
        self.assertEqual(float(n.grad),0.)
    def test_unclipped_gradient(self):
        n = torch.zeros((1,2),requires_grad=True)
        value,_ = loss(n,n.detach(),torch.ones(1),torch.ones_like(n),n.detach(),.2,0.)
        value.backward()
        torch.testing.assert_close(n.grad,torch.tensor([[-.5,-.5]]))
    def test_canonical_and_constant_denominators(self):
        n = torch.zeros((2,4),requires_grad=True)
        mask = torch.tensor([[1.,0,0,0],[1.,1,1,1]])
        a,_ = loss(n,n.detach(),torch.tensor([1.,-1.]),mask,n.detach(),.2,0.)
        b,_ = loss(n,n.detach(),torch.tensor([1.,-1.]),mask,n.detach(),.2,0.,'dr_grpo',4)
        self.assertEqual(float(a.detach()),0.)
        self.assertEqual(float(b.detach()),.375)
    def test_all_masked_is_differentiable_zero(self):
        n = torch.zeros((2,3),requires_grad=True)
        value,_ = loss(n,n.detach(),torch.ones(2),torch.zeros_like(n),n.detach(),.2,.1)
        value.backward()
        self.assertEqual(float(value.detach()),0.)
        torch.testing.assert_close(n.grad,torch.zeros_like(n))
    def test_truncation_masks_entire_completion(self):
        torch.testing.assert_close(mask_truncated_sequences(torch.tensor([[1.,1.,0],[1.,1.,1]]),[False,True]),torch.tensor([[1.,1.,0],[0.,0.,0]]))
    def test_padding_does_not_change_loss(self):
        a,_ = loss(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1),torch.ones(1,1),torch.zeros(1,1),.2,.1)
        b,_ = loss(torch.tensor([[0.,4.]]),torch.zeros(1,2),torch.ones(1),torch.tensor([[1.,0]]),torch.zeros(1,2),.2,.1)
        self.assertEqual(float(a),float(b))
    def test_kl_estimators_have_different_meaning(self):
        _,d = loss(torch.tensor([[-.5]]),torch.tensor([[-.5]]),torch.zeros(1),torch.ones(1,1),torch.zeros(1,1),.2,.1)
        self.assertLess(float(d['reference_kl']),0.)
        self.assertGreater(float(d['loss_kl_penalty']),0.)
    def test_gradient_length_formula(self):
        p = torch.zeros((2,4),requires_grad=True)
        fixed = dict(old_logprobs=p.detach(),ref_logprobs=p.detach(),advantages=torch.tensor([1.,1.]),loss_mask=torch.tensor([[1.,0,0,0],[1.,1,1,1]]))
        cfg = dict(clip_epsilon=.2,kl_beta=.1,max_completion_length=4)
        _,_,a = length_gradient_records(p,fixed,cfg,'grpo')
        _,_,b = length_gradient_records(p,fixed,cfg,'dr_grpo')
        self.assertEqual([r['policy_logprob_gradient_l2'] for r in a],[.5,.25])
        self.assertEqual([r['policy_logprob_gradient_l2'] for r in b],[.125,.25])
    def test_categorical_entropy(self):
        self.assertAlmostEqual(float(categorical_entropy(torch.zeros(1,2,4),torch.tensor([[1.,0]]))),math.log(4),places=6)
    def test_unknown_normalization_rejected(self):
        with self.assertRaises(ValueError):
            loss(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1),torch.ones(1,1),torch.zeros(1,1),.2,.1,'unknown')
