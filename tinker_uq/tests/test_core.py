import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from tinker_uq import training
from tinker_uq.data import read_rows, write_jsonl, one_per_question, read_predictions
from tinker_uq.conformal import quantile, partition_edges, fit_model, prediction_set, evaluate_model
from tinker_uq.analysis import tie_averaged_aurc


class Tokenizer:
    def encode(self,text,**kw): return [ord(c) for c in text]
    def apply_chat_template(self,messages,**kw): return [1,2,3]+[ord(c) for c in messages[0]['content']]


class Future:
    def __init__(self,value): self.value=value
    def result(self): return self.value


class ModelInput:
    @staticmethod
    def from_ints(value): return value


class FakeSampler:
    def compute_logprobs(self,tokens):
        # Multi-token labels A\n and B\n. None only at first prompt token.
        return Future([None]+[-.1]*(len(tokens)-3)+[math.log(.8) if tokens[-2]==65 else math.log(.2),0.])


def rows():
    return [dict(row_id=str(i),question_id=str(i),question=f'Question {i}',answer='SECRET',correct=i%2,split=s)
            for i,s in enumerate(['train','val','cal','test'])]


class CoreTests(unittest.TestCase):
    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl'; rs=rows()
            write_jsonl(p,rs); self.assertEqual(len(read_rows(p)),4)
            write_jsonl(p,rs+[dict(rs[0],row_id='extra',split='test')])
            with self.assertRaises(ValueError): read_rows(p)

    def test_question_only_no_answer_or_label(self):
        a=rows()[0]; b=dict(a,answer='CHANGED',correct=1)
        self.assertEqual(training.prompt_text(a,'q','normal'),training.prompt_text(b,'q','normal'))
        self.assertNotEqual(training.prompt_text(a,'qa','normal'),training.prompt_text(b,'qa','normal'))

    def test_teacher_forcing_alignment(self):
        inputs,targets,start=training.sequence_parts([10,11],[65,10])
        self.assertEqual(inputs,[10,11,65]); self.assertEqual(targets[start:],[65,10])

    def test_full_label_scoring_and_legend(self):
        output=io.StringIO()
        training.score_rows(FakeSampler(),rows(),Tokenizer(),'q','base',output,SimpleNamespace(ModelInput=ModelInput))
        result=[json.loads(s) for s in output.getvalue().splitlines()]
        self.assertEqual(len(result),8)
        for r in result:
            self.assertAlmostEqual(r['p_correct'],.8 if r['legend']=='normal' else .2)

    def test_missing_label_logprob_rejected(self):
        class Bad:
            def compute_logprobs(self,t): return Future([None]*len(t))
        with self.assertRaises(ValueError):
            training.score_rows(Bad(),rows(),Tokenizer(),'q','base',io.StringIO(),SimpleNamespace(ModelInput=ModelInput))

    def test_quantile_rank_and_empty(self):
        self.assertEqual(quantile(list(range(9)),.1),8)
        self.assertTrue(math.isinf(quantile([], .1)))
        self.assertTrue(math.isinf(quantile([.2]*8,.1)))
        self.assertEqual(quantile([.2]*9,.1),.2)

    def test_leave_one_out_rank_coverage(self):
        # Enumerate every possible held-out point among distinct exchangeable scores.
        all_scores=np.linspace(0,1,20)
        for alpha in (.05,.1,.2,.5):
            covered=[s<=quantile(np.delete(all_scores,i),alpha) for i,s in enumerate(all_scores)]
            self.assertGreaterEqual(np.mean(covered)+1e-12,1-alpha)

    def test_ties_collapse_partitions(self):
        self.assertEqual(partition_edges([.5]*20,16),[])
        self.assertEqual(len(partition_edges([0]*10+[1]*10,16)),1)

    def test_empty_cell_full_set_and_boundary(self):
        fit=[dict(question_id=f'f{i}',q=x,qa=x) for i,x in enumerate([0,0,1,1])]
        cal=[dict(question_id=f'c{i}',q=0,qa=0,anchor_p=.9,correct=1) for i in range(20)]
        model=fit_model(fit,cal,'q',2,.1)
        cell,labels=prediction_set(dict(q=1,anchor_p=.99),model)
        self.assertEqual(cell,1); self.assertEqual(labels,[0,1])
        _,labels=prediction_set(dict(q=0,anchor_p=.9),model)
        self.assertEqual(labels,[1])

    def test_fit_overlap_rejected(self):
        row=dict(question_id='same',q=.5,qa=.5,anchor_p=.5,correct=1)
        with self.assertRaises(ValueError): fit_model([row],[row],'q',1,.1)

    def test_sampling_ignores_labels_and_order(self):
        rs=[dict(question_id='q',row_id=str(i),correct=i%2) for i in range(10)]
        with self.assertRaises(ValueError): one_per_question(rs)
        chosen=one_per_question(rs,'sample',3)[0]['row_id']
        changed=[dict(r,correct=1-r['correct']) for r in reversed(rs)]
        self.assertEqual(chosen,one_per_question(changed,'sample',3)[0]['row_id'])

    def test_predictions_reject_cross_arm_label_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.jsonl'
            row=dict(row_id='r',question_id='q',split='test',correct=1,arm='q',stage='trained',legend='normal',p_correct=.5)
            write_jsonl(p,[row,dict(row,arm='qa',correct=0)])
            with self.assertRaises(ValueError): read_predictions(p)

    def test_aurc_ties_are_order_invariant(self):
        rs=[dict(p_correct=.5,correct=z) for z in (1,1,0,0)]
        self.assertAlmostEqual(tie_averaged_aurc(rs),.5)
        self.assertEqual(tie_averaged_aurc(rs),tie_averaged_aurc(rs[::-1]))

    def test_custom_loss_gradient_if_torch_available(self):
        try: import torch
        except ImportError: self.skipTest('torch not installed; live training dependency')
        a=torch.tensor([-.2,-.3,-.1],requires_grad=True)
        b=torch.tensor([-.5,-1.0,-.2],requires_grad=True)
        loss,_=training.loss_function([1,1],[0])([],[a,b])
        loss.backward()
        self.assertEqual(float(a.grad[0]),0.)
        self.assertLess(float(a.grad[1]),0.)
        self.assertGreater(float(b.grad[1]),0.)
        self.assertAlmostEqual(float(loss),-math.log(math.exp(-.4)/(math.exp(-.4)+math.exp(-1.2))),places=6)


if __name__=='__main__': unittest.main()
