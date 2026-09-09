"""Unified command line. Analysis commands do not import Tinker or torch."""
import argparse
import json
import math
from .data import read_rows,audit


def main():
    p=argparse.ArgumentParser(prog='tinker-uq')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit'); a.add_argument('--data',required=True)
    m=sub.add_parser('models',help='List models available to your Tinker account')
    t=sub.add_parser('train',help='Train q and qa adapters and export all split scores')
    for key in ('data','model','out'): t.add_argument('--'+key,required=True)
    t.add_argument('--seed',type=int,default=0); t.add_argument('--epochs',type=int,default=1)
    t.add_argument('--rank',type=int,default=16); t.add_argument('--batch-size',type=int,default=8)
    t.add_argument('--lr',type=float,default=1e-4); t.add_argument('--max-tokens',type=int,default=4096)
    t.add_argument('--score-window',type=int,default=256,help='in-flight logprob requests; affects speed only')
    i=sub.add_parser('iclr'); i.add_argument('--predictions',required=True); i.add_argument('--out',required=True)
    i.add_argument('--seed',type=int,default=0); i.add_argument('--bootstrap',type=int,default=1000)
    f=sub.add_parser('fit-conformal',help='Fit fixed partitions and cell quantiles; no test evaluation')
    f.add_argument('--predictions',required=True); f.add_argument('--out',required=True)
    f.add_argument('--stage',choices=['base','trained'],default='trained')
    f.add_argument('--legend',choices=['normal','reversed'],default='normal')
    f.add_argument('--anchor',choices=['q','qa'],default='qa')
    f.add_argument('--partition-split',choices=['train','val'],default='train')
    f.add_argument('--candidate-policy',choices=['error','sample'],default='error')
    f.add_argument('--seed',type=int,default=0)
    f.add_argument('--k',nargs='+',type=int,default=[1,2,4,8,16])
    f.add_argument('--alpha',type=float,default=.1)
    e=sub.add_parser('evaluate-conformal',help='Evaluate previously fitted calibrators')
    for key in ('predictions','calibration','out'): e.add_argument('--'+key,required=True)
    d=sub.add_parser('demo',help='Generate simulated scores without credentials')
    d.add_argument('--out',required=True); d.add_argument('--seed',type=int,default=0)
    d.add_argument('--n',type=int,default=250)
    args=p.parse_args()
    if args.command=='audit':
        print(json.dumps(audit(read_rows(args.data)),indent=2))
    elif args.command=='models':
        import tinker
        print('\n'.join(x.model_name for x in tinker.ServiceClient().get_server_capabilities().supported_models))
    elif args.command=='train':
        if min(args.epochs,args.rank,args.batch_size,args.max_tokens)<=0 or not math.isfinite(args.lr) or args.lr<=0:
            p.error('Training parameters must be positive and finite')
        from .training import run
        run(args,read_rows(args.data))
    elif args.command=='iclr':
        if args.bootstrap<1: p.error('--bootstrap must be positive')
        from .analysis import run
        run(args)
    elif args.command=='fit-conformal':
        from .conformal import fit
        fit(args)
    elif args.command=='evaluate-conformal':
        from .conformal import evaluate
        evaluate(args)
    elif args.command=='demo':
        from .demo import generate
        generate(args.out,args.seed,args.n)
