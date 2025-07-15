"""
This script is for generating alignscore factuality metrics.

Last Updated: 7-14-2025
"""
from alignscore import AlignScore

scorer = AlignScore(
    model='roberta-base', 
    batch_size=32, 
    device="cpu",#'cuda:0', 
    ckpt_path='AlignScore-base.ckpt', 
    evaluation_mode='nli_sp'
)
score = scorer.score(
    contexts=[
        '''Preempts State laws to the extent that such laws are inconsistent with this Act, 
        except State law that provides additional protection from liability.  
        Specifies that this Act shall not be construed to supersede any Federal or State health or safety law. '''
        ], 
    claims=['''Preempts State laws to the extent . such laws are inconsistent with this Act except State . 
            that provides additional protection from liability .'''
        ]
)

print(score)