# AX-1
This is a classification sample of actors voice by TF.

The voice used is not included due to copyright issues.

## How to use
0.install required modules

1.prepare datasets(.wav)

2.python main.py --train=../X(your train datasets directory) --test=../Y(your test datasets directory)

### required
tensorflow

tqdm

argparse

AudioSegment

pip install git+https://github.com/jSm449g4d/AR9



### Directory structure example
 
Train/

  ┣A(actor)/
  
  │ ┣1(voice).wav
    
  │ ┣2.wav
    
  │ ┣3.wav
    
  │ ┗4.wav
  
  ┣B/
  
  ┗C/
