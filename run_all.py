"""Visible terminal walkthrough: preprocessing reports, loader demos, tests, baseline."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def section(title):
    print('\n'+'='*72+'\n'+title+'\n'+'='*72,flush=True)

def run(script,*arguments):
    subprocess.run([sys.executable,'-B',str(ROOT/script),*arguments],cwd=ROOT,check=True)

if __name__=='__main__':
    section('1. FULL-DATA PREPROCESSING RESULTS (existing validated cache)')
    report=json.loads((ROOT/'prepared_v1'/'validation.json').read_text())
    print(json.dumps({k:report[k] for k in ['status','total_images','all_shapes','all_dtype','all_finite','split_counts','exact_pixel_leakage','normalization']},indent=2),flush=True)
    section('2. ORIGINAL PIXELS + ORDINARY SAMPLING')
    run('demo.py','--raw','--batches','1')
    section('3. NORMALIZED BATCH + ORDINARY SAMPLING')
    run('demo.py','--batches','1')
    section('4. NORMALIZATION + BALANCED SAMPLING + AUGMENTATION')
    run('demo.py','--balanced','--augment','--batches','1')
    section('5. VALIDATION BATCH (no augmentation or resampling)')
    run('demo.py','--split','validation','--batches','1')
    section('6. AUTOMATED TESTS')
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-v','-p','test_data.py'],cwd=ROOT,check=True)
    section('7. QUALITY FLAGS (existing audit; no automatic exclusions)')
    print((ROOT/'prepared_v1'/'quality_report.json').read_text(),flush=True)
    section('8. TRAIN A FRESH IMAGE-STATISTICS BASELINE')
    destination=ROOT/'runs'/datetime.now().strftime('visible_%Y%m%d_%H%M%S')
    run('baseline.py','--data',str(ROOT/'prepared_v1'),'--output',str(destination))
    section('COMPLETE')
    print('Baseline output:',destination,flush=True)
    print('Image preview:',ROOT/'prepared_v1'/'positive_negative.png',flush=True)
    print('The supplied test set has NOT been evaluated. See EXPERIMENTS.md for Hough, CNN, AEs and PINN runs.',flush=True)
