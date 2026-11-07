"""Sequential Hough -> CNN -> symmetry/plain/denoising AEs -> synthetic PINN."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def run(output,cnn_epochs=3,ae_epochs=2,pinn_steps=1000):
    output.mkdir(parents=True,exist_ok=False)
    stages=[('hough','hough.py',[]),
        ('cnn','neural_train.py',['--kind','cnn','--epochs',str(cnn_epochs)]),
        ('symmetry_ae','neural_train.py',['--kind','symmetry_ae','--epochs',str(ae_epochs)]),
        ('plain_ae','neural_train.py',['--kind','plain_ae','--epochs',str(ae_epochs)]),
        ('denoising_ae','neural_train.py',['--kind','denoising_ae','--epochs',str(ae_epochs)]),
        ('pinn','pinn.py',['--steps',str(pinn_steps)])]
    reports={}
    with (output/'terminal_log.txt').open('x',encoding='utf-8') as log:
        for name,script,args in stages:
            heading='\n'+'='*72+'\nSTART '+name+'\n'+'='*72+'\n'
            print(heading,flush=True);log.write(heading);log.flush()
            command=[sys.executable,'-u','-B',str(ROOT/script),'--output',str(output/name),*args]
            if name!='pinn':command+=['--data',str(ROOT/'prepared_v1')]
            with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace') as process:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            if code:raise RuntimeError(f'{name} failed with exit code {code}; see {output / "terminal_log.txt"}')
            reports[name]=json.loads((output/name/'metrics.json').read_text())
            (output/'comparison.json').write_text(json.dumps(reports,indent=2))
    subprocess.run([sys.executable,'-B',str(ROOT/'compare_methods.py'),'--run',str(output)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'symmetry_probe.py'),'--data',str(ROOT/'prepared_v1'),'--run',str(output)],cwd=ROOT,check=True)
    print('\nCOMPLETE. Validation comparisons:',flush=True)
    for name,result in reports.items():
        if 'validation' in result:
            m=result['validation']; print(f"{name:15s} AP={m['average_precision']:.4f} ROC-AUC={m['roc_auc']:.4f}",flush=True)
    print('PINN is a separate synthetic PDE benchmark. No observational test set was scored.',flush=True)
    print('All outputs:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--cnn-epochs',type=int,default=3)
    p.add_argument('--ae-epochs',type=int,default=2)
    p.add_argument('--pinn-steps',type=int,default=1000)
    a=p.parse_args()
    destination=a.output or ROOT/'runs'/datetime.now().strftime('methods_%Y%m%d_%H%M%S')
    run(destination.resolve(),a.cnn_epochs,a.ae_epochs,a.pinn_steps)
