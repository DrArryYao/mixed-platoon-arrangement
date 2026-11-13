"""Run the complete numerical archive; precomputed nonlinear results are reused.
For a fresh run, start in a copy of the archive without the supplied results/.
"""
import subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
for name in ['frequency_studies.py','nonlinear_studies.py','additional_studies.py','communication_checks.py','make_figures.py']:
    subprocess.run([sys.executable,str(here/name)],cwd=here,check=True)
print('All numerical studies and figures completed.')
