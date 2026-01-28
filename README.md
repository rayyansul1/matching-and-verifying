Rayyan Sulaiman ID: 54761895

Python 3
Dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt# matching-and-verifying

In windows powershell:
First activate virtual environment
.\.venv\Scripts\Activate.ps1

Matcher:
python stable.py match data\example.in | Set-Content -Encoding utf8 data\my_match.out
To check output:
type data\my_match.out


Verifier:
python stable.py verify data/example.in data/my_match.out

Scalability Benchmark:
python stable.py bench
dir results
start results\matcher_runtime.png
start results\verifier_runtime.png

Graphs and solutions are located in results folder
