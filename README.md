# net-sort

Implemention of LSTM sequence-to-sequence based combinatorics model using only numpy and scipy

## Models Used
- [X] Encoder-Decoder architecture
- [X] PointerNet architecture described [here](https://arxiv.org/pdf/1506.03134.pdf)

## Instructions to run

- Install dependencies
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

- Modify variables in `main.py`
- Run `main.py`:
```bash
cd src
python main.py
```

## License
MIT
