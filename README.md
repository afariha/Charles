This is the demo tool for the demo paper [ChARLES: Change-Aware Recovery of Latent Evolution Semantics in Relational Data]([url](https://dl.acm.org/doi/10.1145/3722212.3725089)), authored by by Shiyi He, Alexandra Meliou, Anna Fariha.


## 💡 Setup and Run Instructions

1. **Make sure you have `python3` and `pip3` installed on your device.**

2. **Create and activate a new virtual environment and upgrade pip:**

	```
	python3.9 -m venv .venv
	source .venv/bin/activate
	.venv/bin/python3.9 -m pip install --upgrade pip
	```

3. **Install the dependencies:**

	```
	pip3 install -r requirements.txt
	```

4. **Run the Streamlit app:**

	```
	streamlit run charles.py
	```
	
5. **Point to the two files `source.csv` and `target.csv` located in the `datasets` folder in the demo. Or use your own datasets!**

