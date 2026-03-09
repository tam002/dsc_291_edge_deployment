## How to Use

1. Follow previous README.md instructions

2. Make sure to make a Virtual Enviornment (Anything above 3.10+ should be good)

```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. Run the following commands (I believe these are the only libraries necessary other than Step 1):

```bash
pip install streamlit requests sseclient-py psutil
```

4. To start the UI, do the following:

```bash
streamlit run demo_ui.py
```

UI Details:
1. Select the desired quantization level of the model and click **Load Model**

2. Enter a prompt inside the **Enter Prompt** Box, and then click **Run Prompt**

3. The Model Output will appear along with the Metrics

Quick Notes:

- I did not use the same code for calculating the metrics in `run_sweep.py`, so if some of the metrics are not matching / incorrect, that might be why and it would be worth looking into

    - For example, power seems to be set to 5.0 W at default

- Might be worthwhile to include some graphs for the metrics, but I'm not sure what the best way is to include it in the demo 

- There is an inconsistency issue where when switching models or clearing the output, some of the metrics will go back to its default values (it'll be obvious when you see it)

    - It's a pretty easy fix, I can do it if it's a big issue for the demo
