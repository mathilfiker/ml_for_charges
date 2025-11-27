Content of the folder:
- Partial charges obtained with AM1-BCC, ESP, RESP, 1-shot prediction, and Boltzmann percentile charge assignment.
- Molecules the analyses were perfomed on, in .sdf and in .xyz format.
- Parity plots as shown in Fig.5(a) of the main text.
- Numpy arrays containing calculated free energies with each assignment method.
- Numpy arrays containing errors from the AHFE calculations for each assignment method.
- **compare_charges.py** to reproduce the heatmap in Fig.5(b), here already saved as heatmap.png .
- **example_transformation.py** to generate an AHFE transformation with custom charges. It outputs *mobley_36119.json*
- **parity_plots.py** to reproduce the aforementioned parity plots.
