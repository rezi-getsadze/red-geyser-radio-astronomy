# Radio Interferometry & Spectral Analysis: Red Geyser Galaxies

This is the modeling and analysis pipeline for my senior thesis at NYU Abu Dhabi. I used VLBA data at 1.4 GHz and 4.8 GHz to figure out what's actually driving the gas outflows in "Red Geyser" galaxies, specifically looking for coronas, winds, or jets.

## The Files

* `analysis_logic/` - The math. Includes coordinate conversions for the antennas, automated spectral index ($\alpha$) calculations, and Gaussian fitting using `scipy.optimize`.
* `visualization_pipeline/` - Production scripts for the radio spectra. These handle the error propagation and multi-component model overlays for the final figures.
* `data_samples/` - The raw flux measurements for the target galaxies and my masterlist of 22 Green Valley AGN candidates.
* `docs/` - The full context. Includes the final thesis paper (**Getsadze_Thesis.pdf**), galaxy images, and the math proofs for the uncertainty calculations.

## Key Findings

* **Bimodal Outflows:** For example, `fittings4.py` showed that some galaxies (like MaNGA 1-43718) have bimodal profiles, suggesting the gas is being pushed by deflected winds rather than just a simple jet.
* **Spectral Slopes:** I used the spectral index to classify the emissions—detecting coronas where the slopes were flat and jets where they were steep.
* Overall, the results suggest that the outflows in Red Geysers can have different driving mechanisms - not simply jets, as often assumed in much of the literature. Each galaxy exhibits unique AGN activity depending on its specific conditions, implying that different mechanisms could be driving the outflows.

## Requirements
* Python 3.x
* NumPy, SciPy, Matplotlib, Folium
