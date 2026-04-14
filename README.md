# geoflux

A Python package for computing spatially explicit, weighted flux footprints from eddy covariance tower time series.

## Overview

`geoflux` integrates two components:

1. **Flux footprint modeling** — implements the [Kljun et al. (2015)](https://doi.org/10.5194/gmd-8-3695-2015) 2-dimensional, physically-based flux footprint prediction (FFP) climatology model to estimate the upwind source area contributing to eddy covariance measurements.

2. **Reference ET weighting** — downloads hourly meteorological drivers (temperature, radiation, vapor pressure, wind speed) from [NLDAS-2](https://ldas.gsfc.nasa.gov/nldas/v2/forcing) and computes ASCE Standardized Penman-Monteith reference evapotranspiration (ETr). Hourly ETr values are used to weight hourly footprint predictions, which are then aggregated to daily or monthly climatologies.

## Sources

This package is a rewrite code from https://github.com/Open-ET/flux-data-footprint. Please refer to the github link for further information.

## Installation

```bash
pip install git+https://github.com/tuyenhavan/geoflux.git
```

## Usage

See `notebook.ipynb` for example usage.
