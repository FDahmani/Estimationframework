# Estimation Framework

In this repository, the code and input files of my thesis, titled 'Obrit Estimation of Minor Jovian Satellites', can be found. The code forms a framework for the initial state estimation for small moons. The following files are found in the repository:

Observations: Raw observation files, taken from the NSDC website, sorted between inner and outer moons. 
ObservationsProcessed: Observation files that were uniformised to a Tudat-compatible format. 
Sim_ISE: code for the initial state estimation based on simulated perfect SPICE observations, known as the prefit in the thesis
Obs_ISE: code for the initial state estimation based on real astrometric observations
Obs_ISE_SimulatedSC: code for the initial state estimation based on real astrometric observations, with added simulated spacecraft observations for the preliminary analysis.

To run these python files, one needs to make sure that the 'jup344.bsp' kernel is placed in the folder. Furthermore, the processed observation files based on which you want to estimate the initial state of the body, should be moved to the folder 'ObservationsProcessed/CurrentProcess'.

