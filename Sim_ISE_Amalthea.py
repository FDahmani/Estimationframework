# Load required standard modules
import sys
sys.path.insert(0, '/home/fdahmani/tudatcompile/tudat-bundle/cmake-build-release/tudatpy')
import numpy as np
from matplotlib import pyplot as plt 
import os
import csv
import datetime
import matplotlib.dates as mdates
import math
import shutil

# Load required tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion, time_conversion, frame_conversion
from tudatpy.kernel.math import interpolators
from tudatpy import util

# Load spice kernels
spice.load_standard_kernels()
spice.load_kernel('jup344.bsp')

# Define temporal scope of the simulation - equal to the time JUICE will spend in orbit around Jupiter
simulation_start_epoch = time_conversion.julian_day_to_seconds_since_epoch(time_conversion.calendar_date_to_julian_day(datetime.datetime(1990,1,1,12,0,0)))
simulation_end_epoch = time_conversion.julian_day_to_seconds_since_epoch(time_conversion.calendar_date_to_julian_day(datetime.datetime(2003,1,1,12,0,0)))
simulation_duration = simulation_end_epoch - simulation_start_epoch


### Create the Environment
# Create default body settings for selected celestial bodies
jovian_moons_to_create = ['Io', 'Europa', 'Ganymede', 'Callisto','Amalthea']
planets_to_create = ['Jupiter', 'Saturn']
stars_to_create = ['Sun']
bodies_to_create = np.concatenate((jovian_moons_to_create, planets_to_create, stars_to_create))

# Create default body settings for bodies_to_create, with 'Jupiter'/'ECLIPJ2000'
# as global frame origin and orientation.
global_frame_origin = 'Jupiter'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)


# Define bodies that are propagated, and their central bodies of propagation
bodies_to_propagate = ['Amalthea']
central_bodies = ['Jupiter']

### Ephemeris Settings Moons ###
for moon in bodies_to_propagate:
    # Apply tabulated ephemeris settings
    body_settings.get(moon).ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    body_settings.get(moon).ephemeris_settings,
    simulation_start_epoch - 12*7200.0,
    simulation_end_epoch + 12*7200.0,
    time_step=6* 60.0)

#body_settings.get("Jupiter").gravity_field_settings.gravitational_parameter = 1.266865341960128e+17-9.3138e+9
# Create system of selected bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


### Create Propagator Settings
### Acceleration Settings ###
# Dirkx et al. (2016) - restricted to second degree
love_number_moons = 0.3
dissipation_parameter_moons = 0.015
q_moons = love_number_moons / dissipation_parameter_moons
# Lari (2018)
mean_motion_amalthea = 722.6312 * (math.pi / 180) * 1 / constants.JULIAN_DAY
mean_motion_io = 203.49 * (math.pi / 180) * 1 / constants.JULIAN_DAY

# Dirkx et al. (2016) - restricted to second degree
love_number_jupiter = 0.38
dissipation_parameter_jupiter= 1.1E-5
q_jupiter = love_number_jupiter / dissipation_parameter_jupiter

# Lainey et al. (2009)
tidal_frequency_io = 23.3 # rad.day-1
spin_frequency_jupiter = math.pi/tidal_frequency_io + mean_motion_io

# Calculate all required time lags associated with the individual tides
time_lag_amalthea = 1 / mean_motion_amalthea * np.arctan(1 / q_moons)
time_lag_jupiter_amalthea = 1/(spin_frequency_jupiter - mean_motion_amalthea) * np.arctan(1 / q_jupiter)


time_lag_dict = {'Amalthea': (time_lag_amalthea, time_lag_jupiter_amalthea),}


acceleration_settings_moons = dict()

for idx, moon in enumerate(bodies_to_propagate):
    other_moons = np.delete(np.array(bodies_to_propagate), idx)
    acceleration_settings_moon = {
        'Jupiter': [propagation_setup.acceleration.spherical_harmonic_gravity(8, 0)],
        'Io': [propagation_setup.acceleration.point_mass_gravity()],
        'Ganymede': [propagation_setup.acceleration.point_mass_gravity()],
        'Europa': [propagation_setup.acceleration.point_mass_gravity()],
        'Callisto': [propagation_setup.acceleration.point_mass_gravity()],
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        'Saturn': [propagation_setup.acceleration.point_mass_gravity()]        
    }
    acceleration_settings_moons[moon] = acceleration_settings_moon

acceleration_settings = acceleration_settings_moons
# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies)

# Define initial state
initial_states = list()
for body in bodies_to_propagate:
    initial_states.append(spice.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name='Jupiter',
        reference_frame_name=global_frame_orientation,
        aberration_corrections='none',
        ephemeris_time=simulation_start_epoch))
#initial_states = np.concatenate(initial_states)
initial_states = [-1.71358285e+08,  6.02868819e+07,  8.61709429e+05 ,-8.70255003e+03, -2.49379545e+04, -1.02136012e+03]
### Integrator Settings ###
# Use fixed step-size integrator (RKDP8) with fixed time-step of 30 minutes
# Create integrator settings
time_step_sec = 60.0 * 6
integrator_settings = propagation_setup.integrator. \
    runge_kutta_fixed_step_size(initial_time_step=time_step_sec,
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

### Termination Settings ###
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Define Keplerian elements of the moons as dependent variables
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Amalthea', 'Jupiter')]

### Propagator Settings ###
propagator_settings = propagation_setup.propagator. \
    translational(central_bodies=central_bodies,
                  acceleration_models=acceleration_models,
                  bodies_to_integrate=bodies_to_propagate,
                  initial_states=initial_states,
                  initial_time=simulation_start_epoch,
                  integrator_settings=integrator_settings,
                  termination_settings=termination_condition,
                  output_variables=dependent_variables_to_save)
propagator_settings.processing_settings.results_save_frequency_in_steps = 30

## Orbital Estimation
### Create Link Ends for the Moons

link_ends_amalthea = dict()
link_ends_amalthea[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Amalthea')
link_definition_amalthea = estimation_setup.observation.LinkDefinition(link_ends_amalthea)

link_definition_dict = {
    'Amalthea': link_definition_amalthea,
}


### Observation Model Settings
position_observation_settings = [estimation_setup.observation.cartesian_position(link_definition_amalthea)]

### Observation Simulation Settings
# Define epochs at which the ephemerides shall be checked
observation_times = np.arange(simulation_start_epoch + 12*7200.0, simulation_end_epoch - 12*7200.0, 0.7 * 3600)

# Create the observation simulation settings per moon
observation_simulation_settings = list()
for moon in link_definition_dict.keys():
    observation_simulation_settings.append(estimation_setup.observation.tabulated_simulation_settings(
        estimation_setup.observation.position_observable_type,
        link_definition_dict[moon],
        observation_times,
        reference_link_end_type=estimation_setup.observation.observed_body))


### Simulate Ephemeris' States of Satellites
# Create observation simulators
ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
    position_observation_settings, bodies)
# Get ephemeris states as ObservationCollection
print('Checking ephemerides...')
ephemeris_satellite_states = estimation.simulate_observations(
    observation_simulation_settings,
    ephemeris_observation_simulators,
    bodies)


### Define Estimable Parameters
"""
Given the problem at hand - minimising the discrepancy between the SPICE ephemeris and the states of the moons when propagated under the influence of the above-defined accelerations - we are mainly interested in an improved initial state of the moons. We will thus restrict the set of estimable parameters to the moons' initial states.
"""

parameters_to_estimate_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
#parameters_to_estimate_settings.append(estimation_setup.parameter.gravitational_parameter("Jupiter"))
parameters_to_estimate = estimation_setup.create_parameter_set(parameters_to_estimate_settings, bodies)
original_parameter_vector = parameters_to_estimate.parameter_vector


### Perform the Estimation
print('Running propagation...')
with util.redirect_std():
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate,
                                               position_observation_settings, propagator_settings)


convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = 5)
# Create input object for the estimation
estimation_input = estimation.EstimationInput(ephemeris_satellite_states,convergence_checker=convergence_checker)
# Set methodological options
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
# Perform the estimation
print('Performing the estimation...')
print(f'Original initial states: {original_parameter_vector}')


with util.redirect_std(redirect_out=False):
    estimation_output = estimator.perform_estimation(estimation_input)
initial_states_updated = parameters_to_estimate.parameter_vector
print('Done with the estimation...')
print(f'Updated initial states: {initial_states_updated}')
print('start')
print(estimation_output.correlations)
print(estimation_output.design_matrix)
print('stop')

#Load data
simulator_object = estimation_output.simulation_results_per_iteration[-1]
state_history = simulator_object.dynamics_results.state_history
dependent_variable_history = simulator_object.dynamics_results.dependent_variable_history


##Correlations
correlations = estimation_output.correlations
design_matrix = estimation_output.normalized_design_matrix
# print(correlations)
# print(design_matrix)

# plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
# plt.colorbar()

# plt.title("Correlation Matrix")
# plt.xlabel("Index - Estimated Parameter")
# plt.ylabel("Index - Estimated Parameter")

# plt.tight_layout()
# plt.show()

# #Design matrix
# plt.imshow(np.abs(design_matrix), aspect='auto', interpolation='none')
# plt.colorbar()

# plt.title("Design Matrix")
# plt.xlabel("Index - Estimated Parameter")
# plt.ylabel("Index - Estimated Parameter")


# plt.tight_layout()
# plt.show()

### Ephemeris Kepler elements ####
# Initialize containers
ephemeris_state_history = dict()
ephemeris_keplerian_states = dict()
jupiter_gravitational_parameter = bodies.get('Jupiter').gravitational_parameter
k = 0
#Loop over the propagated states and use the IMCEE ephemeris as benchmark solution
for epoch in state_history.keys():
    ephemeris_state = list()
    keplerian_state = list()
    for moon in bodies_to_propagate:
        ephemeris_state_temp = spice.get_body_cartesian_state_at_epoch(
            target_body_name=moon,
            observer_body_name='Jupiter',
            reference_frame_name='ECLIPJ2000',
            aberration_corrections='none',
            ephemeris_time=epoch)
        ephemeris_state.append(ephemeris_state_temp)
        keplerian_state.append(element_conversion.cartesian_to_keplerian(ephemeris_state_temp,
                                                                            jupiter_gravitational_parameter))

    ephemeris_state_history[epoch] = np.concatenate(np.array(ephemeris_state))
    ephemeris_keplerian_states[epoch] = np.concatenate(np.array(keplerian_state))
    k+=1
state_history_difference = np.vstack(list(state_history.values())) - np.vstack(list(ephemeris_state_history.values()))
position_difference = {'Amalthea': state_history_difference[:, 0:3]}
velocity_difference = {'Amalthea': state_history_difference[:, 3:6]}
propagation_kepler_elements = np.vstack(list(dependent_variable_history.values()))
kepler_difference = np.vstack(list(dependent_variable_history.values())) - np.vstack(list(ephemeris_keplerian_states.values()))


# get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the current time as a string
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Create a new folder with the current timestamp as the name
folder_name = os.path.join(script_dir, 'results_prefit', f"plots_{current_time}")
os.makedirs(folder_name, exist_ok=True)

rsw_states = [] 
for k in range(len(state_history_difference)):
    if k == 0:
        continue

    rm_1 = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(state_history.values()))[k])

    ab1 = rm_1@np.vstack(list(state_history.values()))[k, 0:3]
    ab2 = rm_1@np.vstack(list(ephemeris_state_history.values()))[k, 0:3]
    rsw_states.append(ab1-ab2)

time2plt = list()
epochs_julian_seconds = np.vstack(list(state_history.keys()))
for epoch in epochs_julian_seconds:
    epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
    time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))

# #RSW
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,0] * 1E-3,
         label=r'Amalthea0 ($i=1$)', c='#A50034')
# ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,1] * 1E-3,
#          label=r'Amalthea1 ($i=1$)', c='#0076C2')
# ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,2] * 1E-3,
#          label=r'Amalthea2 ($i=1$)', c='#EC6842')
ax1.set_title(r'Difference in Position')
#ax1.plot(time2plt[1:], np.linalg.norm( np.vstack(rsw_states), axis=1) * 1E-3,
#         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Difference [km]')
ax1.legend()
plt.show()

# #cartesian
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.scatter(np.vstack(list(state_history.values()))[:200,0]* 1E-3, np.vstack(list(state_history.values()))[:200,1]* 1E-3,
#          label=r'Propagated ($i=1$)', c='#A50034')
# ax1.scatter(np.vstack(list(ephemeris_state_history.values()))[:1000,0]* 1E-3, (np.vstack(list(ephemeris_state_history.values()))[:1000,1])* 1E-3,
#          label=r'Spice ($i=1$)', c='#0076C2')
# # ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
# #          label=r'Amalthea2 ($i=1$)', c='#EC6842')
# ax1.set_title(r'Difference in Position')
# # ax1.plot(time2plt, np.linalg.norm(position_difference['Amalthea'], axis=1) * 1E-3,
# #          label=r'Amalthea ($i=1$)', c='#A50034')
# ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
# ax1.xaxis.set_minor_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
# ax1.set_ylabel(r'Difference [km]')
# ax1.legend()
# plt.show()


#original
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0 ($i=1$)', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1 ($i=1$)', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2 ($i=1$)', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(position_difference['Amalthea'], axis=1) * 1E-3,
         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Difference [km]')
ax1.legend()
plt.savefig(os.path.join(folder_name, "plot1.png"))
plt.show()

#velocity
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0 ($i=1$)', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1 ($i=1$)', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2 ($i=1$)', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(velocity_difference['Amalthea'], axis=1),
         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Difference [m/s]')
ax1.legend()
plt.savefig(os.path.join(folder_name, "plot2.png"))
plt.show()


#Kepler

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0 ($i=1$)', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1 ($i=1$)', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2 ($i=1$)', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, propagation_kepler_elements[:,0]* 1E-3,
         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Semi-major axis [km]')
ax1.legend()
plt.savefig(os.path.join(folder_name, "plot3.png"))
plt.show()

#Kepler

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0 ($i=1$)', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1 ($i=1$)', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2 ($i=1$)', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, kepler_difference[:,0]* 1E-3,
         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'Semi-major axis [km]')
ax1.legend()
plt.savefig(os.path.join(folder_name, "plot4.png"))
plt.show()



obstime2plt = list()
obs_julian_seconds = np.vstack(observation_times)
for obs in obs_julian_seconds:
    obs_days = constants.JULIAN_DAY_ON_J2000 + obs / constants.JULIAN_DAY
    obstime2plt.append(time_conversion.julian_day_to_calendar_date(obs_days))

residuals = estimation_output.final_residuals
residual_x = residuals[0::3]
residual_y = residuals[1::3]
residual_z = residuals[2::3]
plt.plot(observation_times,residual_x, label='x')
plt.plot(observation_times,residual_y, label='y')
plt.plot(observation_times,residual_z, label='z')
plt.legend()
plt.grid()
plt.savefig(os.path.join(folder_name, "plot5.png"))
plt.show()

rms_list = []
for i in range(len(residual_x)):
        # Calculate the squares of the i-th elements from each list
    squared_x = residual_x[i]**2
    squared_y = residual_y[i]**2
    squared_z = residual_z[i]**2

    # Calculate the mean of the squared values
    mean_squared = (squared_x + squared_y + squared_z) / 3

    # Calculate the RMS value by taking the square root of the mean squared value
    rms = math.sqrt(mean_squared)

    # Append the RMS value to the list
    rms_list.append(rms*1E-3)
# plt.plot(observation_times,rms_list, label='rms')
# plt.legend()
# plt.grid()
# plt.xlabel('time [s since J2000]')
# plt.ylabel('rms of the position')
# plt.savefig(os.path.join(folder_name, "plot6.png"))
# plt.show()


fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

ax1.set_title(r'RMS of position residual')
ax1.plot(obstime2plt, rms_list,
         label=r'Amalthea ($i=1$)', c='#A50034')
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax1.set_ylabel(r'RMS of position residual [km]')
ax1.legend()
plt.savefig(os.path.join(folder_name, "plot6.png"))
plt.show()

script_name = os.path.basename(__file__)
shutil.copy(script_name, os.path.join(folder_name, script_name))

print('Done')