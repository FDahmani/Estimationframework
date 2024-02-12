# Load required standard modules
import sys
import numpy as np
from matplotlib import pyplot as plt 
import os
import csv
import datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import math
import shutil
from scipy import stats

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

def moon_name (moon):
    IDDict = {
        1: 'Io',
        2: 'Europa',
        3: 'Ganymede',
        4: 'Callisto',
        5: 'Amalthea',
        6: 'Himalia',
        7: 'Elara',
        8: 'Pasiphae',
        9: 'Sinope',
        10: 'Lysithea',
        11: 'Carme',
        12: 'Ananke',
        13: 'Leda',
        14: 'Thebe',
        15: 'Adrastea',
        16: 'Metis',
        17: 'Callirrhoe',
        18: 'Themisto',
        19: '519',
        20: 'Taygete',
        'Amalthea': 'Amalthea',
        'Jupiter': 'Jupiter'
    }
    ID = IDDict[moon]
    return ID

def observatory_info (Observatory): #Positive to north and east
    if len(Observatory) == 2:                   #Making sure 098 and 98 are the same
        Observatory = '0' + Observatory
    elif len(Observatory) == 1:                   #Making sure 098 and 98 are the same
        Observatory = '00' + Observatory
    with open('Observatories.txt', 'r') as file:    #https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
        lines = file.readlines()
        for line in lines[1:]:  # Ignore the first line
            columns = line.split()
            if columns[1] == Observatory:
                longitude = float(columns[2])
                latitude = float(columns[3])
                altitude = float(columns[4])
                return np.deg2rad(longitude),  np.deg2rad(latitude), altitude
        print('No matching Observatory found')

# Load spice kernels
spice.load_standard_kernels()
spice.load_kernel('jup344.bsp')

G = constants.GRAVITATIONAL_CONSTANT

# Set start and end epochs
t_0 =   time_conversion.DateTime(1949,7,15,12,0,0).epoch()
t_end = time_conversion.DateTime(2023,1,1,12,0,0).epoch()
t_end_before = time_conversion.DateTime(1900,1,1,12,0,0).epoch()
time_difference = t_end - t_0


folder_path = 'ObservationsProcessed/CurrentProcess'
raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

#Set up environment
bodies_to_create = ["Sun",'Saturn', "Earth",'Jupiter', 'Io', 'Europa','Ganymede','Callisto','Himalia','Elara']



# Create default body settings
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

#Add bodies not covered by kernels
body_settings.get("Himalia").gravity_field_settings = environment_setup.gravity_field.central(1.42598087e+08)             #Paper determining Himalia mass
body_settings.get("Elara").gravity_field_settings = environment_setup.gravity_field.central(869227691196372000 *G)

#For propagated moons, voor initial state, Nan lijst veranderen
body_settings.get('Elara').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(environment_setup.ephemeris.direct_spice("Jupiter",global_frame_orientation,"Elara"),t_0-12*7200,t_end+12*7200,20*60,interpolators.lagrange_interpolation(4))         #Check number of points used in interpolating


# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)




# Create radiation pressure settings
reference_area_radiation = 83500*83500*np.pi
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Jupiter"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
# Add the radiation pressure interface to the environment
environment_setup.add_radiation_pressure_interface(bodies, "Elara", radiation_pressure_settings)

obstimes = []

observation_set_list = []
observation_settings_list = []
observation_simulation_settings = []
noise_level = []

weights_plot = []
weight_plot_time = []
weight_number = []
bias_applied = []

#Start loop over every csv in folder
for file in raw_observation_files:
    arc_start_times_local = []
    bias_values_local = []
    
    #Reading information from file name
    string_to_split = file.split("/")[-1]
    split_string = string_to_split.split("_")

    Moon = int(split_string[0])
    Observatory = split_string[1]
    #rms = float(split_string[3][:-4])
    # Define the position of the observatory on Earth
    observatory_longitude, observatory_latitude, observatory_altitude = observatory_info (Observatory)

    # Add the ground station to the environment
    environment_setup.add_ground_station(
        bodies.get_body("Earth"),
        str(Observatory),
        [observatory_altitude, observatory_latitude, observatory_longitude],
        element_conversion.geodetic_position_type)
    
    observatory_ephemerides = environment_setup.create_ground_station_ephemeris(
        bodies.get_body("Earth"),
        str(Observatory),
        )
    



    # Define the duration of each arc in seconds
    arc_duration = 6 * 60 * 60  


    ###### Weight per arc
    #Reading observational data from file
    ObservationList = []
    Timelist = []
    uncer = []
    with open(file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) 

        arc_times = []
        arc_uncertainties_ra = []
        arc_uncertainties_dec = []

        for row in csv_reader:
            time = float(row[0])
            Timelist.append(time)
            obstimes.append(time)
            ObservationList.append(np.asarray([float(row[1]), float(row[2])]))

            # Uncertainties
            uncer.append(np.asarray([float(row[3]), float(row[4])]))
            uncertainty_ra = float(row[3])
            uncertainty_dec = float(row[4])

            # Arc cut-off
            if arc_times and (time - arc_times[0] > arc_duration):
                # Calculate and append average uncertainties for the completed arc
                avg_uncertainty_ra = sum(arc_uncertainties_ra) / len(arc_uncertainties_ra)
                avg_uncertainty_dec = sum(arc_uncertainties_dec) / len(arc_uncertainties_dec)
                n = len(arc_uncertainties_ra)
                noise_level.extend([avg_uncertainty_ra*np.sqrt(n), avg_uncertainty_dec*np.sqrt(n)] * n)
                weights_plot.append(np.power(avg_uncertainty_ra, -2))
                weight_plot_time.append(arc_times[0]/ (constants.JULIAN_DAY*365.25)+ 2000)
                weight_number.append(n)
                bias_applied.append(avg_uncertainty_ra)
                arc_start_times_local.append(arc_times[0]-60)
                bias_values_local.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))

                # Reset arc lists
                arc_times = []
                arc_uncertainties_ra = []
                arc_uncertainties_dec = []

            # Append current observation to the arc
            arc_times.append(time)
            arc_uncertainties_ra.append(uncertainty_ra)
            arc_uncertainties_dec.append(uncertainty_dec)

    # Final arc
    if arc_uncertainties_ra:
        avg_uncertainty_ra = sum(arc_uncertainties_ra) / len(arc_uncertainties_ra)
        avg_uncertainty_dec = sum(arc_uncertainties_dec) / len(arc_uncertainties_dec)
        n = len(arc_uncertainties_ra)
        noise_level.extend([avg_uncertainty_ra*np.sqrt(n), avg_uncertainty_dec*np.sqrt(n)] * n)
        weights_plot.append(np.power(avg_uncertainty_ra*np.sqrt(n), -2))
        weight_plot_time.append(arc_times[0]/ (constants.JULIAN_DAY*365.25)+ 2000)
        weight_number.append(n)
        
        bias_applied.append(avg_uncertainty_ra)
        arc_start_times_local.append(arc_times[0]-60)
        bias_values_local.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))
    
    angles = ObservationList
    times = Timelist

    
    # Define link ends
    link_ends = dict()                  #To change
    link_ends[observation.transmitter] = observation.body_origin_link_end_id(moon_name(Moon-500))
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", str(Observatory))
    link_definition = observation.LinkDefinition(link_ends)


    # Create observation set 
    observation_set_list.append( estimation.single_observation_set(
        observation.angular_position_type, 
        link_definition,
        angles,
        times, 
        observation.receiver 
    ))

    # Create observation settings for each link/observable                                                                                          
    bias = observation.arcwise_absolute_bias(arc_start_times_local,bias_values_local,observation.receiver)
    if 'jo0010' in file:                                                                                         
        observation_settings_list.append(observation.angular_position(link_definition,bias_settings=bias))     #Extra input, zie slack
    else:
        observation_settings_list.append(observation.angular_position(link_definition))                             

observations = estimation.ObservationCollection( observation_set_list )

## Set up the propagation
# Define bodies that are propagated
bodies_to_propagate = ['Elara']

# Define central bodies of propagation
central_bodies = ["Jupiter"]


accelerations_settings_common = dict(
    Sun=[
        propagation_setup.acceleration.point_mass_gravity(),
    ],
    Jupiter=[
        propagation_setup.acceleration.spherical_harmonic_gravity(8, 0),
    ],
    Saturn=[
        propagation_setup.acceleration.point_mass_gravity(),

    ],
    Io=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)
    ],
    Europa=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)
    ],
    Ganymede=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)
    ],
    Callisto=[
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2)
    ],
    Himalia=[
        propagation_setup.acceleration.point_mass_gravity()  
    ])


# Create global accelerations dictionary
acceleration_settings = {"Elara": accelerations_settings_common}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)


### Define the initial state
"""
Realise that the initial state of the spacecraft always has to be provided as a cartesian state - i.e. in the form of a list with the first three elements representing the initial position, and the three remaining elements representing the initial velocity.
Within this example, we will make use of the `keplerian_to_cartesian_elementwise()` function  - included in the `element_conversion` module - enabling us to convert an initial state from Keplerian elements to a 6x1 cartesian vector.
"""

# Set the initial state of the vehicle
full_initial_state = []
for i in bodies_to_propagate:
    initial_state = [-3.45929377e+09, -9.11227748e+09,  4.87269592e+09 , 3.14098836e+03, -1.39056977e+03 , 7.93367480e+02 ]#1949
    full_initial_state.append(initial_state)
full_initial_state = np.concatenate(full_initial_state)



### Create the integrator settings
"""
# Use fixed step-size integrator (RKDP8) with fixed time-step of 180 minutes
"""

time_steps = [180]
difference_list = []
for time_step in time_steps:
    print(time_step)
    # Create numerical integrator settings
    time_step_sec = time_step * 60.0
    integrator_settings = propagation_setup.integrator. \
        runge_kutta_fixed_step_size(initial_time_step=time_step_sec,
                                    coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)


    ### Create the propagator settings
    # Create termination settings
    #termination_condition = propagation_setup.propagator.time_termination(t_end)
    termination_condition = propagation_setup.propagator.non_sequential_termination(
    propagation_setup.propagator.time_termination(t_end), propagation_setup.propagator.time_termination(t_end_before))

    # Define Keplerian elements of the Galilean moons as dependent variables
    dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Elara', 'Jupiter'),
                                   propagation_setup.dependent_variable.relative_distance('Elara', 'Earth')]
    

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        full_initial_state,
        t_0,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables_to_save,
    )
    propagator_settings.processing_settings.results_save_frequency_in_steps = 10

    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Himalia"))
    # Create the parameters that will be estimated
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings)

    convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = 5)

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(
        observations, convergence_checker=convergence_checker)

    # Set methodological options
    estimation_input.define_estimation_settings(save_state_history_per_iteration=True)

    # Define weighting of the observations in the inversion
    weight_vector = np.power(noise_level, -2)
    estimation_input.weight_matrix_diagonal = weight_vector


    # Perform the covariance analysis
    print('Performing the estimation...')
    estimation_output = estimator.perform_estimation(estimation_input)
    initial_states_updated = parameters_to_estimate.parameter_vector
    print('Done with the estimation...')
    print(f'Updated initial states: {initial_states_updated}')
    print('Done')

    #Load data
    simulator_object = estimation_output.simulation_results_per_iteration[-1]
    state_history = simulator_object.dynamics_results.state_history
    dependent_variable_history = simulator_object.dynamics_results.dependent_variable_history


    ##Correlations
    correlations = estimation_output.correlations
    design_matrix = estimation_output.normalized_design_matrix
    covariance = estimation_output.covariance

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
    # plt.xlabel("Estimated Parameter")
    # plt.ylabel("Index")


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
    position_difference = {'Elara': state_history_difference[:, 0:3]}
    velocity_difference = {'Elara': state_history_difference[:, 3:6]}
    propagation_kepler_elements = np.vstack(list(dependent_variable_history.values()))
    kepler_difference = np.vstack(list(dependent_variable_history.values()))[:, 0:6] - np.vstack(list(ephemeris_keplerian_states.values()))

    # rsw_states = []
    
    # for k in range(len(state_history_difference)):
    #     if k == 0:
    #         continue
    #     # rot_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(state_history_difference[k])
    #     # rsw_state = rot_matrix@state_history_difference[k, 0:3] #3-6?
    #     #rsw_states.append(rsw_state)

    #     rm_1 = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(state_history.values()))[k])
    #     # rm_2 = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(ephemeris_state_history.values()))[k])

    #     ab1 = rm_1@np.vstack(list(state_history.values()))[k, 0:3]
    #     ab2 = rm_1@np.vstack(list(ephemeris_state_history.values()))[k, 0:3]
    #     rsw_states.append(ab1-ab2)
    
    time2plt = list()
    epochs_julian_seconds = np.vstack(list(state_history.keys()))
    for epoch in epochs_julian_seconds:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))
    difference = propagation_kepler_elements[-1,0] - propagation_kepler_elements[0,0]
    difference_list.append(difference)


# get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the current time as a string
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Create a new folder with the current timestamp as the name
folder_name = os.path.join(script_dir, 'results_loadobs', f"plots_{current_time}")
os.makedirs(folder_name, exist_ok=True)

print('Saving as', folder_name)

rot_mat_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(initial_state)))
full_mat = np.zeros((7,7))
full_mat[:3, :3] = rot_mat_rsw
full_mat[3:6, 3:6] = rot_mat_rsw
full_mat[6,6] = 1


rotated_covariance = full_mat@covariance@np.transpose(full_mat)


std_devs = np.sqrt(np.diag(rotated_covariance))

# Create an outer product of the standard deviation vector with itself to form a matrix
std_dev_matrix = np.outer(std_devs, std_devs)

# Divide the covariance matrix element-wise by this matrix to get the correlation matrix
rotated_correlation = rotated_covariance / std_dev_matrix



plt.imshow(np.abs(rotated_correlation), aspect='auto', interpolation='none')
plt.colorbar()
plt.title("Correlation Matrix")
plt.xlabel("Index - Estimated Parameter")
plt.ylabel("Index - Estimated Parameter")

plt.tight_layout()
plt.savefig(os.path.join(folder_name, "correlation_RSW.png"), dpi=300, bbox_inches='tight')
plt.show()


plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
plt.colorbar()

plt.title("Correlation Matrix xyz")
plt.xlabel("Index - Estimated Parameter")
plt.ylabel("Index - Estimated Parameter")

plt.tight_layout()
plt.savefig(os.path.join(folder_name, "correlation.png"), dpi=300, bbox_inches='tight')
plt.show()

# #RSW
# fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,0] * 1E-3,
#          label=r'r', c='#A50034')
# ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,1] * 1E-3,
#          label=r's', c='#0076C2')
# ax1.plot(time2plt[1:], np.vstack(rsw_states)[:,2] * 1E-3,
#          label=r'w', c='#EC6842')

# ax1.set_title(r'Difference in Position')
# # ax1.plot(time2plt[1:], np.linalg.norm( np.vstack(rsw_states), axis=1) * 1E-3,
# #         label=r'Amalthea', c='#A50034')
# ax1.xaxis.set_major_locator(mdates.YearLocator(10))
# ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.set_ylabel(r'Propagated - Spice [km]')
# ax1.legend()
# ax1.grid()
# plt.savefig(os.path.join(folder_name, "RSW.png"), dpi=300, bbox_inches='tight')
# plt.show()

# #cartesian
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# ax1.scatter(np.vstack(list(state_history.values()))[:200,0]* 1E-3, np.vstack(list(state_history.values()))[:200,1]* 1E-3,
#          label=r'Propagated', c='#A50034')
# ax1.scatter(np.vstack(list(ephemeris_state_history.values()))[:1000,0]* 1E-3, (np.vstack(list(ephemeris_state_history.values()))[:1000,1])* 1E-3,
#          label=r'Spice', c='#0076C2')
# # ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
# #          label=r'Amalthea2', c='#EC6842')
# ax1.set_title(r'Difference in Position')
# # ax1.plot(time2plt, np.linalg.norm(position_difference['Amalthea'], axis=1) * 1E-3,
# #          label=r'Amalthea', c='#A50034')
# ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
# ax1.xaxis.set_minor_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
# ax1.set_ylabel(r'Difference [km]')
# ax1.legend()
# plt.show()


#original
fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(position_difference['Elara'], axis=1) * 1E-3,
         label=r'Elara', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Propagated - Spice [km]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "PosDifference.png"), dpi=300, bbox_inches='tight')
plt.show()

#velocity
fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Velocity')
ax1.plot(time2plt, np.linalg.norm(velocity_difference['Elara'], axis=1),
         label=r'Elara', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Propagated - Spice [m/s]')
ax1.set_xlabel('Time')
ax1.legend()
plt.savefig(os.path.join(folder_name, "VelDifference.png"), dpi=300, bbox_inches='tight')
plt.show()


#Kepler

fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Semi-major axis')
ax1.plot(time2plt, propagation_kepler_elements[:,0]* 1E-3,
         label=r'Elara', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Semi-major axis [km]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "SemiMajorAxis.png"), dpi=300, bbox_inches='tight')
plt.show()

#Kepler

fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Semi-major axis')
ax1.plot(time2plt, kepler_difference[:,0]* 1E-3,
         label=r'Elara', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Semi-major axis Propagated - Spice [km]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "SemiMajorAxisDiff.png"), dpi=300, bbox_inches='tight')
plt.show()

obstime2plt = list()
obs_years = list()
obs_julian_seconds = observations.concatenated_times[0::2]
for obs in obs_julian_seconds:
    obs_days = constants.JULIAN_DAY_ON_J2000 + obs / constants.JULIAN_DAY
    obstime2plt.append(time_conversion.julian_day_to_calendar_date(obs_days))
    obs_years.append(time_conversion.julian_day_to_calendar_date(obs_days).year)
residuals = estimation_output.final_residuals
residual_RA = residuals[0::2]
residual_DEC = residuals[1::2]
full_residual = estimation_output.residual_history

angular_residual = []
for m in range(len(residual_RA)):
    angular_residual.append(np.sqrt(residual_RA[m]**2+residual_DEC[m]**2))



fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(position_difference['Elara'], axis=1)/propagation_kepler_elements[:,6],
         label=r'Propagated - Spice', c='#A50034')
ax1.scatter(obstime2plt, angular_residual,
         label=r'Observation Residual', c='#0076C2')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Angular position residual [rad]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
ax1.set_yscale('log')
plt.savefig(os.path.join(folder_name, "AngularDifferenceLOG.png"), dpi=300, bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(position_difference['Elara'], axis=1)/propagation_kepler_elements[:,6],
         label=r'Propagated - Spice', c='#A50034')
ax1.scatter(obstime2plt, angular_residual,
         label=r'Observation Residual', c='#0076C2')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Angular position residual [rad]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "AngularDifference.png"), dpi=300, bbox_inches='tight')
plt.show()

with open( os.path.join(folder_name, "residual_output.csv"), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header row
    writer.writerow(['Seconds since J2000', 'RA residual [rad]','DEC residual [rad]'])
    
    # Write each matching name and result to a new row
    for j in range(len(residual_RA)):
        writer.writerow([obs_julian_seconds[j], residual_RA[j],residual_DEC[j]])


# Residual iteration
for iteration in range(5): 
    fig, ax1 = plt.subplots(1, 1)

    ax1.set_title(iteration)
    residuals_iteration = full_residual[:,iteration]
    rms_list = []
    residual_RA_it = residuals_iteration[0::2]
    residual_DEC_it = residuals_iteration[1::2]
    for i in range(len(residual_RA)):
        # Calculate the squares of the i-th elements from each list
        squared_x = residual_RA_it[i]**2
        squared_y = residual_DEC_it[i]**2

        # Calculate the mean of the squared values
        mean_squared = (squared_x + squared_y) / 2

        # Calculate the RMS value by taking the square root of the mean squared value
        rms = math.sqrt(mean_squared)

        # Append the RMS value to the list
        rms_list.append(rms)
    ax1.scatter(obstime2plt, residual_RA_it,
         label='RA')
    #ax1.scatter(obstime2plt, residual_DEC_it,
    #     label='DEC')    
        
    ax1.xaxis.set_major_locator(mdates.YearLocator(20))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_ylabel(r'angular position residual [rad]')
    ax1.set_xlabel('Time')
    ax1.legend()
    plt.grid()
    plt.savefig(os.path.join(folder_name, str(iteration)), dpi=300, bbox_inches='tight')
    plt.show()




obs_years_array = np.array(obs_years)
#Histogram prep
residual_RA_1 = residual_RA[obs_years_array<1950]
residual_RA_2 = residual_RA[(obs_years_array>=1950) & (obs_years_array<=1980)]
residual_RA_3 = residual_RA[obs_years_array>1980]

residual_DEC_1 = residual_DEC[obs_years_array<1950]
residual_DEC_2 = residual_DEC[(obs_years_array>=1950) & (obs_years_array<=1980)]
residual_DEC_3 = residual_DEC[obs_years_array>1980]

filtered_RA = residual_RA_3[(residual_RA_3 >= -2.5*10**(-6)) & (residual_RA_3 <= 2.5*10**(-6))]
filtered_DEC = residual_DEC_3[(residual_DEC_3 >= -2*10**(-6)) & (residual_DEC_3 <= 2*10**(-6))]

ks_statistic, p_value = stats.kstest(filtered_RA, 'norm', args=(0, np.std(filtered_RA)))

print(f"RA KS Statistic: {ks_statistic}")
print(f"RA P-Value: {p_value}")

ks_statistic, p_value = stats.kstest(filtered_DEC, 'norm', args=(0, np.std(filtered_DEC)))

print(f"DEC KS Statistic: {ks_statistic}")
print(f"DEC P-Value: {p_value}")

#Histograms
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))


axes[0,0].hist(residual_RA_1, bins=30, edgecolor='black')
axes[0,0].set_title('RA residuals <1950')
axes[0,0].set_xlabel('Residual [rad]')
axes[0,0].set_ylabel('Frequency')

axes[0,1].hist(residual_RA_2, bins=30, edgecolor='black')
axes[0,1].set_title('RA residuals 1950-1980')
axes[0,1].set_xlabel('Residual [rad]')
axes[0,1].set_ylabel('Frequency')

axes[0,2].hist(residual_RA_3, bins=30,range=(-0.8*10**(-6), 0.8*10**(-6)), edgecolor='black')
axes[0,2].set_title('RA residuals >1980')
axes[0,2].set_xlabel('Residual [rad]')
axes[0,2].set_ylabel('Frequency')

axes[1,0].hist(residual_DEC_1, bins=30, edgecolor='black')
axes[1,0].set_title('DEC residuals <1950')
axes[1,0].set_xlabel('Residual [rad]')
axes[1,0].set_ylabel('Frequency')

axes[1,1].hist(residual_DEC_2, bins=30, edgecolor='black')
axes[1,1].set_title('DEC residuals 1950-1980')
axes[1,1].set_xlabel('Residual [rad]')
axes[1,1].set_ylabel('Frequency')

axes[1,2].hist(residual_DEC_3, bins=30,range=(-0.8*10**(-6), 0.8*10**(-6)), edgecolor='black')
axes[1,2].set_title('DEC residuals >1980')
axes[1,2].set_xlabel('Residual [rad]')
axes[1,2].set_ylabel('Frequency')

# Adjust the layout
plt.tight_layout()
plt.savefig(os.path.join(folder_name, 'hist'), dpi=300, bbox_inches='tight')
# Show the plot
plt.show()




rms_list = []
for i in range(len(residual_RA)):
        # Calculate the squares of the i-th elements from each list
    squared_x = residual_RA[i]**2
    squared_y = residual_DEC[i]**2

    # Calculate the mean of the squared values
    mean_squared = (squared_x + squared_y) / 2

    # Calculate the RMS value by taking the square root of the mean squared value
    rms = math.sqrt(mean_squared)

    # Append the RMS value to the list
    rms_list.append(rms)
# plt.plot(observation_times,rms_list, label='rms')
# plt.legend()
# plt.grid()
# plt.xlabel('time [s since J2000]')
# plt.ylabel('rms of the position')
# plt.savefig(os.path.join(folder_name, "plot6.png"), dpi=300, bbox_inches='tight')
# plt.show()


fig, ax1 = plt.subplots(1, 1)

ax1.set_title(r'RMS of position residual')
ax1.scatter(obstime2plt, rms_list,
         label=r'Elara', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'RMS of angular position residual [rad]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "RMS_Residual.png"), dpi=300, bbox_inches='tight')
plt.show()


fig, ax1 = plt.subplots(1, 1)

ax1.set_title(r'Angular position residual')
ax1.scatter(obstime2plt, residual_RA,
         label=r'Elara RA', c='#A50034')
ax1.scatter(obstime2plt, residual_DEC,
         label=r'Elara DEC', c='#0076C2')
ax1.xaxis.set_major_locator(mdates.YearLocator(20))
ax1.xaxis.set_minor_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Angular position residual [rad]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "Residuals.png"), dpi=300, bbox_inches='tight')
plt.show()


script_name = os.path.basename(__file__)
shutil.copy(script_name, os.path.join(folder_name, script_name))



print('Done')