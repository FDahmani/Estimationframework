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
t_0 = time_conversion.DateTime(1990,1,1,12,0,0).epoch()
t_end = time_conversion.DateTime(2003,1,1,12,0,0).epoch()
t_end_before = time_conversion.DateTime(1976,1,1,12,0,0).epoch()
time_difference = t_end - t_0


folder_path = 'ObservationsProcessed/CurrentProcess'
raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

#Set up environment
bodies_to_create = ["Sun",'Saturn', "Earth",'Jupiter', 'Io', 'Europa','Ganymede','Callisto','Amalthea','Himalia','Elara','Pasiphae','Sinope',
                    'Lysithea','Carme','Ananke','Leda','Thebe','Adrastea','Metis']



# Create default body settings
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

#For propagated moons, voor initial state, Nan lijst veranderen
body_settings.get('Amalthea').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(environment_setup.ephemeris.direct_spice("Jupiter",global_frame_orientation,"Amalthea"),t_0-12*7200,t_end+12*7200,1*60,interpolators.lagrange_interpolation(4))         #Check number of points used in interpolating


# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

obstimes = []

observation_set_list = []
observation_settings_list = []
observation_simulation_settings = []
noise_level = []
arc_start_times = []
bias_values = []
link_def_list = []

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
    Observatory = str(split_string[1])+str(split_string[2])
    study = str(split_string[2])
    # study_labels.append(study)

    # Define the position of the observatory on Earth
    observatory_longitude, observatory_latitude, observatory_altitude = observatory_info (str(split_string[1]))

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
    uncer = []
    Timelist = []
    with open(file, 'r') as f:
        print(file)
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
            uncer.append(np.asarray([float(row[3]), float(row[4])]))
            # Uncertainties
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
                bias_values.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))
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
        bias_values.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))
        bias_values_local.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))

    angles = ObservationList
    times = Timelist
    arc_start_times.append(arc_start_times_local)
    
    # Define link ends
    link_ends = dict()                  
    link_ends[observation.transmitter] = observation.body_origin_link_end_id(moon_name(Moon-500))
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", str(Observatory))
    link_definition = observation.LinkDefinition(link_ends)
    #for i in range(len(arc_start_times_local)):
    link_def_list.append(link_definition)
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
    observation_settings_list.append(observation.angular_position(link_definition,bias_settings=bias))     #Extra input, zie slack                    
    #observation_settings_list.append(observation.angular_position(link_definition))


observations = estimation.ObservationCollection( observation_set_list ) 

## Set up the propagation
# Define bodies that are propagated
bodies_to_propagate = ['Amalthea']

# Define central bodies of propagation
central_bodies = ["Jupiter"]

# Define the accelerations acting on the moons
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
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Europa=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Ganymede=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Callisto=[
        propagation_setup.acceleration.point_mass_gravity()
    ])



# Create global accelerations dictionary
acceleration_settings = {"Amalthea": accelerations_settings_common}

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
full_initial_state = [] #In case more bodies are desired
for i in bodies_to_propagate:
    #initial_state = spice.get_body_cartesian_state_at_epoch(i,"Jupiter","ECLIPJ2000",'None',t_0)
    initial_state = [-1.71358285e+08,  6.02868819e+07,  8.61709429e+05 ,-8.70255003e+03, -2.49379545e+04, -1.02136012e+03]  #1990
    full_initial_state.append(initial_state)
full_initial_state = np.concatenate(full_initial_state)



### Create the integrator settings
"""
# Use fixed step-size integrator (RKDP8) with fixed time-step of 6 minutes
"""

time_steps = [6]
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
    termination_condition = propagation_setup.propagator.non_sequential_termination(
    propagation_setup.propagator.time_termination(t_end), propagation_setup.propagator.time_termination(t_end_before))
    #termination_condition = propagation_setup.propagator.time_termination(t_end)

    # Define Keplerian elements of the Galilean moons as dependent variables
    dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Amalthea', 'Jupiter'),
                                   propagation_setup.dependent_variable.relative_distance('Amalthea', 'Earth')]
    

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
    propagator_settings.processing_settings.results_save_frequency_in_steps = 1

    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    counter = 6
    # Create the parameters that will be estimated
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    precision = 200000
    hours = 1
    pre = 1/(1E-6)**2
    inv_pre_r = 1/(precision)**2
    inv_pre_v = 1/(precision/(hours*3600))**2
    inverse_apriori_covariance = np.asarray([[inv_pre_r,0,0,0,0,0],[0,inv_pre_r,0,0,0,0],[0,0,inv_pre_r,0,0,0],[0,0,0,inv_pre_v,0,0],[0,0,0,0,inv_pre_v,0,],[0,0,0,0,0,inv_pre_v]])

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings)


    convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = 5)



    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(
        observations,inverse_apriori_covariance=inverse_apriori_covariance, convergence_checker=convergence_checker)
    # estimation_input = estimation.EstimationInput(
    #     observations, convergence_checker=convergence_checker)

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

    # #Correlation Matrix
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
    #Loop over the propagated states and use the SPICE ephemeris as benchmark solution
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
    kepler_difference = np.vstack(list(dependent_variable_history.values()))[:, 0:6] - np.vstack(list(ephemeris_keplerian_states.values()))

    time2plt = list()
    epochs_julian_seconds = np.vstack(list(state_history.keys()))
    for epoch in epochs_julian_seconds:
        epoch_days = constants.JULIAN_DAY_ON_J2000 + epoch / constants.JULIAN_DAY
        time2plt.append(time_conversion.julian_day_to_calendar_date(epoch_days))
    difference = propagation_kepler_elements[-1,0] - propagation_kepler_elements[0,0]
    difference_list.append(difference)

    #RSW
    rsw_states = []
    div = 1000
    print('expected conversions = ', math.floor(len(state_history_difference)/div))
    print(len(time2plt[div:-1:div]))
    for k in range(math.floor(len(state_history_difference)/div)):
        if k%100==0:
            print (k)
        if k == 10:
            print (k)
        if k == 0:
            continue

        rm_1 = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(state_history.values()))[div*k])

        ab1 = rm_1@np.vstack(list(state_history.values()))[div*k, 0:3]
        ab2 = rm_1@np.vstack(list(ephemeris_state_history.values()))[div*k, 0:3]
        rsw_states.append(ab1-ab2)

# get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the current time as a string
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Create a new folder with the current timestamp as the name
folder_name = os.path.join(script_dir, 'results_loadobs', f"plots_{current_time}")
os.makedirs(folder_name, exist_ok=True)

print('Saving as', folder_name)

rot_mat_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(np.vstack(list(initial_state)))
full_mat = np.zeros((6,6))
full_mat[:3, :3] = rot_mat_rsw
full_mat[3:, 3:] = rot_mat_rsw


rotated_covariance = full_mat@covariance@np.transpose(full_mat)


std_devs = np.sqrt(np.diag(rotated_covariance))

# Create an outer product of the standard deviation vector with itself to form a matrix
std_dev_matrix = np.outer(std_devs, std_devs)

# Divide the covariance matrix element-wise by this matrix to get the correlation matrix
rotated_correlation = rotated_covariance / std_dev_matrix

# plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
# plt.colorbar()

# plt.title("Correlation Matrix xyz")
# plt.xlabel("Index - Estimated Parameter")
# plt.ylabel("Index - Estimated Parameter")

# plt.tight_layout()
# plt.savefig(os.path.join(folder_name, "correlation.png"), dpi=300, bbox_inches='tight')
# plt.show()


plt.imshow(np.abs(rotated_correlation), aspect='auto', interpolation='none')
plt.colorbar()

plt.title("Correlation Matrix rsw")
plt.xlabel("Index - Estimated Parameter")
plt.ylabel("Index - Estimated Parameter")

plt.tight_layout()
plt.savefig(os.path.join(folder_name, "correlation_RSW.png"))
plt.show()

# plt.plot(time_steps,difference_list)
# plt.xlabel('time step [min]')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('Change in semi major axis [m]')
# plt.grid()
# plt.show()

#RSW
fig, ax1 = plt.subplots(1, 1)

ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,0] * 1E-3,
         label=r'r', c='#A50034')
ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,1] * 1E-3,
         label=r's', c='#0076C2')
ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,2] * 1E-3,
         label=r'w', c='#EC6842')
ax1.set_title(r'Difference in Position')
# ax1.plot(time2plt[1:], np.linalg.norm( np.vstack(rsw_states), axis=1) * 1E-3,
#         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Propagated - Spice [km]')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "RSW.png"), dpi=300, bbox_inches='tight')
plt.show()

# #cartesian
# fig, ax1 = plt.subplots(1, 1)

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
ax1.plot(time2plt, np.linalg.norm(position_difference['Amalthea'], axis=1) * 1E-3,
         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
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
ax1.plot(time2plt, np.linalg.norm(velocity_difference['Amalthea'], axis=1),
         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
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
         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
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
         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Semi-major axis Propagated - Spice [km]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
plt.savefig(os.path.join(folder_name, "SemiMajorAxisDiff.png"), dpi=300, bbox_inches='tight')
plt.show()

obstime2plt = list()
obs_julian_seconds = observations.concatenated_times[0::2]
for obs in obs_julian_seconds:
    obs_days = constants.JULIAN_DAY_ON_J2000 + obs / constants.JULIAN_DAY
    obstime2plt.append(time_conversion.julian_day_to_calendar_date(obs_days))
residuals = estimation_output.final_residuals
residual_RA = residuals[0::2]
residual_DEC = residuals[1::2]
full_residual = estimation_output.residual_history
angular_residual = []
for m in range(len(residual_RA)):
    angular_residual.append(np.sqrt(residual_RA[m]**2+residual_DEC[m]**2))


filtered_RA = residual_RA[(residual_RA >= -2.5*10**(-6)) & (residual_RA <= 2.5*10**(-6))]
filtered_DEC = residual_DEC[(residual_DEC >= -2*10**(-6)) & (residual_DEC <= 2*10**(-6))]

ks_statistic, p_value = stats.kstest(filtered_RA, 'norm', args=(0, np.std(filtered_RA)) )

print(f"RA KS Statistic: {ks_statistic}")
print(f"RA P-Value: {p_value}")

ks_statistic, p_value = stats.kstest(filtered_DEC, 'norm', args=(0, np.std(filtered_DEC)) )

print(f"DEC KS Statistic: {ks_statistic}")
print(f"DEC P-Value: {p_value}")


#Histograms
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))


axes[0].hist(residual_RA, bins=30, range=(-0.5*10**(-6), 0.5*10**(-6)), edgecolor='black')
axes[0].set_title('RA residuals')
axes[0].set_xlabel('Residual [rad]')
axes[0].set_ylabel('Frequency')

axes[1].hist(residual_DEC, bins=30, range=(-0.5*10**(-6), 0.5*10**(-6)), edgecolor='black')
axes[1].set_title('DEC residuals')
axes[1].set_xlabel('Residual [rad]')
axes[1].set_ylabel('Frequency')


# Adjust the layout
plt.tight_layout()
plt.savefig(os.path.join(folder_name, 'hist'), dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

fig, ax1 = plt.subplots(1, 1)

# ax1.plot(time2plt, position_difference['Amalthea'][:,0] * 1E-3,
#          label=r'Amalthea0', c='#A50034')
# ax1.plot(time2plt, position_difference['Amalthea'][:,1] * 1E-3,
#          label=r'Amalthea1', c='#0076C2')
# ax1.plot(time2plt, position_difference['Amalthea'][:,2] * 1E-3,
#          label=r'Amalthea2', c='#EC6842')
ax1.set_title(r'Difference in Position')
ax1.plot(time2plt, np.linalg.norm(position_difference['Amalthea'], axis=1)/propagation_kepler_elements[:,6],
         label=r'Propagated - Spice', c='#A50034')
ax1.scatter(obstime2plt, angular_residual,
         label=r'Observation Residual', c='#0076C2')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylabel(r'Angular position residual [rad]')
ax1.set_xlabel('Time')
ax1.legend()
ax1.grid()
ax1.set_yscale('log')
plt.savefig(os.path.join(folder_name, "AngularDifferenceLOG.png"), dpi=300, bbox_inches='tight')
plt.show()

with open( folder_name + 'residual_output' +'.csv', 'w', newline='') as csvfile:
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
        
    ax1.xaxis.set_major_locator(mdates.YearLocator(10))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_ylabel(r'angular position residual [rad]')
    ax1.set_xlabel('Time')
    ax1.legend()
    plt.grid()
    plt.savefig(os.path.join(folder_name, str(iteration)), dpi=300, bbox_inches='tight')
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
         label=r'Amalthea', c='#A50034')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
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
         label=r'Amalthea RA', c='#A50034')
ax1.scatter(obstime2plt, residual_DEC,
         label=r'Amalthea DEC', c='#0076C2')
ax1.xaxis.set_major_locator(mdates.YearLocator(10))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
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