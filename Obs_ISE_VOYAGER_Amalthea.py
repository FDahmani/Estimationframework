# Load required standard modules
import sys
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

voyager_obstimes = [time_conversion.DateTime(1979,2,3,5,24,36).epoch(), #Voyager1
                    time_conversion.DateTime(1979,2,3,5,34,12).epoch(),
                    time_conversion.DateTime(1979,2,3,5,47,00).epoch(),
                    time_conversion.DateTime(1979,2,3,5,51,46).epoch(),
                    time_conversion.DateTime(1979,2,4,3,19,16).epoch(),
                    time_conversion.DateTime(1979,2,8,4,53,22).epoch(),
                    time_conversion.DateTime(1979,2,8,4,58,10).epoch(),
                    time_conversion.DateTime(1979,2,18,7,26,58).epoch(),
                    time_conversion.DateTime(1979,2,27,15,27,45).epoch(),
                    time_conversion.DateTime(1979,3,4,15,52,34).epoch(),
                    time_conversion.DateTime(1979,3,4,15,55,46).epoch(),
                    time_conversion.DateTime(1979,3,4,19,55,46).epoch(),
                    time_conversion.DateTime(1979,3,4,19,57,22).epoch(),
                    time_conversion.DateTime(1979,6,23,5,5,33).epoch(), #Voyager2
                    time_conversion.DateTime(1979,2,23,5,7,57).epoch(),
                    time_conversion.DateTime(1979,2,23,5,10,21).epoch(),
                    time_conversion.DateTime(1979,2,23,5,12,45).epoch(),
                    time_conversion.DateTime(1979,2,23,5,15,9).epoch(),
                    time_conversion.DateTime(1979,2,23,5,17,33).epoch(),
                    ]

galileo_obstimes = [time_conversion.DateTime(1996,9,7,8,18,1).epoch(),
                    time_conversion.DateTime(1996,9,7,8,18,10).epoch(),
                    time_conversion.DateTime(1996,9,7,8,18,19).epoch(),
                    time_conversion.DateTime(1996,11,5,19,38,56).epoch(),
                    time_conversion.DateTime(1996,11,5,19,39,4).epoch(),
                    time_conversion.DateTime(1996,11,5,19,39,13).epoch(),
                    time_conversion.DateTime(1996,12,2,20,0,23).epoch(),
                    time_conversion.DateTime(1996,12,18,7,13,7).epoch(),
                    time_conversion.DateTime(1996,12,18,7,13,16).epoch(),
                    time_conversion.DateTime(1996,12,18,7,13,25).epoch(),
                    time_conversion.DateTime(1996,12,18,7,13,34).epoch(),
                    time_conversion.DateTime(1997,2,20,21,57,32).epoch(),
                    time_conversion.DateTime(1997,6,27,9,43,2).epoch(),
                    time_conversion.DateTime(1997,6,27,13,28,31).epoch(),
                    time_conversion.DateTime(1997,9,18,9,22,7).epoch(),
                    time_conversion.DateTime(1997,11,6,20,36,44).epoch(),
                    time_conversion.DateTime(1997,11,6,20,36,51).epoch(),
                    time_conversion.DateTime(1997,11,7,1,0,47).epoch(),
                    time_conversion.DateTime(1997,11,6,20,36,44).epoch(),
                    time_conversion.DateTime(1999,8,12,17,22,58).epoch(),
                    time_conversion.DateTime(1999,8,12,19,13,11).epoch(),
                    time_conversion.DateTime(1999,11,26,8,8,8).epoch(),
                    time_conversion.DateTime(2000,1,4,2,46,43).epoch(),
                    time_conversion.DateTime(2000,12,31,1,0,40).epoch(),
                    time_conversion.DateTime(2001,1,2,1,29,39).epoch(),
                    time_conversion.DateTime(2002,1,18,5,22,1).epoch(),
                    time_conversion.DateTime(2002,1,18,14,6,47).epoch(),
                    ]   

galileoCA_obstimes = [time_conversion.DateTime(2002,11,5,5,0,0).epoch(),
                      time_conversion.DateTime(2002,11,5,5,30,0).epoch(),
                      time_conversion.DateTime(2002,11,5,6,0,0).epoch(),
                      time_conversion.DateTime(2002,11,5,6,19,0).epoch(),
                     ]

cassini_obstimes = [time_conversion.DateTime(2000,12,11,19,19,13).epoch(),
                    time_conversion.DateTime(2000,12,11,19,24,52).epoch(),
                    time_conversion.DateTime(2000,12,11,19,26,00).epoch(),
                    time_conversion.DateTime(2000,12,11,19,27,32).epoch(),
                    time_conversion.DateTime(2000,12,11,19,30,13).epoch(),
                    time_conversion.DateTime(2000,12,11,19,35,52).epoch(),
                    time_conversion.DateTime(2000,12,11,19,37,00).epoch(),
                    time_conversion.DateTime(2000,12,11,19,38,32).epoch(),
                    time_conversion.DateTime(2000,12,11,19,41,13).epoch(),
                    time_conversion.DateTime(2000,12,11,20,5,56).epoch(),
                    time_conversion.DateTime(2000,12,11,20,7,6).epoch(),
                    time_conversion.DateTime(2000,12,11,20,8,38).epoch(),
                    time_conversion.DateTime(2000,12,11,20,11,19).epoch(),
                    time_conversion.DateTime(2000,12,11,20,16,58).epoch(),
                    time_conversion.DateTime(2000,12,11,20,19,38).epoch(),
                    time_conversion.DateTime(2000,12,11,20,22,19).epoch(),
                    time_conversion.DateTime(2000,12,11,20,27,58).epoch(),
                    time_conversion.DateTime(2000,12,11,20,29,6).epoch(),
                    time_conversion.DateTime(2000,12,11,20,30,38).epoch(),
                    time_conversion.DateTime(2000,12,11,20,33,19).epoch(),
                    time_conversion.DateTime(2000,12,11,20,38,58).epoch(),
                    time_conversion.DateTime(2000,12,11,20,40,6).epoch(),
                    time_conversion.DateTime(2000,12,11,20,41,38).epoch(),
                    time_conversion.DateTime(2000,12,11,20,44,19).epoch(),
                    time_conversion.DateTime(2000,12,11,21,9,2).epoch(),
                    time_conversion.DateTime(2000,12,11,21,10,12).epoch(),
                    time_conversion.DateTime(2000,12,11,21,11,44).epoch(),
                    time_conversion.DateTime(2000,12,11,21,14,25).epoch(),
                    time_conversion.DateTime(2000,12,11,21,20,4).epoch(),
                    time_conversion.DateTime(2000,12,11,21,21,12).epoch(),
                    time_conversion.DateTime(2000,12,11,21,22,44).epoch(),
                    time_conversion.DateTime(2000,12,11,21,25,25).epoch(),
                    time_conversion.DateTime(2000,12,11,21,31,4).epoch(),
                    time_conversion.DateTime(2000,12,11,21,32,12).epoch(),
                    time_conversion.DateTime(2000,12,11,21,33,44).epoch(),
                    time_conversion.DateTime(2000,12,11,21,36,25).epoch(),
                    time_conversion.DateTime(2000,12,11,21,42,4).epoch(),
                    time_conversion.DateTime(2000,12,11,21,43,12).epoch(),
                    time_conversion.DateTime(2000,12,11,21,44,44).epoch(),
                    time_conversion.DateTime(2000,12,11,21,47,25).epoch(),
                    time_conversion.DateTime(2000,12,11,22,12,8).epoch(),
                    time_conversion.DateTime(2000,12,11,22,13,18).epoch(),
                    time_conversion.DateTime(2000,12,11,22,14,50).epoch(),
                    time_conversion.DateTime(2000,12,11,22,17,31).epoch(),
                    time_conversion.DateTime(2000,12,11,22,23,10).epoch(),
                    time_conversion.DateTime(2000,12,11,22,24,18).epoch(),
                    time_conversion.DateTime(2000,12,11,22,25,50).epoch(),
                    time_conversion.DateTime(2000,12,11,22,28,31).epoch(),
                    time_conversion.DateTime(2000,12,11,22,34,10).epoch(),
                    time_conversion.DateTime(2000,12,11,22,35,18).epoch(),
                    time_conversion.DateTime(2000,12,11,22,36,50).epoch(),
                    time_conversion.DateTime(2000,12,11,22,39,31).epoch(),
                    time_conversion.DateTime(2000,12,11,22,45,10).epoch(),
                    time_conversion.DateTime(2000,12,11,22,46,18).epoch(),
                    time_conversion.DateTime(2000,12,11,22,47,50).epoch(),
                    time_conversion.DateTime(2000,12,11,22,50,31).epoch(),
                    time_conversion.DateTime(2000,12,12,7,40,2).epoch(),
                    time_conversion.DateTime(2000,12,12,7,41,12).epoch(),
                    time_conversion.DateTime(2000,12,12,7,42,44).epoch(),
                    time_conversion.DateTime(2000,12,12,7,45,25).epoch(),
                    time_conversion.DateTime(2000,12,12,7,51,4).epoch(),
                    time_conversion.DateTime(2000,12,12,7,52,12).epoch(),
                    time_conversion.DateTime(2000,12,12,7,53,44).epoch(),
                    time_conversion.DateTime(2000,12,12,7,56,25).epoch(),
                    time_conversion.DateTime(2000,12,12,8,2,4).epoch(),
                    time_conversion.DateTime(2000,12,12,8,3,12).epoch(),
                    time_conversion.DateTime(2000,12,12,8,4,44).epoch(),
                    time_conversion.DateTime(2000,12,12,8,7,25).epoch(),
                    time_conversion.DateTime(2000,12,12,8,13,4).epoch(),
                    time_conversion.DateTime(2000,12,12,8,14,12).epoch(),
                    time_conversion.DateTime(2000,12,12,8,15,44).epoch(),
                    time_conversion.DateTime(2000,12,12,8,18,25).epoch(),
                    time_conversion.DateTime(2000,12,12,8,43,8).epoch(),
                    time_conversion.DateTime(2000,12,12,8,44,18).epoch(),
                    time_conversion.DateTime(2000,12,12,8,45,50).epoch(),
                    time_conversion.DateTime(2000,12,12,8,48,31).epoch(),
                    time_conversion.DateTime(2000,12,12,8,54,10).epoch(),
                    time_conversion.DateTime(2000,12,12,8,55,18).epoch(),
                    time_conversion.DateTime(2000,12,12,8,56,50).epoch(),
                    time_conversion.DateTime(2000,12,12,8,59,31).epoch(),
                    time_conversion.DateTime(2000,12,12,9,6,18).epoch(),
                    time_conversion.DateTime(2000,12,12,9,7,50).epoch(),
                    time_conversion.DateTime(2000,12,12,9,10,31).epoch(),
                    time_conversion.DateTime(2000,12,12,9,16,10).epoch(),
                    time_conversion.DateTime(2000,12,12,9,17,18).epoch(),
                    time_conversion.DateTime(2000,12,12,9,18,50).epoch(),
                    time_conversion.DateTime(2000,12,12,9,21,31).epoch(),
                    time_conversion.DateTime(2000,12,12,9,46,14).epoch(),
                    time_conversion.DateTime(2000,12,12,9,47,24).epoch(),
                    time_conversion.DateTime(2000,12,12,9,48,56).epoch(),
                    time_conversion.DateTime(2000,12,12,9,51,37).epoch(),
                    time_conversion.DateTime(2000,12,12,9,57,16).epoch(),
                    time_conversion.DateTime(2000,12,12,9,59,56).epoch(),
                    time_conversion.DateTime(2000,12,12,10,2,37).epoch(),
                    time_conversion.DateTime(2000,12,12,10,8,16).epoch(),
                    time_conversion.DateTime(2000,12,12,10,9,24).epoch(),
                    time_conversion.DateTime(2000,12,12,10,10,56).epoch(),
                    time_conversion.DateTime(2000,12,12,10,13,37).epoch(),
                    time_conversion.DateTime(2000,12,12,10,19,16).epoch(),
                    time_conversion.DateTime(2000,12,12,10,20,24).epoch(),
                    time_conversion.DateTime(2000,12,12,10,21,56).epoch(),
                    time_conversion.DateTime(2000,12,12,10,24,37).epoch(),
                    time_conversion.DateTime(2000,12,12,10,49,20).epoch(),
                    time_conversion.DateTime(2000,12,12,10,50,30).epoch(),
                    time_conversion.DateTime(2000,12,12,10,52,2).epoch(),
                    time_conversion.DateTime(2000,12,12,10,54,43).epoch(),
                    time_conversion.DateTime(2000,12,12,19,15,18).epoch(),
                    time_conversion.DateTime(2000,12,12,19,16,50).epoch(),
                    time_conversion.DateTime(2000,12,12,19,19,31).epoch(),
                    time_conversion.DateTime(2000,12,12,19,25,10).epoch(),
                    time_conversion.DateTime(2000,12,12,19,26,18).epoch(),
                    time_conversion.DateTime(2000,12,12,19,27,50).epoch(),
                    time_conversion.DateTime(2000,12,12,19,30,31).epoch(),
                    time_conversion.DateTime(2000,12,12,19,36,10).epoch(),
                    time_conversion.DateTime(2000,12,12,19,37,18).epoch(),
                    time_conversion.DateTime(2000,12,12,19,38,50).epoch(),
                    time_conversion.DateTime(2000,12,12,19,41,31).epoch(),
                    time_conversion.DateTime(2000,12,12,19,47,10).epoch(),
                    time_conversion.DateTime(2000,12,12,19,48,18).epoch(),
                    time_conversion.DateTime(2000,12,12,19,49,50).epoch(),
                    time_conversion.DateTime(2000,12,12,20,17,14).epoch(),
                    time_conversion.DateTime(2000,12,12,20,18,24).epoch(),
                    time_conversion.DateTime(2000,12,12,20,19,56).epoch(),
                    time_conversion.DateTime(2000,12,12,20,22,37).epoch(),
                    time_conversion.DateTime(2000,12,12,20,28,16).epoch(),
                    time_conversion.DateTime(2000,12,12,20,29,24).epoch(),
                    time_conversion.DateTime(2000,12,12,20,30,56).epoch(),
                    time_conversion.DateTime(2000,12,12,20,33,37).epoch(),
                    time_conversion.DateTime(2000,12,12,20,39,16).epoch(),
                    time_conversion.DateTime(2000,12,12,20,41,56).epoch(),
                    time_conversion.DateTime(2000,12,12,20,44,37).epoch(),
                    time_conversion.DateTime(2000,12,12,20,50,16).epoch(),
                    time_conversion.DateTime(2000,12,12,20,51,24).epoch(),
                    time_conversion.DateTime(2000,12,12,20,52,56).epoch(),
                    time_conversion.DateTime(2000,12,12,20,55,37).epoch(),
                    time_conversion.DateTime(2000,12,12,21,20,20).epoch(),
                    time_conversion.DateTime(2000,12,12,21,23,2).epoch(),
                    time_conversion.DateTime(2000,12,12,21,25,43).epoch(),
                    time_conversion.DateTime(2000,12,12,21,31,22).epoch(),
                    time_conversion.DateTime(2000,12,12,21,34,2).epoch(),
                    time_conversion.DateTime(2000,12,12,21,36,43).epoch(),
                    time_conversion.DateTime(2000,12,12,21,42,22).epoch(),
                    time_conversion.DateTime(2000,12,12,21,45,2).epoch(),
                    time_conversion.DateTime(2000,12,12,21,47,43).epoch(),
                    time_conversion.DateTime(2000,12,12,21,53,22).epoch(),
                    time_conversion.DateTime(2000,12,12,21,56,2).epoch(),
                    time_conversion.DateTime(2000,12,12,21,58,43).epoch(),
                    time_conversion.DateTime(2000,12,12,22,23,26).epoch(),
                    time_conversion.DateTime(2000,12,12,22,24,36).epoch(),
                    time_conversion.DateTime(2000,12,12,22,26,8).epoch(),
                    time_conversion.DateTime(2000,12,12,22,28,49).epoch(),
                    time_conversion.DateTime(2000,12,12,22,34,28).epoch(),
                    time_conversion.DateTime(2000,12,12,22,35,36).epoch(),
                    time_conversion.DateTime(2000,12,12,22,37,8).epoch(),
                    time_conversion.DateTime(2000,12,12,22,39,49).epoch(),
                    time_conversion.DateTime(2000,12,12,22,45,28).epoch(),
                    time_conversion.DateTime(2000,12,12,22,46,36).epoch(),
                    time_conversion.DateTime(2000,12,12,22,48,8).epoch(),
                    time_conversion.DateTime(2000,12,12,22,50,49).epoch(),
                    time_conversion.DateTime(2000,12,12,22,56,28).epoch(),
                    time_conversion.DateTime(2000,12,12,22,57,36).epoch(),
                    time_conversion.DateTime(2000,12,12,22,59,8).epoch(),
                    time_conversion.DateTime(2001,1,15,9,43,41).epoch(),
                    time_conversion.DateTime(2001,1,15,9,46,41).epoch(),
                    time_conversion.DateTime(2001,1,15,9,49,41).epoch(),
                    time_conversion.DateTime(2001,1,15,9,52,41).epoch(),
                    time_conversion.DateTime(2001,1,15,9,55,41).epoch(),
                    time_conversion.DateTime(2001,1,15,9,58,41).epoch(),
                    time_conversion.DateTime(2001,1,15,10,1,41).epoch(),
                    time_conversion.DateTime(2001,1,15,10,4,41).epoch(),
                    time_conversion.DateTime(2001,1,15,10,7,41).epoch(),
                    time_conversion.DateTime(2001,1,15,10,25,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,28,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,31,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,34,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,37,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,40,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,43,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,46,1).epoch(),
                    time_conversion.DateTime(2001,1,15,10,49,1).epoch(),
                    time_conversion.DateTime(2001,1,15,11,6,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,9,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,12,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,15,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,18,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,21,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,24,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,27,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,30,21).epoch(),
                    time_conversion.DateTime(2001,1,15,11,47,41).epoch(),
                    time_conversion.DateTime(2001,1,15,11,53,41).epoch(),
                    time_conversion.DateTime(2001,1,15,11,56,41).epoch(),
                    time_conversion.DateTime(2001,1,15,20,27,40).epoch(),
                    time_conversion.DateTime(2001,1,15,20,45,00).epoch(),
                    time_conversion.DateTime(2001,1,15,20,48,00).epoch(),
                    time_conversion.DateTime(2001,1,15,20,51,00).epoch(),
                    time_conversion.DateTime(2001,1,15,20,54,00).epoch(),
                    time_conversion.DateTime(2001,1,15,20,57,00).epoch(),
                    time_conversion.DateTime(2001,1,15,21,00,00).epoch(),
                    time_conversion.DateTime(2001,1,15,21,3,00).epoch(),
                    time_conversion.DateTime(2001,1,15,21,6,00).epoch(),
                    time_conversion.DateTime(2001,1,15,21,9,00).epoch(),
                    time_conversion.DateTime(2001,1,15,21,26,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,29,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,32,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,35,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,38,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,41,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,44,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,47,20).epoch(),
                    time_conversion.DateTime(2001,1,15,21,50,20).epoch(),
                    time_conversion.DateTime(2001,1,15,22,7,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,10,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,13,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,16,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,19,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,22,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,25,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,28,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,31,40).epoch(),
                    time_conversion.DateTime(2001,1,15,22,49,00).epoch(),
                    time_conversion.DateTime(2001,1,15,22,52,00).epoch(),
                    time_conversion.DateTime(2001,1,15,22,55,00).epoch(),
                    time_conversion.DateTime(2001,1,15,22,58,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,1,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,4,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,7,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,10,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,13,00).epoch(),
                    time_conversion.DateTime(2001,1,15,23,33,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,36,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,39,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,42,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,45,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,48,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,51,20).epoch(),
                    time_conversion.DateTime(2001,1,15,23,54,20).epoch(),
                    time_conversion.DateTime(2001,1,16,8,33,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,36,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,39,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,42,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,45,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,48,40).epoch(),
                    time_conversion.DateTime(2001,1,16,8,51,40).epoch(),
                    time_conversion.DateTime(2001,1,16,9,9,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,12,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,15,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,18,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,21,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,24,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,27,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,30,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,33,00).epoch(),
                    time_conversion.DateTime(2001,1,16,9,50,20).epoch(),
                    time_conversion.DateTime(2001,1,16,9,53,20).epoch(),
                    time_conversion.DateTime(2001,1,16,9,56,20).epoch(),
                    time_conversion.DateTime(2001,1,16,9,59,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,2,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,5,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,8,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,11,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,14,20).epoch(),
                    time_conversion.DateTime(2001,1,16,10,31,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,34,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,37,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,40,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,43,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,46,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,49,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,52,40).epoch(),
                    time_conversion.DateTime(2001,1,16,10,55,40).epoch(),
                    time_conversion.DateTime(2001,1,16,11,13,00).epoch(),
                    time_conversion.DateTime(2001,1,16,11,16,00).epoch(),
                    time_conversion.DateTime(2001,1,16,11,22,00).epoch(),
                    time_conversion.DateTime(2001,1,16,11,25,00).epoch(),
                    time_conversion.DateTime(2001,1,16,11,28,00).epoch(),
                    time_conversion.DateTime(2001,1,16,20,34,20).epoch(),
                    time_conversion.DateTime(2001,1,16,20,51,40).epoch(),
                    time_conversion.DateTime(2001,1,16,20,54,40).epoch(),
                    time_conversion.DateTime(2001,1,16,20,57,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,00,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,3,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,6,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,9,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,12,40).epoch(),
                    time_conversion.DateTime(2001,1,16,21,15,40).epoch(),
]

folder_path = 'ObservationsProcessed/CurrentProcess'
raw_observation_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

#Set up environment
bodies_to_create = ["Sun",'Saturn', "Earth",'Jupiter', 'Io', 'Europa','Ganymede','Callisto','Amalthea']



# Create default body settings
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

#For propagated moons, voor initial state, Nan lijst veranderen
body_settings.get('Amalthea').ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(environment_setup.ephemeris.direct_spice("Jupiter",global_frame_orientation,"Amalthea"),t_end_before-12*7200,t_end+12*7200,1*60,interpolators.lagrange_interpolation(4))         #Check number of points used in interpolating


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
environment_setup.add_radiation_pressure_interface(bodies, "Amalthea", radiation_pressure_settings)

obstimes = []

observation_set_list = []
observation_settings_list = []
observation_simulation_settings = []
noise_level = []
arc_start_times = []
bias_values = []

#Start loop over every csv in folder
for file in raw_observation_files:
    arc_start_times_local = []
    bias_values_local = []
    #Reading information from file name
    string_to_split = file.split("/")[-1]
    split_string = string_to_split.split("_")

    Moon = int(split_string[0])
    Observatory = split_string[1]
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
            uncertainty_ra = float(row[3])
            uncertainty_dec = float(row[4])

            # Arc cut-off
            if arc_times and (time - arc_times[0] > arc_duration):
                # Calculate and append average uncertainties for the completed arc
                avg_uncertainty_ra = sum(arc_uncertainties_ra) / len(arc_uncertainties_ra)
                avg_uncertainty_dec = sum(arc_uncertainties_dec) / len(arc_uncertainties_dec)
                n = len(arc_uncertainties_ra)
                noise_level.extend([avg_uncertainty_ra*np.sqrt(n), avg_uncertainty_dec*np.sqrt(n)] * n)
                
                arc_start_times.append(arc_times[0]-60)
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


        arc_start_times.append(arc_times[0]-60)
        arc_start_times_local.append(arc_times[0]-60)
        bias_values.append(np.asarray([avg_uncertainty_ra, avg_uncertainty_dec]))
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
    observation_settings_list.append(observation.angular_position(link_definition,bias_settings=bias))                 


#Voyager simulation
### Create Link Ends for the Moons

link_ends_amalthea = dict()
link_ends_amalthea[estimation_setup.observation.observed_body] = estimation_setup.observation.\
    body_origin_link_end_id('Amalthea')
link_definition_amalthea = estimation_setup.observation.LinkDefinition(link_ends_amalthea)

link_definition_dict = {
    'Amalthea': link_definition_amalthea,
}


### Observation Model Settings
position_observation_settings = estimation_setup.observation.cartesian_position(link_definition_amalthea)
observation_settings_list.append( position_observation_settings)
observation_settings_list.append( position_observation_settings)
observation_settings_list.append( position_observation_settings)
observation_settings_list.append( position_observation_settings)


# picture epochs
observation_times = np.asarray(voyager_obstimes)
observation_times_galileo = np.asarray(galileo_obstimes)
observation_times_galileoCA = np.asarray(galileoCA_obstimes)
observation_times_cassini = np.asarray(cassini_obstimes)

observation_simulation_settings = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_dict['Amalthea'],
    observation_times,
    reference_link_end_type=estimation_setup.observation.observed_body)

observation_simulation_settings_galileo = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_dict['Amalthea'],
    observation_times_galileo,
    reference_link_end_type=estimation_setup.observation.observed_body)

observation_simulation_settings_galileoCA = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_dict['Amalthea'],
    observation_times_galileoCA,
    reference_link_end_type=estimation_setup.observation.observed_body)

observation_simulation_settings_cassini = estimation_setup.observation.tabulated_simulation_settings(
    estimation_setup.observation.position_observable_type,
    link_definition_dict['Amalthea'],
    observation_times_cassini,
    reference_link_end_type=estimation_setup.observation.observed_body)

#
noise_voyager = 1.0E4
noise_galileo = 2.0E4
noise_galileoCA = 1E3
noise_cassini = 5E3

noise_level.extend([noise_voyager,noise_voyager,noise_voyager] * 19)
noise_level.extend([noise_galileo,noise_galileo,noise_galileo] * 26)
noise_level.extend([noise_galileoCA,noise_galileoCA,noise_galileoCA] * 4)
noise_level.extend([noise_cassini,noise_cassini,noise_cassini] * 286)


observation.add_gaussian_noise_to_observable(
    [observation_simulation_settings],
    noise_voyager,
    estimation_setup.observation.position_observable_type
)

observation.add_gaussian_noise_to_observable(
    [observation_simulation_settings_galileo],
    noise_galileo,
    estimation_setup.observation.position_observable_type
)

observation.add_gaussian_noise_to_observable(
    [observation_simulation_settings_galileoCA],
    noise_galileoCA,
    estimation_setup.observation.position_observable_type
)

observation.add_gaussian_noise_to_observable(
    [observation_simulation_settings_cassini],
    noise_cassini,
    estimation_setup.observation.position_observable_type
)

ephemeris_observation_simulators = estimation_setup.create_observation_simulators(
    [position_observation_settings], bodies)
# Get ephemeris states as ObservationCollection
print('Simulating SC data')
ephemeris_satellite_states = estimation.simulate_observations(
    [observation_simulation_settings],
    ephemeris_observation_simulators,
    bodies)    

ephemeris_satellite_states_galileo = estimation.simulate_observations(
    [observation_simulation_settings_galileo],
    ephemeris_observation_simulators,
    bodies) 

ephemeris_satellite_states_galileoCA = estimation.simulate_observations(
    [observation_simulation_settings_galileoCA],
    ephemeris_observation_simulators,
    bodies) 

ephemeris_satellite_states_cassini = estimation.simulate_observations(
    [observation_simulation_settings_cassini],
    ephemeris_observation_simulators,
    bodies) 


inner_dict = next(iter(ephemeris_satellite_states.sorted_observation_sets.values()))
inner_dict_galileo = next(iter(ephemeris_satellite_states_galileo.sorted_observation_sets.values()))
inner_dict_galileoCA = next(iter(ephemeris_satellite_states_galileoCA.sorted_observation_sets.values()))
inner_dict_cassini = next(iter(ephemeris_satellite_states_cassini.sorted_observation_sets.values()))

# Access the only list (or set) in this dictionary
observation_list = next(iter(inner_dict.values()))
observation_list_galileo = next(iter(inner_dict_galileo.values()))
observation_list_galileoCA = next(iter(inner_dict_galileoCA.values()))
observation_list_cassini = next(iter(inner_dict_cassini.values()))


# Access the only SingleObservationSet in the list (or set)
voyager_obs_set = observation_list[0]  # or next(iter(observation_list)) if it's a set
galileo_obs_set = observation_list_galileo[0]
galileoCA_obs_set = observation_list_galileoCA[0]
cassini_obs_set = observation_list_cassini[0]

### Observationcollection -> observation set uitpakken 1ste entry telkens

observation_set_list.append(voyager_obs_set)
observation_set_list.append(galileo_obs_set)
observation_set_list.append(galileoCA_obs_set)
observation_set_list.append(cassini_obs_set)


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
full_initial_state = []
for i in bodies_to_propagate:
    #initial_state = spice.get_body_cartesian_state_at_epoch(i,"Jupiter","ECLIPJ2000",'None',t_0)
    initial_state = [-1.71358285e+08,  6.02868819e+07,  8.61709429e+05 ,-8.70255003e+03, -2.49379545e+04, -1.02136012e+03]  #1990
    full_initial_state.append(initial_state)
full_initial_state = np.concatenate(full_initial_state)



### Create the integrator settings
"""
# Use fixed step-size integrator (RKDP8) with fixed time-step of 30 minutes
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
    """
    By combining all of the above-defined settings we can define the settings for the propagator to simulate the orbit of `Delfi-C3` around Earth. A termination condition needs to be defined so that the propagation stops as soon as the specified end epoch is reached. Finally, the translational propagator's settings are created.
    """

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

    # Create the parameters that will be estimated
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    original_parameter_vector = parameters_to_estimate.parameter_vector

    # Create the estimator
    estimator = numerical_simulation.Estimator(
        bodies,
        parameters_to_estimate,
        observation_settings_list,
        propagator_settings)

    precision = 5E+5
    hours = 1
    inv_pre_r = 1/(precision)**2
    inv_pre_v = 1/(precision/(hours*3600))**2
    inverse_apriori_covariance = np.asarray([[inv_pre_r,0,0,0,0,0],[0,inv_pre_r,0,0,0,0],[0,0,inv_pre_r,0,0,0],[0,0,0,inv_pre_v,0,0],[0,0,0,0,inv_pre_v,0],[0,0,0,0,0,inv_pre_v]])
    convergence_checker = estimation.estimation_convergence_checker(maximum_iterations = 5)

    print('precision is: ', precision)
    print('hours is: ', hours)

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(
        observations,inverse_apriori_covariance=inverse_apriori_covariance, convergence_checker=convergence_checker)
    # estimation_input = estimation.EstimationInput(
    #     observations, convergence_checker=convergence_checker)

    # Set methodological options
    #estimation_input.define_estimation_settings(reintegrate_variational_equations=False)
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
    print('jupiter_gravitational_parameter is', jupiter_gravitational_parameter)
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

    rsw_states = []
    div = 2000
    print('expected conversions = ', math.floor(len(state_history_difference)/div))
    
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
plt.savefig(os.path.join(folder_name, "correlation_RSW.png"), dpi=300, bbox_inches='tight')
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
ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,2] * 1E-3,
         label=r'w', c='#EC6842')
ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,0] * 1E-3,
         label=r'r', c='#A50034')
ax1.plot(time2plt[div:-div:div], np.vstack(rsw_states)[:,1] * 1E-3,
         label=r's', c='#0076C2')

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





# # Residual iteration
# for iteration in range(5): 
#     fig, ax1 = plt.subplots(1, 1)

#     ax1.set_title(iteration)
#     residuals_iteration = full_residual[:,iteration]
#     rms_list = []
#     residual_RA_it = residuals_iteration[0:-57:2]
#     residual_DEC_it = residuals_iteration[1:-57:2]
#     for i in range(len(residual_RA)):
#         # Calculate the squares of the i-th elements from each list
#         squared_x = residual_RA_it[i]**2
#         squared_y = residual_DEC_it[i]**2

#         # Calculate the mean of the squared values
#         mean_squared = (squared_x + squared_y) / 2

#         # Calculate the RMS value by taking the square root of the mean squared value
#         rms = math.sqrt(mean_squared)

#         # Append the RMS value to the list
#         rms_list.append(rms)
#     ax1.scatter(obstime2plt, residual_RA_it,
#          label='RA')
#     #ax1.scatter(obstime2plt, residual_DEC_it,
#     #     label='DEC')    
        
#     ax1.xaxis.set_major_locator(mdates.YearLocator(10))
#     ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     ax1.set_ylabel(r'angular position residual [rad]')
#     ax1.set_xlabel('Time')
#     ax1.legend()
#     plt.grid()
#     plt.savefig(os.path.join(folder_name, str(iteration)), dpi=300, bbox_inches='tight')
#     plt.show()

# rms_list = []
# for i in range(len(residual_RA)):
#         # Calculate the squares of the i-th elements from each list
#     squared_x = residual_RA[i]**2
#     squared_y = residual_DEC[i]**2

#     # Calculate the mean of the squared values
#     mean_squared = (squared_x + squared_y) / 2

#     # Calculate the RMS value by taking the square root of the mean squared value
#     rms = math.sqrt(mean_squared)

#     # Append the RMS value to the list
#     rms_list.append(rms)
# plt.plot(observation_times,rms_list, label='rms')
# plt.legend()
# plt.grid()
# plt.xlabel('time [s since J2000]')
# plt.ylabel('rms of the position')
# plt.savefig(os.path.join(folder_name, "plot6.png"), dpi=300, bbox_inches='tight')
# plt.show()


# fig, ax1 = plt.subplots(1, 1)

# ax1.set_title(r'RMS of position residual')
# ax1.scatter(obstime2plt, rms_list,
#          label=r'Amalthea', c='#A50034')
# ax1.xaxis.set_major_locator(mdates.YearLocator(10))
# ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.set_ylabel(r'RMS of angular position residual [rad]')
# ax1.set_xlabel('Time')
# ax1.legend()
# ax1.grid()
# plt.savefig(os.path.join(folder_name, "RMS_Residual.png"), dpi=300, bbox_inches='tight')
# plt.show()


# fig, ax1 = plt.subplots(1, 1)

# ax1.set_title(r'Angular position residual')
# ax1.scatter(obstime2plt, residual_RA,
#          label=r'Amalthea RA', c='#A50034')
# ax1.scatter(obstime2plt, residual_DEC,
#          label=r'Amalthea DEC', c='#0076C2')
# ax1.xaxis.set_major_locator(mdates.YearLocator(10))
# ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax1.set_ylabel(r'Angular position residual [rad]')
# ax1.set_xlabel('Time')
# ax1.legend()
# ax1.grid()
# plt.savefig(os.path.join(folder_name, "Residuals.png"))
# plt.show()

with open( folder_name + 'residual_output' +'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header row
    writer.writerow(['Seconds since J2000', 'RA residual [rad]','DEC residual [rad]'])
    
    # Write each matching name and result to a new row
    for j in range(len(residual_RA)):
        writer.writerow([obs_julian_seconds[j], residual_RA[j],residual_DEC[j]])

script_name = os.path.basename(__file__)
shutil.copy(script_name, os.path.join(folder_name, script_name))



print('Done')