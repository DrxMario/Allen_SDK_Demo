#!/usr/bin/env python 

########################################
# Mario Rosasco, 2017
######################################## 
from Model import *
from Visualization import *
from scipy.optimize import minimize
from allensdk.ephys.ephys_features import detect_putative_spikes
import numpy as np
import sys

def main(modelID):

    print "Loading parameters for model", modelID
    selection=raw_input('Would you like to download NWB data for model? [Y/N] ')
    if selection[0] == 'y' or selection[0] == 'Y':
        currModel = Model(modelID, cache_stim = True)
    if selection[0] == 'n' or selection[0] == 'N':
        currModel = Model(modelID, cache_stim = False)
    currModel.init_model()
    
    while(True):
        print "Initialized biophysical model", modelID
        print '''
        Please select from the following options:
        1 - Run test pulse on model
        2 - Fit model parameter to data
        3 - Display static neuron model
        4 - Visualize model dynamics
        5 - Quit
        '''
        try:
            selection=int(raw_input('Please choose an option above: '))
        except ValueError:
            print "Invalid selection."
            continue
        
        # test pulse example        
        if selection == 1:
            # Run the model with a test pulse of the 'long square' type
            print "Running model with a long square current injection pulse of 210pA"
            output = currModel.long_square(0.21)
            currModel.plot_output()
        
        # fit parameter example    
        elif selection == 2: 
            if not currModel.bp.cache_stimulus:
                print "Current model was not instantiated with NWB data cached. Please reload the current model and cache experimental stimulus data."
                continue
            
            print "Fitting somatic sodium conductance for model", modelID, "to experimental data in sweep 41."
            print "Please be patient, this may take some time."
            
            # Define which section and which parameter to fit.
            # Here we'll fit the somatic sodium conductance.
            currModel.set_fit_section('soma', 0)
            currModel.set_parameter_to_fit('gbar_NaV')
            
            # Running the model with an NWB pulse as stimulus takes a 
            # very long time because of the high sampling rate.
            # As a computationally-cheaper approximation for stimuli of
            # type Long Square pulse, we can rebuild the stimulus with the
            # default (lower) sampling rate in h.IClamp
            
            # currModel.run_nwb_pulse(41) # too slow
            output = currModel.long_square(0.21)
            
            # Set the experimental reference sweep and set up the variables for the objective function
            currModel.set_reference_sweep(ref_index=41)
            currModel.set_up_objective(measure='spike frequency')
            
            # Use SciPy's minimize functions to fit the specified parameter
            #results = minimize(currModel.objective_function, currModel.theta, method='Nelder-Mead', tol=1e-3)
            #results = minimize(currModel.objective_function, currModel.theta, method='Powell', tol=1e-3)
            
            #results = minimize(currModel.objective_function, currModel.theta, method='COBYLA', tol=1e-5)
            currModel.gradient_descent(alpha=0.00005, epsilon=0.001, threshold=0.01, max_cycles=1000)
            currModel.plot_fit()
            
            output = currModel.long_square(0.21)
            currModel.plot_output()
            times = np.array(output['t'])/1000
            spikes = detect_putative_spikes(np.array(output['v']), times, 0.1, 1.1)
            avg_rate = currModel.average_rate_from_delays(times, spikes, 0.1, 1.1)
            print "spike rate for theta of", currModel.theta, ":", avg_rate
            
        # static visualization example
        elif selection == 3:
            run_visualization(currModel)
        
        elif selection == 4:
            run_visualization(currModel, show_simulation_dynamics = True)
            
        elif selection == 5:
            quit()
            
        else:
            print "Invalid selection."
            continue
    
def run_visualization(currModel, show_simulation_dynamics = False):
    print "Setting up visualization..."
            
    morphology = currModel.get_reconstruction()

    # Prepare model coordinates for uploading to OpenGL.
    tempIndices = []
    tempVertices = []
    n_index = 0
    tempX = []
    tempY = []
    tempZ = []
    tempCol = []
    
    if not show_simulation_dynamics:
        print '''
        Soma - Red
        Axon - Green
        Dendrites - Blue
        Apical Dendrites - Purple'''
        
    # array of colors to denote individual compartment types
    compartmentColors=[[0.0,0.0,0.0,0.0], # padding for index convenience
                [1.0, 0.0, 0.0, 1.0], #1: soma - red
                [0.0, 1.0, 0.0, 1.0], #2: axon - green
                [0.0, 0.0, 1.0, 1.0], #3: dendrites - blue
                [1.0, 0.0, 1.0, 1.0]] #4: apical dendrites - purple
    color_dim = 4
    
    # used to set up section monitoring for visualization of dynamics
    compartmentNames=['none', # padding for index convenience
    'soma', #1: soma
    'axon', #2: axon
    'dend', #3: dendrites - blue
    'dend'] #4: apical dendrites - purple
    
    sectionIndices=[0,0,0,0,0]
    segmentsPerSection = {}
    sec_name = ''

    # initialize storage arrays for each vertex. 
    index = 0
    n_compartments = len(morphology.compartment_list)
    tempX = [0] * n_compartments
    tempY = [0] * n_compartments
    tempZ = [0] * n_compartments
    tempCol = [0] * n_compartments * color_dim
    
    for n in morphology.compartment_list:
        # add parent coords
        tempX[n['id']] = n['x']
        tempY[n['id']] = -n['y']
        tempZ[n['id']] = n['z']
        
        # add color data for parent
        col_i = 0
        offset = n['id']*color_dim
        for cval in compartmentColors[n['type']]:
            tempCol[offset+col_i] = cval
            col_i += 1
        
        # if at a branch point or an end of a section, set up a vector to monitor that segment's voltage
        type = compartmentNames[n['type']]
        sec_index = sectionIndices[n['type']]
        
        if not (len(morphology.children_of(n)) == 1): #either branch pt or end

            sec_name = type + '[' + str(sec_index) + ']'
            sectionIndices[n['type']] += 1
            
            currModel.monitor_section_voltage(type, sec_index)
            segmentsPerSection[sec_name] = 1
            
        else:
            segmentsPerSection[sec_name] += 1
        
        index += 1
                
        for c in morphology.children_of(n):
            
            # add child coods
            tempX[c['id']] = c['x']
            tempY[c['id']] = -c['y']
            tempZ[c['id']] = c['z']
            
            # add index data:
            # draw from parent to child, for each child
            tempIndices.append(n['id'])
            tempIndices.append(c['id'])
            index += 1

            # add color data for child
            col_i = 0
            offset = c['id']*color_dim
            for cval in compartmentColors[c['type']]:
                tempCol[offset+col_i] = cval
                col_i += 1
                
            segmentsPerSection[sec_name] += 1
        
    # get ranges for scaling
    maxX = max(tempX)
    maxY = max(tempY)
    maxZ = max(tempZ)
    minX = min(tempX)
    minY = min(tempY)
    minZ = min(tempZ)
    xHalfRange = (maxX - minX)/2.0
    yHalfRange = (maxY - minY)/2.0
    zHalfRange = (maxZ - minZ)/2.0
    longestDimLen = max(xHalfRange, yHalfRange, zHalfRange)
    
    # center coords about 0,0,0, with range -1 to 1
    tempX = [((((x-minX)*(2*xHalfRange))/(2*xHalfRange)) - xHalfRange)/longestDimLen for x in tempX]
    tempY = [((((y-minY)*(2*yHalfRange))/(2*yHalfRange)) - yHalfRange)/longestDimLen for y in tempY]
    tempZ = [((((z-minZ)*(2*zHalfRange))/(2*zHalfRange)) - zHalfRange)/longestDimLen for z in tempZ]
    
    # convert everything to a numpy array so OpenGL can use it
    indexData = np.array(tempIndices, dtype='uint16')
    vertexData = np.array([tempX,tempY,tempZ], dtype='float32')
    tempCol  = np.array(tempCol, dtype='float32')
    vertexData = np.append(vertexData.transpose().flatten(), tempCol)    
    #################### /Preparing Model Coords
    
    # Set up the Visualization instance
    n_vertices = len(tempX)
    currVis = Visualization(data=vertexData, indices=indexData, nVert=n_vertices, colorDim=color_dim)
    
    if show_simulation_dynamics:
        currModel.run_test_pulse(amp=0.25, delay=20.0, dur=20.0, tstop=60.0)
        #currModel.plot_output() # uncomment this line to display the somatic potential over time before the visualization begins
        
        sectionOutput = currModel.section_output
        n_segments = n_vertices
        
        # set up looping color change data
        all_voltages = []
        n_pts = len(sectionOutput['t'])
        
        for t in range(n_pts): # for each timepoint...
            for key in sectionOutput.keys(): # for each section...
                if key != 't':
                    for s in range(segmentsPerSection[key]): # for each segment...
                        all_voltages.append(sectionOutput[key][t]) # ...set up color for segment
                    
        all_voltages = np.array(all_voltages, dtype='float32')
        all_voltages -= min(all_voltages)
        all_voltages /= max(all_voltages)
        temp_col = []
        n_pts = 0
        for v in all_voltages:
            temp_col.append(v)
            temp_col.append(0.0)
            temp_col.append(1.0-v)
            temp_col.append(1.0)
            n_pts += 1
        voltage_col = np.array(temp_col, dtype='float32')
    
    
        currVis.change_color_loop(voltage_col, n_colors=n_segments, n_timepoints=n_pts, offset=0, rate=0.10)
    
    currVis.run()
    

if __name__ == '__main__':
    if len(sys.argv) == 1: # no model ID passed as argument
        modelID = 497233230
    else:
        try:
            modelID=int(sys.argv[1])
        except ValueError:
            print "Could not interpret model ID. Initializing with example model 497233230"
            modelID = 497233230
        
    main(modelID)