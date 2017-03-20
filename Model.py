########################################
# Mario Rosasco, 2017
######################################## 

from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import create_utils
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.core import swc
from allensdk.ephys.ephys_features import *
import matplotlib.pyplot as plt
import numpy as np
import subprocess, os

class Model(object):
    '''
    Class to manage:
        - retrieving/setting up the data files from the AllenSDK
        - running NEURON simulations
        - fitting model parameters to experimental data
        
    Parameters
    ----------
    model_id: int
        the id number that is associated with the biophysical model. Note that this is NOT the same as the cell id.
        
    cache_stim: boolean
        if set to true, this will cause the potentially-large NWB experimental data to be downloaded when the model is retrieved.
    '''

    def __init__(self, model_id, cache_stim = False):
        self.model_id = model_id
        self.bp = BiophysicalApi()
        self.model_dir = os.path.join(os.getcwd(),'model_'+str(model_id),'')
        self.bp.cache_stimulus = cache_stim
        self.h = None
        self.utils = None
        self.description = None
        self.section_output = {}
        ############################
        self.reference_sweep = None
        self.reference_output = None
        self.j_history = []
        self.theta_history = []
        self.theta = None
        self.theta_name = ''
        self.fit_section = None
        
        
    def __compile_modfiles(self):
        '''
        Helper function to compile modfiles in the ./modfiles/ folder
        '''
        #compilation on Windows
        if os.name == 'nt' and not os.path.isfile('nrnmech.dll'):
            # note that doing this programmatically is simpler on a *nix system, because of the nrnivmodl program.
            # For compilation on Windows, the solution is usually to call 
            # %NEURONHOME%\bin\nrniv -nopython %NEURONHOME%\lib\hoc\mknrndll.hoc
            # which brings up a GUI to select the modfiles folder, then produces a nrnmech.dll file.
            # Here I'm directly calling the shell scripts that are called from the NEURON GUI. This
            # works on my machines where 64-bit NEURON v7.4 was installed from binaries, but I would want to try it on
            # other types of installations (eg: compiled from source) before I call it "stable."
            
            shLoc = os.path.join(os.environ['NEURONHOME'], 'mingw', 'bin', 'sh.exe')
            mknrnLoc = os.path.join(os.environ['NEURONHOME'], 'lib', 'mknrndll.sh')
            nrnLoc = '/' + os.environ['NEURONHOME'].replace(':', '')

            print "Attempting to compile modfiles..."
            print [shLoc, mknrnLoc, nrnLoc]
            
            os.chdir('modfiles')
            subprocess.call([shLoc, mknrnLoc, nrnLoc])
            os.chdir('..')
            
            # copy the compiled file into the current directory to be used when we run NEURON
            dllLoc = os.path.join('modfiles', 'nrnmech.dll')
            if os.system("copy %s %s" %(dllLoc, '.')) == 0:
                pass
            else:
                print "Could not compile modfiles using"
                print shLoc, mknrnLoc, nrnLoc
                print "Windows operating system detected. Please confirm that NEURON is installed appropriately for this usage."
                quit()
            
        # compilation on *nix systems
        elif os.name != 'nt' :
            # compile the modfiles
            subprocess.call(['nrnivmodl', 'modfiles'])
            
            
    def init_model(self):
        '''
        Retrieves the model information for self.model_id, and initializes the modeling and visualization data.
        '''
        # Retrieve the model data
        try:
            self.bp.cache_data(self.model_id, working_directory = self.model_dir)
        except:
            print "Could not access biophysical data for model " + str(self.model_id)
            print "Please confirm model ID and try again."
            quit()
            
        # Compile the mod files for the current model
        curr_dir = os.getcwd()
        os.chdir(self.model_dir)
        self.__compile_modfiles()
            
        # set up the NEURON environment using the manifest file
        description = Config().load('manifest.json')
        
        # Specify model type (perisomatic/all-active)
        currType = description.data['biophys'][0]['model_type']
        
        # initialize the Hoc interpreter we'll use to affect the model
        utils = create_utils(description, model_type = currType)

        # configure the model for NEURON
        morphologyLoc = description.manifest.get_path('MORPHOLOGY')
        utils.generate_morphology(morphologyLoc.encode('ascii', 'ignore'))
        utils.load_cell_parameters()
        
        # store the hoc interpreter and cell information in the object instance
        self.h = utils.h
        self.utils = utils
        self.description = description
        
        # set up a monitor to log simulation time
        self.section_output["t"] = self.h.Vector()
        self.section_output["t"].record(self.h._ref_t)
        
        # return to the previous directory
        os.chdir(curr_dir)
        
    def run_test_pulse(self, amp=0.18, delay=1000.0, dur=1000.0, tstop=3000.0):
        '''
        Performs a simple simulation with a square current injection pulse
        
        Parameters
        ----------
        amp: float
            amplitude of stimulus (in nA)
        
        delay: float
            length of time to wait before applying stimulus (in ms)
            
        dur: float
            duration of the stimulus (in ms)
        
        tstop: float
            length of entire recording (in ms)
        
        Returns
        -------
        vector 
            where vector['v'] is an array of floats indicating the response in mV
            and vector['t'] is an array of floats indicating the time of each response point in ms.
        '''
        # generate response using a simple current-clamp stimulus
        stim = self.h.IClamp(self.h.soma[0](0.5))
        stim.amp = amp
        stim.delay = delay
        stim.dur = dur

        self.h.tstop = tstop

        # returns vector where vec['v'] = voltage, vec['t'] = time
        self.model_output = self.utils.record_values()

        self.h.finitialize()
        self.h.run()
        
        return self.model_output
        
    def long_square(self, pulse_amp):
        '''
        Convenience function to run a model with stimulus of type "Long Square"
        
        Parameters
        ----------
        pulse_amp: float
            amplitude of stimulus (in nA)
            
        Returns
        -------
        vector 
            where vector['v'] is an array of floats indicating the response in mV
            and vector['t'] is an array of floats indicating the time of each response point in ms.
        '''
        return self.run_test_pulse(amp=pulse_amp, delay=100.0, dur=1001.0, tstop=1200.0)
        
    def run_nwb_pulse(self, sweep_index):
        '''
        Performs a simulation using the NWB stimulus for the indicated sweep
        
        Parameters
        ----------
        sweep_index: int
            indicates which sweep in the NWB file to use to generate the stimulus
        
        Returns
        -------
        vector 
            where vector['v'] is an array of floats indicating the response in mV
            and vector['t'] is an array of floats indicating the time of each response point in ms.
        '''
        if not self.bp.cache_stimulus:
            print "Current model was not instantiated with NWB data cached. Please reload the current model and cache experimental stimulus data."
            return
            
        # set up the stimulus using the manifest data
        stimulus_path = os.path.join(self.model_dir, self.description.manifest.get_path('stimulus_path'))
        run_params = self.description.data['runs'][0]
        sweeps = run_params['sweeps']
        
        if not(sweep_index in sweeps):  
            print "Specified sweep index is not present in the current NWB dataset."
            return
        
        # run model with sweep stimulus
        self.utils.setup_iclamp(stimulus_path, sweep=sweep_index)
        self.model_output = self.utils.record_values()

        self.h.finitialize()
        self.h.run()
        
        return self.model_output
        
    def plot_output(self):
        '''
        Uses matplotlib to plot the data stored in self.model_output
        '''
        junction_potential = self.description.data['fitting'][0]['junction_potential']
        plt.plot(self.model_output['t'], np.array(self.model_output['v']) - junction_potential)
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.show()
        
    def plot_fit(self):
        '''
        Uses matplotlib to plot the results of parameter fitting, 
        using the data stored in self.j_history and self.theta_history
        '''
        fig, theta_ax = plt.subplots()
        cycles = range(len(self.theta_history))
        theta_ax.plot(cycles, self.theta_history, 'b-')
        theta_ax.set_ylabel('Theta', color='b')
        
        theta_ax.set_xlabel('Cycle Number')
        j_ax = theta_ax.twinx()
        
        j_ax.plot(cycles, self.j_history, 'r-')
        j_ax.set_ylabel('Cost', color='r')
        
        plt.tight_layout()
        plt.show()
        
    def get_reconstruction(self):
        '''
        Checks for a morphology .swc file associated with the biophysical model.
        If such a file is found, uses the AllenSDK swc helper functions to return a Morphology instance.
        
        Returns
        -------
        Morphology
            a morphology instance if a morphology file is found associated with the model. None otherwise.
        '''
        file_name = self.bp.ids['morphology'].values()[0]
        if len(file_name) > 0:
            file_name = os.path.join(os.getcwd(), self.model_dir, file_name)
            return swc.read_swc(file_name)
        else:
            print "Could not find morphology file for model ", self.model_id, ", please check directory structure and try again."
            return None
            
    def monitor_section_voltage(self, sec_type, sec_index):
        '''
        Sets up a voltage monitor for the specified section.
        
        Parameters
        ----------
        sec_type: string
            either 'soma', 'axon', or 'dend'
            
        sec_index: int
            indicates which section of sec_type to monitor the output from
        '''
        sec_name = sec_type + '[' + str(sec_index) + ']'
        
        self.section_output[sec_name] = self.h.Vector()
        
        if sec_type=='dend':
            self.section_output[sec_name].record(self.h.dend[sec_index](0.5)._ref_v)
        elif sec_type=='axon':
            self.section_output[sec_name].record(self.h.axon[sec_index](0.5)._ref_v)
        elif sec_type=='soma':
            self.section_output[sec_name].record(self.h.soma[sec_index](0.5)._ref_v)
        else:
            print "Could not attach voltage monitor for section", sec_type, "[", sec_index, "]"
            
            
    def set_fit_section(self, section_name, section_id=0):
        '''
        Sets the neuron compartment in which to perform parameter fitting.
        
        Parameters
        ----------
        section_name: string
            either 'soma', 'axon', or 'dend'
            
        section_id: int
            index of the section of the type indicated by section_name
        '''
        if section_name == 'soma':
            self.fit_section = self.h.soma[section_id]
        elif section_name == 'axon':
            self.fit_section = self.h.axon[section_id]
        elif section_name == 'dend':
            self.fit_section = self.h.dend[section_id]
        else:
            print "Could not set section for parameter fitting. Invalid section name."
            self.fit_section = None
            
    def set_parameter_to_fit(self, theta_name):
        '''
        Sets the parameter within the current section to be fit to the data. 
        Note that this function must be called after defining which section the parameter applies to.
        
        Parameters
        ----------
        theta_name: string
            The name of the parameter (mechanism in NEURON terminology) to fit 
        '''
        try:
            self.theta_name = theta_name
            if theta_name == 'gbar_NaV':
                self.theta = self.fit_section.gbar_NaV
            elif theta_name == 'gbar_Kv2like':
                self.theta = self.fit_section.gbar_Kv2like
            elif theta_name == 'gbar_Kv3_1':
                self.theta = self.fit_section.gbar_Kv3_1
            else:
                print "Parameter fitting for mechanism", theta_name, "not yet implemented."
                self.theta = None
                self.theta_name = ''
        except NameError:
            print "Could not set parameter for fitting. Current section does not have a parameter/mechanism named", theta_name
            self.theta = None
        
    def change_fit_parameter(self, new_theta):
        '''
        Changes the value of the fit parameter.
        Note that this must be called after set_parameter_to_fit().
        
        Parameters
        ----------
        new_theta: float
            The new value of the parameter to assign.
        '''
        theta_name = self.theta_name
        if theta_name == 'gbar_NaV':
            self.fit_section.gbar_NaV = new_theta
            self.theta = new_theta
        elif theta_name == 'gbar_Kv2like':
            self.fit_section.gbar_Kv2like = new_theta
            self.theta = new_theta
        elif theta_name == 'gbar_Kv3_1':
            self.fit_section.gbar_Kv3_1 = new_theta
            self.theta = new_theta
        elif theta_name == None:
            print "Attempting to assign a model parameter without specifying which parameter. Please call set_parameter_to_fit() first."
        else:
            print "Parameter fitting for mechanism", theta_name, "not yet implemented."
            self.theta = None
            self.theta_name = ''
            
    def set_reference_sweep(self, ref_index):
        '''
        Checks to see if the given index refers to one of the sweeps associated with the model.
        If it is, sets the sweep with this index as the reference sweep.
        
        Parameters
        ----------
        ref_sweep: int
            the index of the reference sweep in the NWB file to compare the model to
        '''
        if not self.bp.cache_stimulus:
            print "Current model was not instantiated with NWB data cached. Please reload the current model and cache experimental stimulus data."
            return
            
        # check for the existence of the requested sweep
        stimulus_path = os.path.join(self.model_dir, self.description.manifest.get_path('stimulus_path'))
        run_params = self.description.data['runs'][0]
        sweeps = run_params['sweeps']
        
        if not(ref_index in sweeps):  
            print "Specified sweep index is not present in the current NWB dataset."
            self.reference_sweep = None
            return
        
        self.reference_sweep = ref_index
            
            
    # Note that the allensdk.ephys.ephys_features.average_rate() function
    # estimates average spike firing frequency by calculating (num spikes)/time.
    # For sufficiently long recordings this should converge on the correct
    # value, but for short pulses with a small number of spikes, you run 
    # into a granularity issue.
    def average_rate_from_delays(self, t, spikes, start, end):
        '''
        Calculate average firing rate during interval between 'start' and 'end',
        based on interspike interval durations.
        
        Parameters
        ----------
        t: numpy array of floats
            times for each point in the recording (in seconds)
            
        spikes: numpy array of ints
            indices into the time/response arrays where a putative spike was detected
            
        start: float
            start of time window for spike detection (in seconds)
            
        end: float
            end of time window for spike detection (in seconds)
            
        Returns
        -------
        float
            average firing rate in spikes/second
        '''
        if start is None:
            start = t[0]

        if end is None:
            end = t[-1]
        
        n_spikes = len(spikes)
        if n_spikes == 0:
            return 0.0
        
        prev_spike_t = start
        sum_delays = 0.0
        
        for spike_index in spikes:
            curr_spike_t = t[spike_index]
            sum_delays += curr_spike_t - prev_spike_t 
            prev_spike_t = curr_spike_t
            
        return (float(n_spikes) / sum_delays)
            
    def __J(self, model_val, exp_val, method='MSE'):
        '''
        Cost function to compare distance of model value from experimental value
        
        Parameters
        ----------
        model_val: numeric
            the target value predicted by the model
            
        exp_val: numeric
            the target value observed in the experiment
            
        method: string
            defines the type of cost function to use. Currently only MSE is implemented
            
        Returns
        -------
        float
            the loss/cost associated with the difference between the model and the 
            experimental values
        '''
        if method == 'MSE':
            return float(model_val - exp_val)**2
            
    def set_up_objective(self, measure='spike frequency'):
        '''
        Prepares the model for parameter optimization by assigning the output measure to be used in the cost function.
        
        Parameters
        ----------
        measure: string
            Name of the output measure to be used in optimization. Currently only 'spike frequency' is implemented.
        '''
        if (measure == 'spike frequency'):
            # get the experimental data from the NWB file 
            data_set = NwbDataSet(os.path.join(self.model_dir, self.description.manifest.get_path('stimulus_path')))
            spike_times = data_set.get_spike_times(self.reference_sweep)
            
            # calculate firing frequency for the NWB data
            sum_intervals = 0.0
            for i in range(len(spike_times)-1):
                sum_intervals += (spike_times[i+1] - spike_times[i])
                
            self.reference_output = len(spike_times) / sum_intervals
        else:
            print "Model fitting using the output measure", measure, "has not been implemented yet."
        
    def objective_function(self, theta, measure='spike frequency', stim_amplitude = 0.21):
        '''
        Performs one simulation with the currently assigned theta set to the indicated value, 
        then returns the cost compared to the reference output based on the indicated cost measure.
        
        Parameters
        ----------
        theta: float
            The value to assign to the current theta (stored at Model.theta_name)
            
        measure: string
            Name of the output measure to be used in optimization. Currently only 'spike frequency' is implemented.
            
        stim_amplitude: float
            The stimulation amplitude in nA to apply to simulate the reference sweep data.
            
        Returns
        -------
        cost: float
            A measure of difference between the reference sweep data and the current model
        '''
        if (measure == 'spike frequency'):
            if self.theta_name == 'gbar_NaV':
                self.fit_section.gbar_NaV = theta
            elif self.theta_name == 'gbar_Kv2like':
                self.fit_section.gbar_Kv2like = theta
            elif self.theta_name == 'gbar_Kv3_1':
                self.fit_section.gbar_Kv3_1 = theta
            else:
                print "Parameter fitting for mechanism", self.theta_name, "not yet implemented."
        
            # Run the model
            response = np.array(self.long_square(stim_amplitude)['v'])
            times = np.array(self.model_output['t']) / 1000.0
            startSec = 0.1
            endSec = 1.1
            spikes = detect_putative_spikes(response, times, startSec, endSec)
            avg_rate = self.average_rate_from_delays(times, spikes, startSec, endSec)
            
            cost = self.__J(avg_rate, self.reference_output)
            
            self.theta = theta
            self.theta_history.append(theta)
            self.j_history.append(cost)
            
            return cost
    
    def gradient_descent(self, alpha=0.001, epsilon=0.001, threshold=0.001, max_cycles=100):
        '''
        Attempts to minimize Model.objective_function() using an estimated gradient descent approach.
        
        Iteratively performs simulations to optimize the current theta, which must be assigned before 
        this function is called using a call to set_parameter_to_fit().
        
        Parameters
        ----------
        alpha: float
            The "learning rate" of the algorithm. Can be tuned empirically. 
            Lower values produce smoother, more stable optimization curves, but take longer to converge.
            
        epsilon: float
            Initial guess for amount to vary theta by. This will only be used in the initial estimate of 
            dJ/dTheta, and so should be set to a value that is small relative to theta.
            
        threshold: float
            The threshold for convergence. The function will return when the change in J for a given iteration falls below this value.
            
        max_cycles: int 
            The maximum number of iterations to perform. The function will return regardless of values of J after this many iterations. 
            
        Returns
        -------
        cost: float
            A measure of difference between the reference sweep data and the current model
        '''
        old_theta = self.theta
        old_cost = self.objective_function(old_theta)
        
        delta_cost = threshold + 1.0
        cycle = 0
        
        while (cycle < max_cycles) and (abs(delta_cost) > threshold):
            new_theta = old_theta + epsilon
            new_cost = self.objective_function(new_theta)
            
            delta_cost = old_cost - new_cost
            gradient = delta_cost / -epsilon
            epsilon = -alpha * gradient
            
            old_cost = new_cost
            old_theta = new_theta
            cycle += 1
            
            print "theta", old_theta, "cost", old_cost, "gradient", gradient
            
        
    