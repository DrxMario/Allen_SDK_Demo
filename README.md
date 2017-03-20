# Allen_SDK_Demo
A small program to demonstrate various ways of using the Allen Institute for Brain Science's SDK
This program requires PyOpenGL, NEURON installed with Python as an interpreter, and the Allen SDK (and all
dependencies). It has been tested under 64-bit Windows 10 and Ubuntu 16.04.

# Installation and Demonstration
To launch a demonstration of the work described in this document, make sure the folder in which you'd like to run the program contains the files: 

* Model.py
* Visualization.py
* 170126-FragmentShader.frag
* 170126-VertexShader.vert
* demo.py

and run the demo.py program. This program can be launched with an Allen database biophysical model ID passed in as an argument. If no argument is passed, demo.py runs with all-active biophysical model 497233230 as an example. The program will then bring up a list of options that demonstrate various ways that the SDK can be used. Please note that an internet connection is required for this demo program to access the biophysical models through the Allen SDK. Please also note that option 2 ("Fit model parameter to data") involves running many simulations, and thus can take quite a while to finish executing.

# Running Simulations of Biophysical Models using Python and the Allen SDK
In order to provide an easily readable interface for performing modeling functions, the Model class was designed, contained in the Model.py module. A Model object is instantiated with a biophysical model id number from the Allen database, and upon initialization contains an instance of the Allen SDK BiophysicalAPI class. The BiophysicalAPI object is then used for model and cell data retrieval and NEURON model functionality.

After generating an instance of a Model object, the user should first call  Model.init_model(),  which will perform the necessary steps to set up the Model appropriately for the indicated model id. The steps are comparable to those [listed here](https://alleninstitute.github.io/AllenSDK/biophysical_models.html), with the exception of the compilation of NEURON mechanism files. Programmatic compilation of the mechanisms contained in the modfiles folder is easily achieved on Linux and comparable systems by calling
```Python
nrnivmodl modfiles
```
In contrast, on Windows systems a nrnmech.dll file must be generated from the compiled mechanisms and placed in the execution folder. This is usually achieved by using a graphical compilation utility. To compile the mechanisms programmatically on Windows, the Model class uses the Model.\_\_compile_modfiles() member function. This function is called by the init_model() function, and in turn calls the shell script interpreter included in mingw-based Windows distributions of NEURON to run the NEURON scripts that perform modfile compilation.This initialization should work for both perisomatic and all-active biophysical models. Once the biophysical
model is downloaded and the modfiles are compiled, the response to a long square pulse current injection stimulus can be simulated by calling
```Python
Model.long_square(pulse_amp)
```
where pulse amplitude is given in nA. This function is just a convenience wrapper on the similar
```Python
Model.run_test_pulse(amp, delay, dur, tstop)
```
that simulates the long square pulses used in many of the experimental data sweeps. It is also possible to
run a model simulation using experimental data contained in the NWB file associated with the model by
calling
```Python
Model.run_nwb_pulse(sweep_index)
```
This generally takes much longer to execute than the simulated pulse, due to the high sampling frequency
of the experimental NWB data and the fact that the Allen SDK Utils class runs a 12 second simulation
window to account for potentially long NWB recordings.

# Data Visualization
Similar to the Model class, the Visualization class in the Visualization.py module is designed to provide a simple interface to perform visualization functions. A Visualization object should be instantiated with a data array and an index array, as in the demo.py example:
```Python 
currVis = Visualization(data=vertexData, indices=indexData, ...)
```
The current implementation requires the data array to have a very specific format, because data must be passed to PyOpenGL in well-defined blocks of memory. Each section that will be drawn has both data representing the position of the vertex for that section, and the color in which to draw that vertex. The data array must conform to the following specifications:
* A 1-D numpy array with elements of type 32 bit float (dtype=’float32’)
* Each vertex has 7 corresponding pieces of data in the data array– the X, Y, and Z positions in space, and R, G, B, and gamma values describing a color
* To display correctly, the X, Y, and Z positions in space should lie in the range [-1.0,1.0]
* The R, G, B, and gamma values must lie in the range [0.0, 1.0]
* The data array must be formatted with all the position values coming before all the color values.
* This means that for vertex  i,  the (X, Y, Z) values are found beginning at array index (3 * i) and the color values are found beginning (3 * N) + (4 * i)

The index array can be of arbitrary length, but it must be a numpy array with elements of 16 bit integer type, and all values must fall in the range [0, len(data_array) ). The index array specifies pairs of vertices in the data array, and is be used by OpenGL to draw a line between each vertex pair. To retrieve morphological data from the Allen database, the Model class contains a method get_reconstruction() , which will generate a Morphology object (defined in allensdk.core.swc.py) for the .swc file associated with the biophysical model. The code in the  run_visualization(...) function in demo.py gives an example of converting such a Morphology object into arrays that can be used with a Visualization object.

When Visualization.run() is called from a Visualization instance, an interactive OpenGL window containing the morphological reconstruction of the neuron is displayed (Fig 6). Dragging the cursor while holding the left mouse button will translate the visualization in the view window, whereas dragging the cursor while holding the right mouse button will rotate the neuron about the vertical axis. The center mouse button can be used to zoom in and out, and the space bar will trigger a slow rotation of the visualization.

# Acknowledgements
Thank you to the members of the Allen Institute for Brain Science for their work developing the Allen SDK, and to Jason McKesson for his OpenGL tutorial.
