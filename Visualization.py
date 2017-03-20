########################################
# Mario Rosasco, 2017
########################################
# 
# GLSL shader programs and general OpenGL workflow described in tutorials 
# by Jason L. McKesson and adapted for use here under the MIT license. 
# (https://alfonse.bitbucket.io/oldtut/index.html)
#
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import os, sys
from math import tan, atan, cos, acos, sin, sqrt

def calc_frustum_scale(fov_deg):
    '''
    Helper function used to set the frustum scale for OpenGL.
        
    Parameters
    ----------
    fov_deg: float
        angle specifying the width of the field of view in degrees
        
    Returns
    -------
    float
        the appropriate frustum scale for the specified FOV width
    '''
    deg_to_rad = 3.14159 * 2.0 / 360.0
    fov_rad = fov_deg * deg_to_rad
    return 1.0 / tan(fov_rad / 2.0)

class Visualization(object):
    '''
    Class used to set up and use PyOpenGL to display a 3D model of a neuron.
    
    Parameters
    ----------
    data: numpy array of dtype='float32'
        coordinates for the vertices (segment locations) of a neuron
        
    indices: numpy array of dtype='uint16'
        the order in which OpenGL is to read the vertices in data
        
    vertexDim: int
        the dimension of each vertex point. eg: [x,y,z] coordinates have dim=3.
        
    colorDim: int
        the dimension of each color point. eg [r,g,b,gamma] values have dim=4.
        
    nVert: int
        the number of vertices (and color values) in data. The length of data is 
        therefore nVert*(vertexDim + colorDim).
        
    width, height: int
        the width and height, respectively, of the OpenGL display window in pixels 
    '''

    def __init__(self, data=None, indices=None, vertexDim=3, colorDim=4, nVert=0, width=500, height=500):
        self.vertex_data = data
        self.indices = indices
        self.v_dim = vertexDim
        self.c_dim = colorDim
        self.n_vert = nVert
        ###################################
        self.vertex_buffer = None
        self.index_buffer = None
        self.vao = None
        ####################################
        self.model_to_cam_unif = None
        self.cam_to_clip_unif = None
        ####################################
        self.frustum_scale = calc_frustum_scale(45.0)
        self.cam_to_clip_matrix = np.zeros((4,4), dtype='float32')
        self.transform_matrix = np.identity(4, dtype='float32')
        self.auto_rotate = False
        self.loop_colors = False
        self.loop_color_vals = None
        self.loop_color_length = 0
        self.n_loop_points = 0
        self.loop_rate = 0
        self.loop_offset = 0
        ####################################
        self.program = None
        ####################################
        self.left_btn_down = False
        self.rt_btn_down = False
        self.cursor_x = 0
        self.cursor_y = 0
        ####################################
        self.window_width = width
        self.window_height = height
        
        
    def load_shader(self, shader_type, shader_file):
        '''
        Creates and compiles shaders according to the given GLSL type
        
        Parameters
        ----------
        shader_type
            a GL enum value indicating vertex, geometry, or fragment shader type
            
        shader_file
            a filename for a GLSL program specifying a shader of the indicated type
            
        Returns
        -------
        shader
            compiled OpenGL shader object
        '''
        
        # check if file exists, read shader data
        localdir = os.getcwd()
        shader_file = os.path.join(localdir, shader_file)
        if not os.path.isfile(shader_file):
            raise IOError('Could not find target file ' + shader_file)
        shaderData = None
        with open(shader_file, 'r') as f:
            shaderData = f.read()
        
        shader = glCreateShader(shader_type)
        glShaderSource(shader, shaderData) 
        
        glCompileShader(shader)
        
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status == GL_FALSE:
            info = glGetShaderInforLog(shader)
            type_string = ""
            if shader_type is GL_VERTEX_SHADER:
                type_string = "vertex"
            elif shader_type is GL_GEOMETRY_SHADER:
                type_string = "geometry"
            elif shader_type is GL_FRAGMENT_SHADER:
                type_string = "fragment"
            
            print "Compilation failure for " + type_string + " shader:\n" + info
        
        return shader
        
    def create_program(self, shader_list):
        '''
        Compiles a list of shaders to produce a full OpenGL program
        
        Parameters
        ----------
        shader_list
            a list of filenames containing GLSL shader programs
            
        Returns
        -------
        program
            the compiled OpenGL program
        '''
        program = glCreateProgram()
    
        for shader in shader_list:
            glAttachShader(program, shader)
            
        glLinkProgram(program)
        
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if status == GL_FALSE:
            info = glGetProgramInfoLog(program)
            print "Linker failure: \n" + info
            
        for shader in shader_list:
            glDetachShader(program, shader)
            
        return program
        
    def init_program(self):
        '''
        Sets up the list of shaders, and compiles them into a usable program  
        '''
        shaderList = []
    
        shaderList.append(self.load_shader(GL_VERTEX_SHADER, "170126-VertexShader.vert"))
        shaderList.append(self.load_shader(GL_FRAGMENT_SHADER, "170126-FragmentShader.frag"))
        
        self.program = self.create_program(shaderList)
        
        for shader in shaderList:
            glDeleteShader(shader)
        
        self.model_to_cam_unif = glGetUniformLocation(self.program, "modelToCameraMatrix")
        self.cam_to_clip_unif = glGetUniformLocation(self.program, "cameraToClipMatrix")
        
        fzNear = 1.0
        fzFar = 61.0
        
        # Note that this and the transformation matrix below are both
        # ROW-MAJOR ordered. Thus, it is necessary to pass a transpose
        # of the matrix to the glUniform assignment function.
        self.cam_to_clip_matrix[0][0] = self.frustum_scale
        self.cam_to_clip_matrix[1][1] = self.frustum_scale
        self.cam_to_clip_matrix[2][2] = (fzFar + fzNear) / (fzNear - fzFar)
        self.cam_to_clip_matrix[2][3] = -1.0
        self.cam_to_clip_matrix[3][2] = (2 * fzFar * fzNear) / (fzNear - fzFar)
        
        glUseProgram(self.program)
        glUniformMatrix4fv(self.cam_to_clip_unif, 1, GL_FALSE, self.cam_to_clip_matrix.transpose())
        glUseProgram(0)
        
    def init_vertex_buffer(self):
        ''' 
        Sets up the buffer to store vertex coordinates for OpenGL's access
        '''
        
        self.vertex_buffer = glGenBuffers(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData( # PyOpenGL allows for the omission of the size parameter
            GL_ARRAY_BUFFER,
            self.vertex_data,
            GL_STATIC_DRAW
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.index_buffer = glGenBuffers(1)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.indices,
            GL_STATIC_DRAW
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        
    def translate(self, d, axis=0):
        '''
        Updates the transformation matrix to render an object translated along the indicated axis.
        
        Parameters
        ----------
        d: float
            the value by which to translate the object (in screen space)
            
        axis: int
            0 = x axis
            1 = y axis
            2 = z axis
        '''
        curr = self.transform_matrix[axis][3]
        self.transform_matrix[axis][3] = curr + d
        
    def setup_visualization(self):
        '''
        Sets up the OpenGL environment
        '''
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL;
        glutInitDisplayMode (displayMode)
        
        glutInitContextVersion(3,3)
        glutInitContextProfile(GLUT_CORE_PROFILE)

        glutInitWindowSize (self.window_width, self.window_height)
        
        glutInitWindowPosition (300, 200)
        
        window = glutCreateWindow("Neuron Visualization")
            
        self.init_program()
        self.init_vertex_buffer()
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        glLineWidth(2)
        
        sizeof_float = 4 # all arrays should be dtype='float32'
        col_offset = c_void_p(self.v_dim * self.n_vert * sizeof_float)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, self.v_dim, GL_FLOAT, GL_FALSE, 0, None)
        glVertexAttribPointer(1, self.c_dim, GL_FLOAT, GL_FALSE, 0, col_offset)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        
        glBindVertexArray(0)
        
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CW)
        
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LEQUAL)
        glDepthRange(0.0, 1.0)
        
        # bind handlers for interaction and animation
        glutDisplayFunc(self.__display) 
        glutReshapeFunc(self.__reshape)
        glutKeyboardFunc(self.__keyboard)
        glutMouseFunc(self.__mouse)
        glutMotionFunc(self.__drag)
        
        # Move the neuron into the center of the viewspace (-Z direction)
        self.translate(-2,2)
        
    def change_colors(self, new_colors, offset=0):
        '''
        Updates the section of the vertex buffer object that contains the color data,
        then re-buffers the updated data into OpenGL
        
        Parameters
        ----------
        new_colors: numpy array of dtype=float32
            represents vertex colors in (R,G,B,Gamma) format
            
        offset: int
            the index of the vertex where the updated colors apply
            eg: offset = 2 starts updating colors at the third vertex.
        '''
        
        col_offset = self.v_dim * self.n_vert + offset * self.c_dim
        
        try:
            self.vertex_data[col_offset:col_offset + len(new_colors)] = new_colors
        except ValueError:
            print "Update color array did not fit in the target buffer array. Could not update colors."
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
    def change_color_loop(self, new_colors, n_colors, n_timepoints, offset=0, rate=1):
        '''
        Sets up the time-dependent change of the model's coloration.
        
        Parameters
        ----------
        new_colors: numpy array of dtype=float32
            represents vertex colors in (R,G,B,Gamma) format. Must therefore be of length n_timepoints * n_colors * 4.
        
        n_colors: int
            represents the number of colors (and thus the number of vertices/segments) that will be recolored at each timepoint.
            
        n_timepoints: int
            represents the number of timepoints in the recoloration loop. 
            If n_timepoints = 1, this function will have identical behavior to change_colors().
            
        offset: int
            the index of the vertex where the updated colors apply
            eg: offset = 2 starts updating colors at the third vertex.
            
        rate: float
            A multiplier that is applied to the OpenGL clock before calculating the color to apply. 
            Values <1.0 therefore decrease the loop speed, while values >1.0 increase the loop speed.
        '''
        self.loop_colors = True
        self.loop_color_vals = new_colors
        self.loop_color_length = n_colors
        self.loop_rate = rate
        self.loop_offset = offset
        self.n_loop_points = n_timepoints
    
    def run(self):
        '''
        Starts the main OpenGL loop using GLUT
        '''
        
        print '''
        Starting up visualization.
        Controls:
            Hold left mouse button - drag model
            Hold right mouse button - rotate model
            Mouse scroll wheel - zoom in and out
            Space bar - toggle constant rotation
            Esc - quit
        '''
        self.setup_visualization()
        glutMainLoop();
        
    def __display(self):
        '''
        Called to update the display.
        '''
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.program)
        
        glBindVertexArray(self.vao)
            
        glUniformMatrix4fv(self.model_to_cam_unif, 1, GL_FALSE, self.transform_matrix.transpose())
        glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_SHORT, None)
        
        if self.loop_colors:
            time = glutGet(GLUT_ELAPSED_TIME)
            adjusted_time = int(time * self.loop_rate)
            
            c_index = (adjusted_time)*(self.loop_color_length * self.c_dim)
            c_index = c_index % len(self.loop_color_vals)
            
            print (float(c_index) / len(self.loop_color_vals)) * 60.0, "ms"
            
            new_col = self.loop_color_vals[c_index : c_index + (self.loop_color_length * self.c_dim)]
            self.change_colors(new_col, self.loop_offset)
            
        if self.auto_rotate:
            time = glutGet(GLUT_ELAPSED_TIME)
            self.__auto_rotate_y(time)
        
        glBindVertexArray(0)
        glUseProgram(0)
        
        glutSwapBuffers()
        glutPostRedisplay()
        
    def __reshape(self, w, h):
        '''
        Called whenever the window's size is changed (including once at program start)
        
        Parameters
        ----------
        w, h: int
            width and height, respectively, of screen in pixels
        '''
        
        self.cam_to_clip_matrix
        self.cam_to_clip_matrix[0][0] = self.frustum_scale * (h / float(w))
        self.cam_to_clip_matrix[1][1] = self.frustum_scale

        glUseProgram(self.program)
        glUniformMatrix4fv(self.cam_to_clip_unif, 1, GL_FALSE, self.cam_to_clip_matrix.transpose())
        glUseProgram(0)
        
        line_width = w/500
        if line_width < 1: line_width = 1
        
        glViewport(0, 0, w, h)
        
    def __keyboard(self, keyval, x, y):
        '''
        Handler for keyboard interaction
        
        Parameters
        ----------
        keyval: char
            The character of the key that was pressed
            
        x,y: int
            the position of the mouse when the key was pressed
        '''
        # ord() gets the keycode for the letter that's passed as an argument
        keycode = ord(keyval)
        if keycode == 27: # escape; exit program.
            glutLeaveMainLoop()
            return
        elif keycode == 32: # spacebar - toggle auto rotation on/off
            self.auto_rotate = not(self.auto_rotate)
            
    def __mouse(self, button, status, x, y):
        '''
        Handler for mouse interaction.
        
        Parameters
        ----------
        button: int
            a code indicating which button was pressed. On my machine:
                0 = left
                1 = center
                2 = right
                3 = scroll forward
                4 = scroll back
        status: int
            indicates whether the button is currently pressed (0) or not (1)
        x, y: int
            x and y position (in pixels) of the cursor when the event was triggered
        '''
        if button == 0: # left click
            self.left_btn_down = (status == 0)
            if self.left_btn_down:
                self.cursor_x = x
                self.cursor_y = y
        if button == 2: # rt click
            self.rt_btn_down = (status == 0)
            return
        if button == 1: # center click
            # do other stuff
            return
        if button == 3 and status == 0: # forward scroll
            self.translate(-0.1, 2)
        if button == 4 and status == 0: # reverse scroll
            self.translate(0.1, 2)
            
    def __drag(self, x, y):
        '''
        Handler for mouse interaction while button is pressed and mouse is moved simultaneously
        
        Parameters
        ----------
        x, y: ints
            x and y position (in pixels) of the cursor when the event was triggered
        '''
        if self.left_btn_down: # left button translates in x-y plane
            dx = -(self.cursor_x - float(x))*2.0 / glutGet(GLUT_WINDOW_WIDTH)
            dy = (self.cursor_y - float(y))*2.0 / glutGet(GLUT_WINDOW_HEIGHT)
            self.translate(dx, 0)
            self.translate(dy, 1)
            self.cursor_x = x
            self.cursor_y = y
        if self.rt_btn_down: # right button rotates in x-z plane
            if not(self.auto_rotate):
                dx = -(self.cursor_x - float(x))*2.0 / glutGet(GLUT_WINDOW_WIDTH)
                self.__rotate_y(dx)
                self.cursor_x = x
    
    def __auto_rotate_y(self, t):
        '''
        Causes the model to rotate about the y axis based on time
        
        Parameters
        ----------
        t: numeric
            current time in milliseconds
        '''
        loop_length=10000.0 # milliseconds
        scale = 3.141592 * 2.0 / loop_length
        time_in_loop = t % loop_length
        ang_rad = time_in_loop * scale
        
        cos_ang = cos(ang_rad)
        sin_ang = sin(ang_rad)
        
        self.transform_matrix[0][0] = cos_ang
        self.transform_matrix[2][0] = sin_ang
        self.transform_matrix[0][2] = -sin_ang
        self.transform_matrix[2][2] = cos_ang
    
    def __rotate_y(self, dx):
        '''
        Sets the transform matrix to rotate the object about the y axis
        
        Parameters
        ----------
        dx: int
            Indicates the magnitude of the movement in the x-axis, which defines the magnitude of the rotation.
        '''
        curr_ang_rad = acos(self.transform_matrix[0][0])
        ang_rad = (dx*(2*3.141592)) + curr_ang_rad
        cos_ang = cos(ang_rad)
        sin_ang = sin(ang_rad)
        
        self.transform_matrix[0][0] = cos_ang
        self.transform_matrix[2][0] = sin_ang
        self.transform_matrix[0][2] = -sin_ang
        self.transform_matrix[2][2] = cos_ang