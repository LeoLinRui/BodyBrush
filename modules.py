import numpy as np
from multiprocessing import Process, Queue
import cv2
import math
import time


def alphaBlend(top, bottom):
    '''
    takes an HSLA top and a bottom, blends and returns a HSL image
    '''
    assert top.shape[0] == 4, "top must have alpha channel"
    assert top.shape[1] == bottom.shape[1], "top and bottom must have equal shape"
    assert top.shape[2] == bottom.shape[2], "top and bottom must have equal shape"

    a = top[3] / 255
    b = 1 - a

    return np.array([top[0]*a + bottom[0]*b, top[1]*a + bottom[1]*b, top[2]*a + bottom[2]*b])


def cleanUp(self):
    '''
    keep the path:list at a set length by:
        - delete coords in path that's from too long ago
        - keep a list of fully rendered layers from the delete point to the most recent render
        - update the object variable baseBG to the state at the delete point
    '''

    # trim list
    if len(self.raw_point_list) > self.path_limit:
        self.raw_point_list = self.raw_point_list[-(self.path_limit):]

    if len(self.rendered_layer_buffer) > self.path_limit:
        return self.rendered_layer_buffer.pop(0)
    else:
        return self.rendered_layer_buffer[0]


def loadBrushProfile(path):
    '''
    takes a path to an image
    returns a square 2D array 0-255
    '''
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = math.degrees(np.arctan2(y, x)) + 90
    return(rho, phi)


def defaltInp(dic, key, default_value):
    if key in dic.keys():
        if type(dic[key]) != type(default_value):
            print("WARNING: When initializing renderer process, render data of type {} does not match expected type of {}".format(type(dic[key]), type(default_value)))
        return dic[key]
    else:
        return default_value


def getStroke(raw_point_list, Gaussian_coeff=10):
        
    def bresenham_get_line(start, end):

        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
    
        points1 = get_line((0, 0), (3, 4))
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """

        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
    
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points


    path = []

    if len(raw_point_list) >= 2:
        for i, point in enumerate(raw_point_list):
            if i == len(raw_point_list) - 1:
                break # to avoid out of bound or first dot
            else:
                path += bresenham_get_line(point, raw_point_list[i+1])

    #TODO add Gaussian smooth to the list of points
    #TODO temporal decay
    # print(path) # should be a flat 2D list of pixels coord

    return np.array(path)


#===============================================================================================================================
# Path Class
#=============================================================================================================================== 


class Path():
    def __init__(self, map_resolution:list):
        
        # set static coefficients
        self.base_res = map_resolution
        #self.temporal_decay = temporal_decay
        #self.inertia_coeff = inertia_coeff
        #self.base_color = base_color

        # set  variables
        self.color = np.array([0, 0, 0])
        self.inertia = 0.0
        self.stroke_width = 0.0

        # create point lists
        self.raw_point_list = None
        self.point_list = None

        # max size of path:list
        self.path_limit = 20

        # rendered image buffer
        self.rendered_layer_buffer = []

        # renderd flag to enforce rendering after each point addition
        # this avoids issues with buffer 
        # mismatching with raw point list
        self.RENDERED = True
        self.FIRST_DOT = True

        # list for stroke vectoring
        self.stroke_vectors = []

        self.grain_path = r"I:\TRS Project 2\BodyBrush\brush_profile\grain.jpg"
        self.shape_path = r"I:\TRS Project 2\BodyBrush\brush_profile\shape.jpg"

        self.brush_shape = loadBrushProfile(self.shape_path)
        self.brush_grain = loadBrushProfile(self.grain_path)


    def addPoint(self, coord:list):
        # enforce rendering
        assert self.RENDERED == True, "addPoint() called before rendering"

        # get coord 
        point_x = coord[0]
        point_y = coord[1]

        # update list of points
        if self.raw_point_list is None:
            self.FIRST_DOT = True
            self.raw_point_list = np.array([point_x, point_y])
        else:
            self.FIRST_DOT = False
            self.raw_point_list = np.vstack((self.raw_point_list, np.array([point_x, point_y])))

        self.RENDERED = False

    def renderData(self):
        self.RENDERED = True
        return {"raw_point_list":self.raw_point_list, "brush_shape":self.brush_shape, "brush_grain":self.brush_grain, "first_dot":self.FIRST_DOT}


#===============================================================================================================================
# PathManager Class
#=============================================================================================================================== 

    
class PathManager():
    def __init__(self, detection_resolution):
        
        self.res = detection_resolution

        self.master_dict = {} # {pathID:[path object, render process, input q, output q]} 

        self.BG = np.vstack([[np.ones(self.res)*179], [np.ones(self.res)*10], [np.ones(self.res)*100]])

        self.rendering_sequence = range(17) #change layer order here

        if __name__ == '__main__':

            for i in range(17):
                in_q = Queue()
                out_q = Queue()

                process = Process(target=renderer, args=(in_q, out_q))
                process.start()

                self.master_dict[i] = [Path(map_resolution=self.res), process, in_q, out_q]

            print("PathManager initialization complete.")


    def addPoints(self, points:list):
        if __name__ == '__main__':
            #TODO case 1: that path does not yet exist, add a path (path object)
            #case 2: path already exist, add point to that path
            for pt in points:
                self.master_dict[int(pt[0])][0].addPoint([int(pt[1]), int(pt[2])])
            print("points added to path")


    def render(self):
        if __name__ == '__main__':
            print("Master rendering process started")

            # loop through all 17 items to: get render parameters from Path and put into inp q
            for layer in self.master_dict.values():
                render_data = layer[0].renderData()
                layer[2].put(render_data)

            print("render data had been added to input queues")

            rendered_layers = [] # will be in the numerical indexing order 

            # retrieve rendered layers from out q and put into rendered_layers for blending
            for layer in self.master_dict.values():
                rendered_layers.append(layer[3].get())

            print("rendered layers list", len(rendered_layers))

            # perform alpha blending to the layer according to the redering sequence
            bottom_layer = self.BG
            for i in self.rendering_sequence:
                bottom_layer = alphaBlend(rendered_layers[i], bottom_layer)
                
            bottom_layer = np.transpose(bottom_layer, [2, 1, 0])
            return bottom_layer


#===============================================================================================================================
# renderer process
#=============================================================================================================================== 

def renderer(inp_queue, out_queue):

    '''
    The render method takes in path (a list of xy coordinates that has already been smoothed) and
    returns stroke_layer (an HSLA 3D array)

    input: 
        path(list of coords), 
        BG(all layers below that was rendered), 
        brush shape (n by n grayscale image), 
        brush grain (m by m gray scale image),
        base color (H, S)

    1. render the alpha layer(step_size, brush size, jitter): walk brush shape on the mask. 
    2. render grascale grain on to the L layer(step_size, grain size, jitter): do the same walk.
    3. render the H and S layer(): fill two layers with the same H and S values
        - have the option to render using the weighted average of base HS and BG HS instead of base HS
    4. blend the rendered HSLA layer with the baseBG layer
    '''

    print("Renderer process started")

    # calculate other parameters from path

    def calculateSpeed(raw_points, path):
        #TODO calculate spped
        return np.ones(len(path))

    def calculateAcceleration(raw_points, path):
        #TODO calculate acceleration
        return np.ones(len(path))


    while True:
        # get input from input queue
        render_data = inp_queue.get()
        job_id = render_data[0]
        render_data = render_data[1]
        
        # parse render_data input and create all hyperparameters
        raw_point_list = render_data["raw_point_list"]

        RENDER_SCALE = defaltInp(render_data, "RENDER_SCALE", 1)
        BASE_RES = defaltInp(render_data, "BASE_RES", [1000, 1000])
        SHAPE_STEP_SIZE = int(defaltInp(render_data, "SHAPE_STEP_SIZE", 1) * RENDER_SCALE)
        GRAIN_STEP_SIZE = int(defaltInp(render_data, "GRAIN_STEP_SIZE", 1) * RENDER_SCALE)
        SHAPE_SIZE = int(defaltInp(render_data, "SHAPE_SIZE", 20) * RENDER_SCALE)
        GRAIN_SIZE_FACTOR = defaltInp(render_data, "GRAIN_SIZE_FACTOR", 0.5) * RENDER_SCALE
        SHAPE_JITTER_FACTOR = defaltInp(render_data, "SHAPE_JITTER_FACTOR", 0.2) * RENDER_SCALE
        GRAIN_JITTER_FACTOR = defaltInp(render_data, "GRAIN_JITTER_FACTOR", 0.2) * RENDER_SCALE
        BASE_COLOR = defaltInp(render_data, "BASE_COLOR", [50, 50])
        VECTOR_LEN = 200
        TEMPORAL_DECAY = 10000

        # flags
        FIRST_DOT = render_data["first_dot"]

        #========================================================================================
        t_start = time.time()
        # caculate render resolution
        render_res = np.array(RENDER_SCALE * np.array(BASE_RES), dtype=int)

        # load brush profile
        brush_shape = render_data["brush_shape"]
        brush_grain_original = render_data["brush_grain"]
        
        
        #========================================================================================

        # create the 4 channels
        a_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        h_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        s_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        l_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)

        #========================================================================================
        # check for first dot
        if FIRST_DOT:
            # happens when no points has been added but render is always called
            # retuern a transparent layer
            out_queue.put([job_id, np.vstack([[h_channel], [s_channel], [l_channel], [a_channel]])])
            
        else:
            #get stroke path from raw points
            path = getStroke(raw_point_list)

            # path needs to be scaled
            path = path * RENDER_SCALE
                
            #========================================================================================
            # 1. render the alpha layer(step_size, brush size, jitter): walk brush shape on the mask.

            stepped_path = path[-(TEMPORAL_DECAY * SHAPE_STEP_SIZE)::SHAPE_STEP_SIZE]

            t_start = time.time()
            for i, pt in enumerate(stepped_path):

                # prepare brush shape
                brush_shape = cv2.resize(brush_shape, (SHAPE_SIZE, SHAPE_SIZE))

                # jitter
                jitter = np.random.randint(0, int(SHAPE_SIZE * SHAPE_JITTER_FACTOR), size=2)
                x, y = pt + jitter
                x, y = int(x), int(y)
                
                # draw the shape
                dx , dy = int(x + SHAPE_SIZE), int(y + SHAPE_SIZE)
                        
                # handle edge case
                if dx < render_res[1] and dy < render_res[0]:

                    #rotate the stamp
                    brush_shape_rotated = brush_shape

                    if i > VECTOR_LEN:
                        
                        v_origin = stepped_path[- (i - int(RENDER_SCALE * VECTOR_LEN)):][0]
                        v = pt - v_origin # find catesian vector between the two points
                        v = cart2pol(v[0], v[1])
                        brush_shape_rotated = rotateImage(brush_shape, v[1])

                    # "stamp" the stroke
                    # use addition here to avoid 0 overwriting 255, then cap it at 255

                    try:
                        a_channel[x:dx, y:dy] = a_channel[x:dx, y:dy] + brush_shape_rotated
                    except:
                        print("WARNING: alpha channel stamping range error, stamp ignored at", dx, dy)
                        pass
                    finally:
                        a_channel[x:dx, y:dy] = np.clip(a_channel[x:dx, y:dy], 0, 255)


            l_channel = cv2.resize(brush_grain_original, (render_res[0], render_res[1]))

            l_channel = cv2.xphoto.oilPainting(l_channel, 7, 1)


            #========================================================================================
            # 3. render the H and S layer(): fill two layers with the same H and S values

            h_channel = np.full([render_res[0], render_res[1]], BASE_COLOR[0]) 
            s_channel = np.full([render_res[0], render_res[1]], BASE_COLOR[1])

            #========================================================================================
            # 4. blend the rendered HSLA layer with the baseBG layer
            
            stroke_layer = np.vstack([[h_channel], [s_channel], [l_channel], [a_channel]])


            #========================================================================================
            # misc.

            #self.rendered_layer_buffer.append(stroke_layer)
            #cleanUp()
            RENDERED = True    
            out_queue.put([job_id, stroke_layer])
            print("time cost of shape stamp traverse of length {} is {:.4f} ".format(len(stepped_path),  time.time() - t_start))