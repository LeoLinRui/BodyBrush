import numpy as np
from multiprocessing import Process, Queue
import cv2
from utils import alphaBlend


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
        # this avoids issues with buffer mismatching with raw point list
        self.RENDERED = True

        # list for stroke vectoring
        self.stroke_vectors = []


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
        return {"raw_point_list":self.raw_point_list}


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

    from utils import alphaBlend, cleanUp, loadBrushProfile, rotateImage, cart2pol, defaltInp, getStroke
    print("Renderer process started")


    while True:
        # get input from input queue
        render_data = inp_queue.get()
        print("render data successfully recieved by the rendering process")
        
        # parse render_data input and create all hyperparameters
        raw_point_list = render_data["raw_point_list"]

        RENDER_SCALE = defaltInp(render_data, "RENDER_SCALE", 1)
        BASE_RES = defaltInp(render_data, "BASE_RES", 1)
        SHAPE_STEP_SIZE = int(defaltInp(render_data, "SHAPE_STEP_SIZE", 1) * RENDER_SCALE)
        GRAIN_STEP_SIZE = int(defaltInp(render_data, "GRAIN_STEP_SIZE", 1) * RENDER_SCALE)
        SHAPE_SIZE = int(defaltInp(render_data, "SHAPE_SIZE", 50) * RENDER_SCALE)
        GRAIN_SIZE_FACTOR = defaltInp(render_data, "GRAIN_SIZE_FACTOR", 0.5) * RENDER_SCALE
        SHAPE_JITTER_FACTOR = defaltInp(render_data, "SHAPE_JITTER_FACTOR", 0.2) * RENDER_SCALE
        GRAIN_JITTER_FACTOR = defaltInp(render_data, "GRAIN_JITTER_FACTOR", 0.2) * RENDER_SCALE
        BASE_COLOR = defaltInp(render_data, "BASE_COLOR", [50, 50])
        VECTOR_LEN = 200

        #========================================================================================

        # caculate render resolution
        render_res = np.array(RENDER_SCALE * np.array(BASE_RES), dtype=int)

        # load brush profile
        brush_shape = loadBrushProfile(render_data["brush_shape"])
        brush_grain_original = loadBrushProfile(render_data["brush_grain"])

        #========================================================================================

        # create the 4 channels
        a_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        h_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        s_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)
        l_channel = np.zeros([render_res[0], render_res[1]], dtype=np.uint16)

        #========================================================================================
        # check if point list is empty
        if raw_point_list is None:
            # happens when no points has been added but render is always called
            # retuern a transparent layer
            return np.vstack([[h_channel], [s_channel], [l_channel], [a_channel]])
        
        #get stroke path from raw points
        path = getStroke(raw_point_list)

        # path needs to be scaled
        path = path * RENDER_SCALE

        #========================================================================================
        # calculate other parameters from path

        def calculateSpeed(raw_points, path):
            #TODO calculate spped
            return np.ones(len(path))

        def calculateAcceleration(raw_points, path):
            #TODO calculate acceleration
            return np.ones(len(path))
            
        #========================================================================================
        # 1. render the alpha layer(step_size, brush size, jitter): walk brush shape on the mask.

        stepped_path = path[::SHAPE_STEP_SIZE]

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
            if dx > render_res[1] - 1:
                brush_shape = brush_shape[0:int(dx - render_res[1] + 1), :]
            if dy > render_res[0] - 1:
                brush_shape = brush_shape[:, 0:int(dx - render_res[0] + 1)]
            
            #rotate the stamp
            brush_shape_rotated = brush_shape
            if i > VECTOR_LEN:
                v_origin = stepped_path[- (i - int(RENDER_SCALE * VECTOR_LEN)):][0]
                v = pt - v_origin # find catesian vector between the two points
                v = cart2pol(v[0], v[1])
                brush_shape_rotated = rotateImage(brush_shape, v[1])

            # "stamp" the stroke
            # use addition here to avoid 0 overwriting 255, then cap it at 255
            a_channel[x:dx, y:dy] = a_channel[x:dx, y:dy] + brush_shape_rotated
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
        out_queue.put(stroke_layer)