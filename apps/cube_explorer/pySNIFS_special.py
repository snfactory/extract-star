from matplotlib.mlab import linspace, dist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.numerix import array
from matplotlib.transforms import blend_xy_sep_transform
from scipy.special import sqrt

import thread
import pylab


class Cursor:
    """
    A horizontal and vertical line span the axes that and move with
    the pointer.  You can turn off the hline or vline spectively with
    the attributes

      horizOn =True|False: controls visibility of the horizontal line
      vertOn =True|False: controls visibility of the horizontal line      

    And the visibility of the cursor itself with visible attribute
    """
    def __init__(self, ax, useblit=False, **lineprops):
        """
        Add a cursor to ax.  If useblit=True, use the backend
        dependent blitting features for faster updates (GTKAgg only
        now).  lineprops is a dictionary of line properties.  See
        examples/widgets/cursor.py.
        """
        self.ax = ax
        self.canvas = ax.figure.canvas

        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('draw_event', self.clear)

        self.visible = True
        self.horizOn = True
        self.vertOn = True
        self.useblit = useblit

        self.lineh = ax.axhline(0, visible=False, **lineprops)
        self.linev = ax.axvline(0, visible=False, **lineprops)

        self.background = None
        self.needclear = False


        
    def clear(self, event):
        'clear the cursor'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)        

    def onmove(self, event):
        'on mouse motion draw the cursor if visible'
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)        

            if self.needclear:            
                self.canvas.draw()
                self.needclear = False
            return 
        self.needclear = True
        if not self.visible: return 
        self.linev.set_xdata((event.xdata, event.xdata))

        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)        
	
	self._update()
        

    def _update(self):
        
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)            
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False

class HorizontalSpanSelector:
    """
    Select a min/max range of the x axes for a matplotlib Axes

    Example usage:

      ax = subplot(111)
      ax.plot(x,y)

      def onselect(xmin, xmax):
      print xmin, xmax
      span = HorizontalSpanSelector(ax, onselect)

    """
    def __init__(self, ax, onselect, minspan=None, useblit=False, rectprops=None):
        """
        Create a span selector in ax.  When a selection is made, clear
        the span and call onselect with

          onselect(xmin, xmax)

        and clear the span.

        If minspan is not None, ignore events smaller than minspan

        The span rect is drawn with rectprops; default
          rectprops = dict(facecolor='red', alpha=0.5)

        set the visible attribute to False if you want to turn off
        the functionality of the span selector


        """
        if rectprops is None:
            rectprops = dict(facecolor='red', alpha=0.5)        
            
        self.ax = ax
        self.visible = True
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('draw_event', self.update_background)

        self.rect = None
        self.background = None

        self.rectprops = rectprops
        self.onselect = onselect
        self.useblit = useblit
        self.minspan = minspan

        trans = blend_xy_sep_transform(self.ax.transData, self.ax.transAxes)

        self.rect = Rectangle( (0,0), 0, 1,
                               transform=trans,
                               visible=False,
                               **self.rectprops                               
                               )
        
        if not self.useblit: self.ax.add_patch(self.rect)
        self.pressx = None
        
    def update_background(self, event):
        'force an update of the background'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        
    def ignore(self, event):
        'return True if event should be ignored'
        return  event.inaxes!=self.ax or not self.visible or event.button !=1 


    def press(self, event):
        'on button press event'
        if self.ignore(event): return
        
        self.rect.set_visible(self.visible)
        self.pressx = event.xdata
      
        return False


    def release(self, event):
        'on button release event'
        if self.pressx is None or self.ignore(event): return

        self.rect.set_visible(False)
        self.canvas.draw()
        xmin = self.pressx
        xmax = event.xdata
	y = event.ydata
        if xmin>xmax: xmin, xmax = xmax, xmin
        span = xmax - xmin
        if self.minspan is not None and span<self.minspan: return
        self.onselect(xmin, xmax, y)
        self.pressx = None
        return False

    def update(self):
        'draw using newfangled blit or oldfangled draw depending on useblit'
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.rect)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

        return False

    def onmove(self, event):
        'on motion notify event'
        if self.pressx is None or self.ignore(event): return
        x,y = event.xdata, event.ydata

        minx, maxx = x, self.pressx
        if minx>maxx: minx, maxx = maxx, minx
        self.rect.xy[0] = minx
        self.rect.set_width(maxx-minx)            
        self.update()
        return False


class CircleSpanSelector:
    """
    Select a center/radius range of the circle for a matplotlib Axes

    Example usage:

      ax = subplot(111)
      ax.plot(x,y)

      def onselect(center, radius, x, y):
          print center, radius
      span = CircleSpanSelector(ax, onselect)
    """
   
    def __init__(self, ax, onselect, minspan=None, useblit=False, circprops=None):
	""" 

        Create a span selector in ax.  When a selection is made, clear
        the span and call onselect with

          onselect(center, radius, x, y) 
	
	where x and y are the coordinate used to calculate the radius
        and clear the span.

        If minspan is not None, ignore events smaller than minspan

        The span rect is drawn with rectprops; default
          circprops = dict(fc='blue', alpha=0.5)

        set the visible attribute to False if you want to turn off
        the functionality of the span selector


        """
        if circprops is None:
            circprops = dict(fc='w', alpha=0.5)        
            
        self.ax = ax
        self.visible = True
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('draw_event', self.update_background)

        self.circ = None
        self.background = None

        self.circprops = circprops
        self.onselect = onselect
        self.useblit = useblit
        self.minspan = minspan

        self.circ = Circle( (0,0), 1, **self.circprops)
        
	self.unit_verts = [v for v in self.circ.verts]
	self.circ.set_visible(False)

        if not self.useblit: self.ax.add_patch(self.circ)
        self.pressx = None
        
    def update_background(self, event):
        'force an update of the background'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        
    def ignore(self, event):
        'return True if event should be ignored'
        return  event.inaxes!=self.ax or not self.visible or event.button !=1 


    def press(self, event):
        'on button press event'
        if self.ignore(event): return
        
	print "press"
        self.circ.set_visible(self.visible)
        self.pressx = event.xdata
        self.pressy = event.ydata
	self.circ.set_visible(False)

        return False


    def release(self, event):
        'on button release event'
        if self.pressx is None or self.ignore(event): return
        if self.pressy is None or self.ignore(event): return

        self.canvas.draw()
	self.center = [self.pressx, self.pressy]
	self.radius = sqrt((event.xdata-self.center[0])**2 + (event.ydata-self.center[1])**2)
	y = event.ydata
	x = event.xdata
        if self.minspan is not None and radius<self.minspan: return
        self.onselect(self.center, self.radius, x, y)
        self.pressx = None
	self.pressy = None
        return False

    def update(self):
        'draw using newfangled blit or oldfangled draw depending on useblit'
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.circ)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

        return False

    def onmove(self, event):
        'on motion notify event'
        if self.pressx is None or self.ignore(event): return
        if self.pressy is None or self.ignore(event): return

        self.center = [self.pressx,self.pressy]
	self.radius = sqrt((event.xdata-self.center[0])**2 + (event.ydata-self.center[1])**2)
	if self.radius > 0.5:
	    self.circ.set_visible(True)
	else:
	    self.circ.set_visible(False)
        self.circ.verts = [(v[0]*self.radius+self.center[0],v[1]*self.radius+self.center[1]) for v in self.unit_verts]
	pylab.draw()

        self.update()
        return False
