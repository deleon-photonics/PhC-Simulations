import gdspy
import random
import string
import numpy as np

class taper:
    def __init__(self, **kwargs):
        self.start_width = 0.5
        self.end_width = 5
        self.length = 10

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class grating:
    def __init__(self, **kwargs):
        self.period = 1
        self.duty_cycle = 0.5
        self.width = 5
        self.num_gratings = 1

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

def make_taper(taper, layer):
    taper_cell = lib.new_cell('taper' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    if use_positive_tone == 1:
        vertices = [(0, -taper.start_width/2), (0, taper.start_width/2),
                    (taper.length, taper.end_width/2), 
                    (taper.length, taper_support_width/2), (taper.length + taper_support_length, taper_support_width/2), 
                    (taper.length + taper_support_length, -taper_support_width/2), (taper.length, -taper_support_width/2),
                    (taper.length, -taper.end_width/2)]
        border_vertices = [(0, -taper.start_width/2 - waveguide_border_spacing),
                           (taper.length, -taper.end_width/2 - waveguide_border_spacing),
                           (taper.length, -taper.end_width/2 - waveguide_border_spacing - waveguide_border_width),
                           (0, -taper.start_width/2 - waveguide_border_spacing - waveguide_border_width)]
        border = gdspy.Polygon(border_vertices)
        taper_cell.add(border)
    else:
        vertices = [(0, -taper.start_width/2), (0, taper.start_width/2),
                    (taper.length, taper.end_width/2), 
                    (taper.length, -taper.end_width/2)]
    taper = gdspy.Polygon(points=vertices, layer=layer)
    taper_cell.add(taper)
    
    return taper_cell

def make_grating(grating, layer):
    grating_cell = lib.new_cell('grating' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    x = grating.period*(1-grating.duty_cycle) + taper_support_length
    for i in range(grating.num_gratings):
        rect = gdspy.Rectangle((x, -grating.width/2), (x + grating.period*grating.duty_cycle, grating.width/2), layer=layer)
        x = x + grating.period
        grating_cell.add(rect)

    if use_positive_tone == 1:
        top_rec = gdspy.Rectangle((0, grating.width/2), (x, grating.width/2 + grating_edge_support))
        bot_rec = gdspy.Rectangle((0, -grating.width/2), (x, -grating.width/2 - grating_edge_support))
        x = x-grating.period
        back_support = gdspy.Rectangle((x, -grating.width/2), (x + 10, grating.width/2))
        grating_cell.add([back_support, top_rec, bot_rec])
    
    return grating_cell

def make_coupler_waveguide(grating, taper, coupler_width, coupler_length, layer):
    coupler_cell = lib.new_cell('coupler' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))

    #Coupler waveguide
    coupler_wg = gdspy.Rectangle((-coupler_length/2, -coupler_width/2), (coupler_length/2, coupler_width/2), layer=layer)

    #Taper
    taper_cell = make_taper(taper, layer)
    left_taper = gdspy.CellReference(taper_cell, (-coupler_length/2, 0), 180, x_reflection=1)
    right_taper = gdspy.CellReference(taper_cell, (coupler_length/2, 0))
    
    #Grating_Coupler
    grating_cell = make_grating(grating, layer)
    left_grating = gdspy.CellReference(grating_cell, (-coupler_length/2 - taper.length, 0), 180)
    right_grating = gdspy.CellReference(grating_cell, (coupler_length/2 + taper.length, 0), 0)

    coupler_cell.add([left_grating, left_taper, coupler_wg, right_taper, right_grating])

    if use_positive_tone == 1:
        waveguide_border = gdspy.Rectangle((-coupler_length/2, -coupler_width/2 - waveguide_border_spacing),
                                           (coupler_length/2, -coupler_width/2 - waveguide_border_spacing - waveguide_border_width))
        
        coupler_cell.add(waveguide_border)

    return coupler_cell

def make_microdisk_and_coupler(grating, taper, disk_r, coupler_width, coupler_gap, layer):
    disk_cell = lib.new_cell('disk' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))

    coupler_length = disk_r

    disk = gdspy.Round(center=(0, disk_r + coupler_gap + coupler_width/2),
                       radius=disk_r, layer=layer)
    disk_cell.add(disk)

    coupler_cell = gdspy.CellReference(make_coupler_waveguide(grating, taper, coupler_width, coupler_length, layer=layer),
                                       (0,0))
    disk_cell.add(coupler_cell)

    if use_positive_tone == 1:
        disk_cell = positive_tone(disk_cell)
    
    return disk_cell

def make_gap_sweep(grating, taper, disk_r, coupler_width, gap_list, layer):
    gap_sweep_cell = lib.new_cell('gap_sweep' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    label_cell = lib.new_cell('radius label' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))

    #Add a label
    label = "R=" + str(disk_r)
    radius_label = gdspy.Text(label, size=10)
    label_cell.add(radius_label)
    label_bbox = label_cell.get_bounding_box()
    label_xspan = label_bbox[1][0] - label_bbox[0][0]
    label_yspan = label_bbox[1][1] - label_bbox[0][1]
    
    #if use_positive_tone == 1:
    #    label_cell = positive_tone(label_cell)
    label_ref = gdspy.CellReference(label_cell, (-pattern_x_spacing - label_xspan/2, 2*disk_r - label_yspan))
    gap_sweep_cell.add(label_ref)

    #Add a reference waveguide cell to calibrate polarization angle
    reference_waveguide = make_coupler_waveguide(grating, taper, coupler_width, disk_r, 0)
    if use_positive_tone == 1:
        reference_waveguide = positive_tone(reference_waveguide)
    gap_sweep_cell.add(gdspy.CellReference(reference_waveguide, (-pattern_x_spacing , 0)))

    x = 0
    for gap in gap_list:
        coupler = make_microdisk_and_coupler(grating, taper, disk_r, coupler_width, gap, layer)
        bbox = coupler.get_bounding_box()
        xspan = bbox[1][0] - bbox[0][0]
        coupler_ref = gdspy.CellReference(coupler, (x, 0))
        gap_sweep_cell.add(coupler_ref)
        x = x + pattern_x_spacing
    
    return gap_sweep_cell

def make_gap_and_radius_sweep(grating, taper, radius_list, coupler_width, gap_list, layer):
    radius_sweep_cell = lib.new_cell('radius_and_gap_sweep' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))

    y = 0
    for disk_r in radius_list:
        gap_sweep = make_gap_sweep(grating, taper, disk_r, coupler_width, gap_list, layer)
        bbox = gap_sweep.get_bounding_box()
        yspan = bbox[1][1] - bbox[0][1]
        gap_sweep_ref = gdspy.CellReference(gap_sweep, (0, y))
        radius_sweep_cell.add(gap_sweep_ref)
        y = y + pattern_y_spacing
    
    return radius_sweep_cell

def positive_tone(cell):
    positive_tone_cell = lib.new_cell('positive_tone' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))
    mask_cell = lib.new_cell('mask' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10)))

    bbox = cell.get_bounding_box()

    mask_cell.add(gdspy.Rectangle((bbox[0][0] - border_size, bbox[0][1] - border_size), 
                                  (bbox[1][0] + border_size, bbox[1][1] + border_size)))

    invert = gdspy.boolean(mask_cell, cell, 'not')
    positive_tone_cell.add(invert)

    return positive_tone_cell


#Instantiate fixed objects
coupler_w = 0.525
gap_list = [0.2, 0.3, 0.4, 0.5]
radius_list = [25]

#Pattern spacings
pattern_x_spacing = 150
pattern_y_spacing = 125


#Constants for positive tone resist
use_positive_tone = 1   #If using positive tone resist
border_size = 15 #Size of the border around the patterns when flipping tone
taper_support_width = 8
taper_support_length = 1.05
grating_edge_support = 1
waveguide_border_spacing = 2
waveguide_border_width = 15

grating_coupler = grating(period=1.05, duty_cycle = 0.7, num_gratings = 6, width = 8)
grating_taper = taper(start_width = coupler_w, end_width = 3, length = 10)

lib = gdspy.GdsLibrary()

main_cell = lib.new_cell('main')

sweep = make_gap_and_radius_sweep(grating_coupler, grating_taper, radius_list, coupler_w, gap_list, 0)

bbox = sweep.get_bounding_box()
xshift = -(bbox[0][0] + bbox[1][0]) / 2
yshift = -(bbox[0][1] + bbox[1][1]) / 2

main_cell.add(gdspy.CellReference(sweep, (xshift, yshift)))

lib.write_gds('main_cell.gds')

# Optionally, save an image of the cell as SVG.
main_cell.write_svg('main_cell.svg')

