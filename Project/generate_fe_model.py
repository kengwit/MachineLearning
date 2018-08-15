"""
.py

"""

from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True,
                                reportDeprecated=False)

# Parameters
BlockWidth  = 200.
BlockHeight = 20.
BlockLength = 25.
								
								
# Create a model.

myModel = mdb.Model(name='geomodel')

# Create a new viewport in which to display the model
# and the results of the analysis.

myViewport = session.Viewport(name='geomodel', origin=(20, 20), width=150, height=120)
    
#-----------------------------------------------------

import part

# Create a sketch for the base feature.

mySketch = myModel.ConstrainedSketch(name='blockProfile',sheetSize=1.5*BlockWidth)

# Create the rectangle.

mySketch.rectangle(point1=(-0.5*BlockWidth,0.5*BlockHeight), point2=(0.5*BlockWidth,-0.5*BlockHeight))

# Create a three-dimensional, deformable part.

myBlock = myModel.Part(name='Block', dimensionality=THREE_D, type=DEFORMABLE_BODY)

# Create the part's base feature by extruding the sketch through a distance of 25.0.

myBlock.BaseSolidExtrude(sketch=mySketch, depth=BlockLength)

#-----------------------------------------------------
