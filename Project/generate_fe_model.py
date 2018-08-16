"""
.py

"""

from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True,
                                reportDeprecated=False)
import part

# Parameters
# note: X-Y plane is plan view
BlockXDim = 120. # longitudinal
BlockYDim = 60.  # width
BlockZDim = 60.  # vertical depth
								
								
#-----------------------------------------------------
# Create a model.
#-----------------------------------------------------

myModel = mdb.Model(name='geomodel')

# Create a new viewport in which to display the model
# and the results of the analysis.
myViewport = session.Viewport(name='geomodel', origin=(20, 20), width=150, height=120)
    


#-----------------------------------------------------
# Create the block.
#-----------------------------------------------------
mySketch1 = myModel.ConstrainedSketch(name='blockProfile',sheetSize=1.5*BlockXDim)
mySketch1.rectangle(point1=(-0.5*BlockXDim,0.5*BlockYDim), point2=(0.5*BlockXDim,-0.5*BlockYDim))
myBlock = myModel.Part(name='Block', dimensionality=THREE_D, type=DEFORMABLE_BODY)
myBlock.BaseSolidExtrude(sketch=mySketch1, depth=BlockZDim)

#-----------------------------------------------------
# Create the cylinder.
#-----------------------------------------------------
mySketchPath = myModel.ConstrainedSketch(name='cylinderPath',sheetSize=1.5*BlockXDim)
mySketchPath.Line(point1=(0.0,-100.0), point2=(0.0,100.0))

mySketch2 = myModel.ConstrainedSketch(name='cylinderProfile',sheetSize=1.5*BlockXDim)
mySketch2.CircleByCenterPerimeter(center=(0.0,0.0), point1=(10.,0.))
myCylinder = myModel.Part(name='Cylinder', dimensionality=THREE_D, type=DEFORMABLE_BODY)
#####myCylinder.BaseSolidExtrude(sketch=mySketch2, depth=BlockZDim)
myCylinder.BaseSolidSweep(sketch=mySketch2, path=mySketchPath)


#-----------------------------------------------------
# Create the tunnel assembly
#-----------------------------------------------------
a = mdb.models['geomodel'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)


p = mdb.models['geomodel'].parts['Block']
a.Instance(name='Block-1', part=p, dependent=ON)

p = mdb.models['geomodel'].parts['Cylinder']
a.Instance(name='Cylinder-1', part=p, dependent=ON)

# translate the block
a.translate(instanceList=('Block-1', ), vector=(0.0, 0.0, -0.5*BlockZDim))

# Create the tunnel instance
#a1 = mdb.models['geomodel'].rootAssembly
a.InstanceFromBooleanCut(name='tunnel', 
    instanceToBeCut=a1.instances['Block-1'], 
    cuttingInstances=(a1.instances['Cylinder-1'],), 
    originalInstances=SUPPRESS)
	
# Rotate entire tunnel assembly so that Z is now along the vertical
#a.rotateAboutAxis(instanceList=('tunnel', ), axisPoint=(0.0, 0.0, 0.0), axisDirection= , angle)
