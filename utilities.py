'''Script to create a sequential dataset from a normal dataset using diff modes such as :
1.classification mode          : Each timestep in one timesequence will have explicit target values
2.Sequence Classification mode : There is only one label for whole timesequence, not for individual timesteps
3.Transcription mode           : Unsegmented sequence labelling, without knowing alignment of inp/targ seqs:TODO'''

from numpy import *
from pylab import * 
from pybrain.datasets.classification import SequenceClassificationDataSet
 

def create_dataset(inp,targ,numclasses,numtimesteps,length,mode="SC"):
   timestep_size=inp.shape[1]  
   ds=SequenceClassificationDataSet(timestep_size,1,nb_classes=numclasses)
   for i in range(length):
     ds.newSequence()
     img=inp[i] 
     target=targ[i]
     if mode =="SC":
        #default mode is Seq Classification so prepare dataset as follows
        for timestep in range(numtimesteps):
           ds.addSample(img,target)
     if mode=="C":  #classification mode
       pass
     if mode =="Transcription":
       pass
   #convert dataset to one of many representations (multiclass) 
   ds._convertToOneOfMany()
   return ds 


#more fns to be added
 
