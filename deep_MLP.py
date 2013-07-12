#!/usr/bin/python
'''Script to create a deep MLP using 3 hidden layers , one input and one output layer'''
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer,SigmoidLayer,SoftmaxLayer,FullConnection
from numpy import * 
import cPickle 
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError 
from pybrain.supervised.trainers import BackpropTrainer
from pylab import *
from matplotlib import pyplot as plt  
import sys 
from time import time 
#from pybrain.tools.neuralnets import saveTrainingCurve,saveNetwork

def find_gradients(results,initial,final,weight_range):
    #function to plot the numerical and analytical gradients of a particular network 
    #initial and final indicate the limits that you set for the weights that are to be considerered
    #results=trainer._checkGradient(dataset=dset,silent=True)
    analytic_gradients=[]
    numeric_gradients=[] 
    for i in range(len(results)):
        #loop will be repeated 100 times because of the 100 training datapts 
        final_conn = results[i][initial:final] #last 30 values 
        #print len(final_conn)
        assert len(final_conn) == weight_range
        conn_wts=[]
        an_conn=[]
        for conn in final_conn:
             conn_wts.append(conn[1])
             an_conn.append(conn[0])
        #now take the max out and append it to the hidden_[] list 
        maxconn = max(conn_wts)
        maxconn_an = max(an_conn)
        numeric_gradients.append(maxconn)
        analytic_gradients.append(maxconn_an)
    return (numeric_gradients,analytic_gradients)
    
    
def create_dataset(data_x,data_y): 
    dataset=ClassificationDataSet(784,1,nb_classes=10)
    length=data_x.shape[0] 
    for i  in range(length):
       img = data_x[i]
       target=data_y[i]
       dataset.addSample(img,[target])
    return dataset

if __name__=="__main__": 
    #create the network 
    n=FeedForwardNetwork() 
    hidden_layers=[] 
    num_hidden=int(sys.argv[1])  #first argument to program is the number of hidden layers that you require, usually should not go  beyond 3 
    hidden_size=int(sys.argv[2])  #number of (sigmoidal) neurons in the hidden layer(s) 
    print "Number of hidden layers : ", num_hidden 
    print "Hidden size of the hidden layers : " , hidden_size 
    for i in range(num_hidden):
          hidden_layers.append(SigmoidLayer(hidden_size))
    inLayer=LinearLayer(784)
    outLayer=SoftmaxLayer(10)
    
    n.addInputModule(inLayer)
    for i in range(num_hidden): 
       n.addModule(hidden_layers[i]) 
    n.addOutputModule(outLayer) 
  
    #create the connections and add them 
    #in_to_hidden_0=FullConnection(inLayer,hidden_layers[0])
    #hidden_0_to_hidden_1 = FullConnection(hidden_layers[0],hidden_layers[1])
    #hidden_1_to_hidden_2 = FullConnection(hidden_layers[1],hidden_layers[2])
    #hidden_2_to_out = FullConnection(hidden_layers[2],outLayer)

    #n.addConnection(in_to_hidden_0) 
    #n.addConnection(hidden_0_to_hidden_1)
    #n.addConnection(hidden_1_to_hidden_2)
    #n.addConnection(hidden_2_to_out) 
    #n.sortModules()
    conn_in_to_hidden = FullConnection(inLayer,hidden_layers[0])
    conn_hidden_to_out = FullConnection(hidden_layers[-1],outLayer)
    #now we make hidden connections 
    i=0
    hidden_connections=[]   #the hidden connections will be stored in this list above   
    while(i<num_hidden-1):
       hidden_conn = FullConnection(hidden_layers[i],hidden_layers[i+1])
       hidden_connections.append(hidden_conn) 
       i=i+1 
    #now add the connections 
    n.addConnection(conn_in_to_hidden)
    n.addConnection(conn_hidden_to_out)
    for conn in hidden_connections: 
       n.addConnection(conn) 
    #sort and initialize the modules
    n.sortModules()
        
    print "Network : "  , n 
    f = open("/home/saurav/Desktop/CVPR_work/ocropus/ocropy/mnist.pkl","rb")
    train,valid,test=cPickle.load(f)
    train_x,train_y = train
    valid_x,valid_y=valid
    test_x,test_y = test

    trndata=create_dataset(train_x ,train_y)
    validdata=create_dataset(valid_x,valid_y)
    testdata=create_dataset(test_x,test_y)

    ####toy example#####
    trndata._convertToOneOfMany()
    testdata._convertToOneOfMany()
    validdata._convertToOneOfMany() 
    ## data basically divided up into multi vector representations 
    #create the trainer that uses the backprop algo 
    trainer=BackpropTrainer(n,dataset=trndata,momentum=0.1,verbose=True)
    results=trainer._checkGradient(dataset=trndata,silent=True)
    #do the gradient check for this network and we will plot the results later 
    #wt_container_sizes=[] #this list contains the sizes of all the wt connections 
    #wt_container_sizes.append(784*hidden_size)  #input_to_hidden_0 
    #if more hidden layers exist
    #i=0 
    #while i<num_hidden-1:
    #  wt=hidden_size*hidden_size  
    #  wt_container_sizes.append(wt)
    #  i=i+1 
    #wt_container_sizes.append(hidden_size*10)  #the hidden_to_output_connections 
    #print "Weight connections are : " , wt_container_sizes 
    #TODO later use the wt_container_sizes[] list 
    print "Using 3 hidden layers containing 3 hidden neurons each" 
    grads=[] 
    grads.append(find_gradients(results,0,7840,7840))
    grads.append(find_gradients(results,7840,7940,100))
    grads.append(find_gradients(results,7940,8040,100))
    grads.append(find_gradients(results,8040,8140,100))
    fig=plt.figure()
    xlabel("Epochs")
    ylabel("inputs to hidden_0 weight gradients")
    plot(grads[0][0],'r')
    plot(grads[0][1],'g')
    fig.savefig('input_to_hidden_0_grads.png')
    fig=plt.figure() 
    xlabel('Epochs') 
    ylabel('Hidden_0_to_hidden_1 weight gradients') 
    plot(grads[1][0],'r')
    plot(grads[1][1],'g')
    fig.savefig('hidden_0_to_hidden_1_grads.png')
    fig=plt.figure()
    xlabel('Epochs') 
    ylabel('hidden_1_to_hidden_2_gradients') 
    plot(grads[2][0],'r')
    plot(grads[2][1],'g')
    fig.savefig('hidden_1_to_hidden_2_gradients.png') 
    fig=plt.figure() 
    xlabel('Epochs') 
    ylabel('hidden_2_to_output_wt_gradients') 
    plot(grads[3][0],'r') 
    plot(grads[3][1],'g')
    fig.savefig('hidden_2_to_output_wt_gradients.png') 
    print "Checking gradients finished" 
    print "Starting training ..... " 
    numEpochs=1000 #setup for maxEpochs to run if early stopping condition is not met
    trnerrors=[]
    validerrors=[]
    count=0
    bestValidError=float("inf")
    t1=time()
    for epoch in range(numEpochs): 
        trainer.trainEpochs(1)
        trnerror=percentError(trainer.testOnClassData(),trndata['class'])
        validerror=percentError(trainer.testOnClassData(dataset=validdata),validdata['class'])
        #if validation error after n epochs doesn't decrease we stop the iterations
        trnerrors.append(trnerror)
        validerrors.append(validerror)
        if epoch==0:  
            bestValidError=validerror
            count=count+1
        if validerror < bestValidError:
            count=0 
            bestValidError=validerror
            continue 
        if count>30:
           print "30 Error tests without decrease in valid error. Stopping !!" 
           break 
        count=count+1 
    t2=time() 
    print "Training ended after " , (t2-t1)/(60*60.) , "hours"     
    #compute the test error
    tsterror = percentError(trainer.testOnClassData(dataset=testdata),testdata['class'])
    print "Test Error is : %5.2f%%" %(tsterror)
    #print "Saving training curves and network to disk"
    #saveTrainingCurve('training_curve.csv')
    #saveNetwork('deep_MLP.net')
    name=str(num_hidden)+"_hidden_layers_"+str(hidden_size)+".png"
    fig =plt.figure()
    subplot(211)
    xlabel("Epochs")
    ylabel("training errors % ")
    plot(trnerrors)
    subplot(212)
    xlabel("Epochs")
    ylabel("validation Errors")
    plot(validerrors,'g')
    fig.savefig(name)
    show()
    
    
