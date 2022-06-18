# Loading the library
import pypnet

# Creating the P-NET object (pnet) with named parameters. All parameters are optional. 
"pnet = pypnet.new(layers=[layers count], inputs=[inputs count], outputs=[outputs count],"
"intervals=[intervals count], density=[automatically changes the density of intervals, in depends from count of weight calls], epoch=[epochs count],"
"patterns=[patterns count], autointervals=[flag to turn on\off automatic setup count of intervals],"
"pnetfile=[file *.nnw to load p-net from], datafile=[file to load data set for training from]);"

# Loading from file. If *.nnw then P-NET neural network else dataset for training. 
"pypnet.load(pnet, file_name)"

# P-NET training 
pypnet.train(pnet)

# Saving P-NET to file
"pypnet.save(pnet, file_name)"

# Printing array of outputs (results) by a given array of inputs.
"results = pypnet.compute(pnet, inputs)"

# Reading values of P-NET properties (parameters) by name (prop_name: string)
"property = pypnet.get(pnet, prop_name)"
# lists can also be obtained (like 'weights' 'signals' 'coeffis' 'ratings' etc)

# Setting values of P-NET properties (parameters) by name (prop_name: string)
"pypnet.set(pnet, prop_name, value) "

# Example how to use get\set to set half of the P-NET weights array to 0.
"mm = pn.get(pnet, 'inputs') * pn.get(pnet, 'intervals') # расчет количество весов"
if(mm > 0):
"  lst = pn.get(pnet, 'weights')  "
  for i in range(round(mm / 2)): 
    lst[i] = 0.0
"  if(pn.set(pnet, 'weights', lst)): "
"    print(pn.get(pnet, 'weights'))"






