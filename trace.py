#!/usr/bin/python

import os
from qr_model import *

def traceEdge(state_graph,edge,from_id,out):
    to_id=state_graph.get_node(edge[1]).attr["id"]
    label=edge.attr["label"]
    nextS=state_graph.get_node(edge[1])
    previousS=state_graph.get_node(edge[0])
    out.write("\t ("+from_id+" => "+to_id+")\n")
    
    desc=""
    
    dV=nextS.attr["dV"]
    dI=nextS.attr["dI"]
    dO=nextS.attr["dO"]
    V=nextS.attr["vV"]
    I=nextS.attr["vI"]
    O=nextS.attr["vO"]
    
    if  (state_graph.get_node(edge[0]).attr["vI"]!="0") and (I=="0"):
        desc+="If our Inflow reaches zero "
    elif("dV += 1" in label) and (V=="0") and previousS.attr["vV"]=="1":
        desc+="If the water volume reaches zero and the outflow is null"
    elif("dV += -1" in label) and (V=="2") and previousS.attr["vV"]=="1":
        desc+="If the water volume reaches the bathtub capacity and the outflow reaches the maximum"
        if "dI += -1" in label:
            desc+=", we close the tap"
    elif "change" in label:
        desc+="If we "
        if(dI=="0"):
            if "dI += -1" in label:
                desc+="stop turning on the tap"
            if "dI += 1" in label:
                desc+="stop turning off the tap"
        elif(dI=="-1"):
            desc+="start closing the tap"
        elif(dI=="1"):
            desc+="start opening the tap"
        else:
            desc+="error"
    else:
        desc+="If we let time pass"
        
    if(from_id==to_id):
        desc+=", we can stay in this State with the same situation"
    else:   
        if(I=="1"):
            if "\nI += 1" in label:
                desc+=", the water starts flowing in"
        elif(I=="0") and ("\nI += -1" in label):
            desc+=", the tap is now closed"
        
        if(dV=="0"):
            if "dV += -1" in label:
                desc+=", the volume and the outflow stop increasing"
            if "dV += 1" in label:
                desc+=", the volume and the outflow stop decreasing"
        elif(dV=="1") and ("dV += 1" in label):
            desc+=", the volume and the outflow start increasing"
        elif(dV=="-1") and ("dV += -1" in label):
            desc+=", the volume and the outflow start decreasing"
        
        if(V=="1"):
            if "\nV += -1" in label:
                desc+=", the water is nomore overflowing"
            if "\nV += 1" in label:
                desc+=", the water starts flowing into the bathtub"
        desc+=", then we move to State "+to_id
    
    out.write("\t\t"+desc+":\n")
    out.write("\t\t\t"+nextS.replace("\n","\n\t\t\t")+"\n")
#     print(state_graph.get_node(edge[1]))

    
    
def traceNode(state_graph,node,out):
    attr=node.attr
    current_id=node.attr["id"]
    out.write("STATE "+current_id+"\n")
    out.write(node+"\n")
    
    dV=node.attr["dV"]
    dI=node.attr["dI"]
    dO=node.attr["dO"]
    V=node.attr["vV"]
    I=node.attr["vI"]
    O=node.attr["vO"]
#     print (dI,dV,dO,V,I,O)
    
    out.write("\tI:\t")
    if(I=="1"):
        out.write("The water is flowing in and ")
    elif(I=="0"):
        out.write("There is no water flowing in and ")

    if(dI=="1"):
        out.write("I am opening the tap")
    elif(dI=="0"):
        out.write("I am not touching the tap")
    elif(dI=="-1"):
        out.write("I am closing the tap")
    out.write("\n")
    
    
    out.write("\tV:\t")
    if(V=="2"):
        out.write("The bathtub is full and ")
    elif(V=="1"):
        out.write("There is water in the bathtub and ")
    elif(V=="0"):
        out.write("There is no water in the bathtub and ")
        
    if(dV=="1"):
        out.write("the volume of water is increasing")
    elif(dV=="0"):
        out.write("the volume of water is not changing")
    elif(dV=="-1"):
        out.write("the volume of water is decreasing")
    out.write("\n")
    
    
    out.write("\tO:\t")
    if(O=="2"):
        out.write("The flow of the sink is maximum and ")
    elif(O=="1"):
        out.write("There is water flowing out from the sink and ")
    elif(O=="0"):
        out.write("There is no water flowing out and ")
        
    if(dO=="1"):
        out.write("the outflow is increasing")
    elif(dO=="0"):
        out.write("the outflow is not changing")
    elif(dO=="-1"):
        out.write("the outflow is decreasing")
    out.write("\n")
    
    neighbors=state_graph.out_edges([node])
    out.write("\nTransitions starting from State "+current_id+":\n")
    for i in range(len(neighbors)):
        out.write("\n")
        traceEdge(state_graph,neighbors[i],current_id,out)
    out.write("\n_________________________________________________________________\n\n\n")
    

def trace(state_graph):
    out = open("./trace.txt","w")

    for i in state_graph.nodes():
        traceNode(state_graph,i,out)
        
    out.close()
    print("Trace succesfully generated and saved as trace.txt")