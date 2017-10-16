#!/usr/bin/python

import sys
import pygraphviz as pgv
import matplotlib.pyplot as plt
import numpy as np
import math

class Variable:
    def __init__(self, name, values, values_types): #, zero):
        if len(values) != len(values_types):
             raise ValueError('Error! values and values_types have different sizes!')
        
        for v in values_types:
            if not type(v) is bool:
                raise ValueError('Error! values_types elements have to be boolean!')
        
        self.name = name
        self.values = values
        self.values_types = values_types
        
        self.p_source = []
        self.p_target = []
        self.i_source = []
        self.i_target = []
        
#         if not self.isValidValue(zero):
#             raise ValueError('Error! Not valid "zero" index!')
#         else:
#             self.zero = zero
    
    def getMaxVal(self):
        return len(self.values) -1
    
    def setIndex(self, i):
        self.index = i
    
    def isRangeValue(self, i):
        return self.values_types[i]
    
    def getNumValues(self):
        return len(self.values)

    def getValueName(self, v):
        return self.values[v]
    
    def isValidValue(self, value):
        if not isinstance(value, VariableValue):
            raise ValueError('Error! VariableValue expected!')
    
        return value.val >= 0 and \
               value.val < self.getNumValues()

    def isValid(self, value):
        if not isinstance(value, VariableValue):
            raise ValueError('Error! VariableValue expected!')
    
        return self.isValidValue(value) and \
               (self.values_types[-1] or value.val != self.getMaxVal() or value.delta <= 0) and \
               (self.values_types[0] or value.val != 0 or value.delta >= 0)
        
        
        
    
    def addPConstraintTarget(self, c):
        if not isinstance(c, P_Constraint):
            raise ValueError('Error! P_Constraint expected!')
        if c.v2 != self:
            raise ValueError('Error! P_Constraint with this variable as target expected!')
        
        self.p_target.append(c)
    
    def addPConstraintSource(self, c):
        if not isinstance(c, P_Constraint):
            raise ValueError('Error! P_Constraint expected!')
        if c.v1 != self:
            raise ValueError('Error! P_Constraint with this variable as source expected!')
        
        self.p_source.append(c)
    
    def addIConstraintTarget(self, c):
        if not isinstance(c, I_Constraint):
            raise ValueError('Error! I_Constraint expected!')
        if c.v2 != self:
            raise ValueError('Error! I_Constraint with this variable as target expected!')
        
        self.i_target.append(c)
    
    def addIConstraintSource(self, c):
        if not isinstance(c, I_Constraint):
            raise ValueError('Error! I_Constraint expected!')
        if c.v1 != self:
            raise ValueError('Error! I_Constraint with this variable as source expected!')
        
        self.i_source.append(c)

class P_Constraint:
    
    def __init__(self, incremental, v1, v2):
        if not type(incremental) is bool:
             raise ValueError('Error! Expected boolean value for incremental!')
        
        if not isinstance(v1, Variable) or not isinstance(v2, Variable):
            raise ValueError('Error! Expected a Variable instance!')
            
        self.incremental = 1 if incremental else -1
        self.v1 = v1
        self.v2 = v2
        
        v1.addPConstraintSource(self)
        v2.addPConstraintTarget(self)
    

    def constraintDeriv(self, variables):
        return set([-1, 0, 1])

    def constraintSign(self, variables):
        # if the constraint is applicable, i.e., the hypothesis holds and the delta
        # of the second variable is incrementable or decrementable (it is 0)
    
        return self.incremental * variables[self.v1.name].delta
        
    
class I_Constraint:
    
    def __init__(self, incremental, v1, v2):
        if not type(incremental) is bool:
             raise ValueError('Error! Expected boolean value for incremental!')
        
        if not isinstance(v1, Variable) or not isinstance(v2, Variable):
            raise ValueError('Error! Expected a Variable instance!')
            
        self.incremental = 1 if incremental else -1
        self.v1 = v1
        self.v2 = v2
        
        v1.addIConstraintSource(self)
        v2.addIConstraintTarget(self)  
    
   
    def constraintDeriv(self, variables):
        s = set({self.incremental * variables[self.v1.name].delta})
        if variables[self.v2.name].delta == 0:
            s |= {0}
        return s
        
        
    def constraintSign(self, variables):
        #if the constraint is applicable, i.e., the hypothesis holds and the delta
        #of the second variable is incrementable or decrementable (it is 0)
        
        return self.incremental * min(1, variables[self.v1.name].val)
        
        
class V_Constraint:
    
    #the 'constraint' function should have as a signature: dict{"variable_name" : VariableValue}
    #and access variables using this dictionary and the name used to define the variables
    
    def __init__(self, constraint): #, v1, v2):
        if not callable(constraint):
             raise ValueError('Error! Expected "constraint" to be a boolean function!')
        
        #if not isinstance(v1, Variable) or not isinstance(v2, Variable):
        #    raise ValueError('Error! Expected a Variable instance!')
            
        self.constraint = constraint
        #self.v1 = v1
        #self.v2 = v2
    
    
    
    def checkConstraint(self, variables):
        return self.constraint(variables)

class VariableValue:
    
    def __init__(self, variable, val=None, delta=None):
        
        if not isinstance(variable, Variable):
            raise ValueError('Error! Expected a Variable instance!')
        
        if (val != None and type(val) != int) or (delta != None and type(delta) != int):
            raise ValueError('Error! Expected int type for val and delta!')
        
        self.variable = variable
        self.val = val
        self.delta = delta
    
    def isRangeValue(self):
        return self.variable.isRangeValue(self.val)
    
    def getVariable(self):
        return self.variable
    
    def setValue(self, i):
        if not self.variable.isValidValue(i):
            return False
        else:
            self.val = i
            return True
    
    def setDelta(self, i):
        if i not in {-1, 0, 1}:
            return False
        else:
            self.delta = i
            return True
    
    def incrementValue(self, i):
        return self.setValue(self.val + i)
    
    def isValidValue(self):
        return self.variable.isValidValue(self)
    
    def isValid(self):
        return self.variable.isValid(self)
    
    def getValueName(self):
        return self.variable.getValueName(self.val)
    
class Model:
    def __init__(self, variables):
        if not type(variables) is list:
            raise ValueError('Error! expected list of Variable!')
        
        for v in variables:
            if not isinstance(v, Variable):
                raise ValueError('Error! expected list of Variable!')
        
        self.variables = variables
        self.v_constraints = []
        self.p_constraints = []
        self.i_constraints = []
        
        for i, v in enumerate(self.variables):
            v.setIndex(i)
    
    def pointValuesMask(self, v):
        m = np.zeros(len(self.variables), dtype=int)
        
        for var in self.variables:
            m[var.index] = 1 - int(var.isRangeValue(v[var.index]))
        
        return m
        
    def getRangeValueVariables(self, values):
        l = []
        
        for v in self.variables:
            if values[v.name].isRangeValue():
                l.append(v)
        
        return l
    
    def timeStep(self, v, d):
        
        m = self.pointValuesMask(v)
        
        
        values = self.buildValuesDict(v, d)
        range_values = self.getRangeValueVariables(values)
        
        
        
        vals = v + m*d
        
        combinations = []
        
        def generate(i, v):
            if i < 0:
                combinations.append(np.array(v, copy=True, dtype=int))
            else:
                generate(i-1, v)
                
                if d[range_values[i].index] != 0:
                    v[range_values[i].index] += d[range_values[i].index]
                    generate(i-1, v)
                    v[range_values[i].index] -= d[range_values[i].index]
        
        generate(len(range_values)-1, vals)
        
        return combinations
    
    def buildValuesDict(self, values, deltas):
        if len(values) != len(deltas) or len(deltas) != len(self.variables):
            raise ValueError("Error! the length of values and delta has to be the same of the number of variables!")
        
        dictionary = {}
        
        for i, (v, d) in enumerate(zip(values, deltas)):
            dictionary[self.variables[i].name] = VariableValue(self.variables[i], int(v), int(d))
        
        return dictionary
    
    def buildValuesArray(self, values):
        if len(values) != len(self.variables):
            raise ValueError("Error! the length of values and delta has to be the same of the number of variables!")
        
        N = self.getNVariables()
        
        v = np.empty(N, dtype=int)
        d = np.empty(N, dtype=int)
        
        for i, var in enumerate(self.variables):
            v[i] = values[var.name].val
            d[i] = values[var.name].delta
        
        return v, d
    
    
        
    def getVariablesNames(self):
        names = []
        for v in self.variables:
            names.append(v.name)
        return names
        
    def getVariable(self, idx):
        if type(idx) is int:
            return self.variables[idx]
        elif type(idx) is str:
            for v in self.variables:
                if v.name == idx:
                    return v
            return None
        else:
            raise ValueError('Error! Expected integer or string!')
        
    def getNVariables(self):
        return len(self.variables)
    
    def addPConstraint(self, incremental, v1, v2):
        c = P_Constraint(incremental, self.getVariable(v1), self.getVariable(v2))
        self.p_constraints.append(c)
    
    def addVConstraint(self, constraint):
        c = V_Constraint(constraint)
        self.v_constraints.append(c)
    
    def addIConstraint(self, incremental, v1, v2):
        c = I_Constraint(incremental, self.getVariable(v1), self.getVariable(v2))
        self.i_constraints.append(c)

    def getDeltaPossibilities(self, variable, values, use_second_order=True):
        var = self.getVariable(variable)

        # incr, stat, decr = False, False, False
        incr, decr = False, False
        stat = len(var.i_target + var.p_target) > 0
        second_order = set()
        
        for c in var.i_target + var.p_target:
            s = c.constraintSign(values)
            
            if use_second_order:
                second_order |= c.constraintDeriv(values)
            
            #print(c.v1.name + '  ' + c.v2.name + ':    ' + str(s))
            
            incr = incr or s > 0
            # stat = stat or s == 0
            decr = decr or s < 0

            stat = stat and s == 0
    
        possibilities = []
        if incr and not decr:
            possibilities.append(1)
        elif decr and not incr:
            possibilities.append(-1)
        elif stat:
            possibilities.append(0)
        elif incr and decr:
            if use_second_order:
                if -1 in second_order and 1 in second_order:
                    second_order |= {0}
                possibilities = set([values[var.name].delta + d for d in second_order]) & {-1, 0, 1}
            else:
                possibilities = [-1, 0, 1]
            
        return possibilities


    def checkValuesValidity(self, v, d):
    
        values = self.buildValuesDict(v, d)
    
        for k, v in values.items():
            if not v.isValidValue():
                return False
    
        for c in self.v_constraints:
            if not c.checkConstraint(values):
                return False
            
        return True

    def checkValidity(self, v, d):
        values = self.buildValuesDict(v, d)
    
        for k, v in values.items():
            if not v.isValid():
                return False
            
            deltas = self.getDeltaPossibilities(v.variable.name, values, use_second_order=False)
            if len(deltas) > 0 and not v.delta in deltas:
                return False
    
        for c in self.v_constraints:
            if not c.checkConstraint(values):
                return False
    
        return True


    def to_string_dict(self, values):
        s=""
        for n in self.getVariablesNames():
            s += '\t' + n + '\t(' + str(values[n].getValueName()) + ', ' + ['-', '0', '+'][values[n].delta +1] + ')\t\n'
        return s
    
    def to_string(self, v, d):
        return self.to_string_dict(self.buildValuesDict(v, d))



def envisioning(v, d, model, input=None, graph=None, kill=0):
    
    if graph ==None:
        graph = pgv.AGraph(directed=True)
    
    paths_dict = {}
    
    try:
        build_envisioning(v, d, model, graph, paths_dict, input=input, kill=kill)
    finally:
        return graph
    

def build_envisioning(v, d, model, graph, paths_dict, input=None, kill=0):
    
    if not model.checkValuesValidity(v, d):
        return {}

    current_node = model.to_string(v, d)
    
    if current_node in paths_dict:

        if len(paths_dict[current_node]) > 40:
            print((v,d))
            
            for l, n in paths_dict[current_node]:
                print('to ' + n)
                print(l)
            print('TOO LONG')
            raise ValueError('too long')

        return paths_dict[current_node]
    
    print('Building:')
    print(current_node)
    
    valid = state_node = model.checkValidity(v, d)

    color = 'red' if valid else 'gray'
    style = 'filled, bold' #if valid else ''
    fontcolor = 'black'# if valid else 'red'
    
    graph.add_node(current_node, color=color, style=style, fontcolor=fontcolor, shape='rectangle', validity=state_node)
    
    paths = {current_node : ['']} if state_node else {}
    
    paths_dict[current_node] = paths
    
    _d = np.array(d, copy=True, dtype=int)

    values = model.buildValuesDict(v, d)

    def update(paths, new_paths, v, d, label):
        if state_node:
            if kill == 0:
                for (node, l) in new_paths.items():
                    graph.add_edge(current_node, node, label='\n'.join([label] + l))
        elif model.to_string(v, d) != current_node:
            for (node, l) in new_paths.items():
                if node not in paths or len(paths[node]) > len(l) + 1:
                    paths[node] = [label] + l
            
        if kill > 0:
            graph.add_edge(current_node,
                           model.to_string(v, d),
                           label=label)

    for i, n in enumerate(model.getVariablesNames()):
        for p in model.getDeltaPossibilities(n, values):
            dp = p - _d[i]
        
            if math.fabs(dp) == 1:
                _d[i] = p
                new_paths = build_envisioning(v, _d, model, graph, paths_dict, input, kill=kill)
                
                label = 'd' + n + ' += ' + str(dp)
                if len(new_paths) > 0:
                    valid = True
                    update(paths, new_paths, v, _d, label)
                    
                _d[i] = d[i]

    if state_node:
        steps = model.timeStep(v, d)
        
        _d = np.array(d, copy=True, dtype=int)
        
        for s in steps:
            print('\tstep : ' + str(s) + ' ' + str(d))
            if input != None:
                for c in range(values[input].delta-1, values[input].delta+2):
                    if c in {-1, 0, 1}:
                        _d[model.getVariable(input).index] = c
    
                        new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                        label = 'Time, d' + input + ' += ' + str(c)
    
                        if len(new_paths) > 0:
                            update(paths, new_paths, s, _d, label)
            else:
                new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                label = 'Time'
    
                if len(new_paths) > 0:
                    update(paths, new_paths, s, _d, label)
                        
    if not valid or (kill == 0 and not state_node):
        graph.delete_node(current_node)
    
    return paths
    
    