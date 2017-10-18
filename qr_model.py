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
        
        self.p_source = {}
        self.p_target = {}
        self.i_target = {}
        self.i_source = {}
        
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
        
        
        
    
    def addPConstraintTarget(self, incremental, src):
        if type(src) != str:
            raise ValueError('Error! string expected as src!')
        
        if src not in self.p_target:
            self.p_target[src] = incremental
            
        elif self.p_target[src] != incremental:
            raise ValueError('Error! Added the same Proportional constraint on ' + self.name + ' with two different signs!')
        
    
    def addPConstraintSource(self, incremental, target):
        if type(target) != str:
            raise ValueError('Error! string expected as target!')
    
        if target not in self.p_source:
            self.p_source[target] = incremental

        elif self.p_source[target] != incremental:
            raise ValueError('Error! Added the same Proportional constraint from ' + self.name + ' with two different signs!')

    def addIConstraintTarget(self, incremental, src):
        if type(src) != str:
            raise ValueError('Error! string expected as src!')
    
        if src not in self.i_target:
            self.i_target[src] = incremental
    
        elif self.i_target[src] != incremental:
            raise ValueError(
                'Error! Added the same Incremental constraint on ' + self.name + ' with two different signs!')

    
    def addIConstraintSource(self, incremental, target):
        if type(target) != str:
            raise ValueError('Error! string expected as target!')
    
        if target not in self.i_source:
            self.i_source[target] = incremental
            
        elif self.i_source[target] != incremental:
            raise ValueError(
                'Error! Added the same Incremental constraint from ' + self.name + ' with two different signs!')



        
    
        
        

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
    
        incremental = 1 if incremental else -1
        
        var1, var2 = self.getVariable(v1), self.getVariable(v2)

        print('adding ' + v1 + ' as P source for ' + v2)
        
        var1.addPConstraintSource(incremental, v2)
        var2.addPConstraintTarget(incremental, v1)
        self.p_constraints.append((v1, v2, incremental))

        for v, i in var1.p_target.items():
            print('\tadding ' + v + ' as P source for ' + v2)
            var2.addPConstraintTarget(i * incremental, v)
            self.getVariable(v).addPConstraintSource(i * incremental, v2)
            for t, incr in var2.p_source.items():
                print('\tadding ' + v + ' as P source for ' + t)
                self.getVariable(t).addPConstraintTarget(i * incr * incremental, v)
                self.getVariable(v).addPConstraintSource(i * incr * incremental, t)
        
        for v, i in var1.i_target.items():
            print('\tadding ' + v + ' as I source for ' + v2)
            var2.addIConstraintTarget(i * incremental, v)
            self.getVariable(v).addIConstraintSource(i * incremental, v2)
            for t, incr in var2.p_source.items():
                print('\tadding ' + v + ' as I source for ' + t)
                self.getVariable(t).addIConstraintTarget(i * incr * incremental, v)
                self.getVariable(v).addIConstraintSource(i * incr * incremental, t)

    def addIConstraint(self, incremental, v1, v2):
        
        incremental = 1 if incremental else -1
        
        var1, var2 = self.getVariable(v1), self.getVariable(v2)

        print('adding ' + v1 + ' as I source for ' + v2)
        
        var1.addIConstraintSource(incremental, v2)
        var2.addIConstraintTarget(incremental, v1)
        self.i_constraints.append((v1, v2, incremental))
        
        
        
        # for t, incr in var2.p_source.items():
        #     print('\tadding ' + v1 + ' as I source for ' + t)
        #     self.getVariable(t).addIConstraintTarget(incremental * incr, v1)
        #     var1.addIConstraintSource(incremental * incr, t)
        
        
    def addVConstraint(self, constraint):
        self.v_constraints.append(constraint)

    def i_constraint(self, v1, v2, incremental, values):
        s = set({incremental * values[v1].delta})
        if values[v2].delta == 0:
            s |= {0}
        return incremental * min(1, values[v1].val), s
    
    def p_constraint(self, v1, v2, incremental, values):
        return incremental * values[v1].delta, {-1, 0, 1}

    
    def getDeltaPossibilities(self, variable, values, use_second_order=True):
        var = self.getVariable(variable)

        incr, decr = False, False
        stat = len(var.i_target) + len(var.p_target) > 0
        second_order = set()

        for c, i in var.p_target.items():
            sign, so = self.p_constraint(c, variable, i, values)
    
            if use_second_order:
                second_order |= so
    
            incr = incr or sign > 0
            decr = decr or sign < 0
            stat = stat and sign == 0
        
        for c, i in var.i_target.items():
            sign, so = self.i_constraint(c, variable, i, values)
    
            if use_second_order:
                second_order |= so
    
            incr = incr or sign > 0
            decr = decr or sign < 0
            stat = stat and sign == 0
            
        
    
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
            if not c(values):
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
            if not c(values):
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
    
    if graph is None:
        graph = pgv.AGraph(directed=True)
    
    paths_dict = {}

    build_envisioning(v, d, model, graph, paths_dict, input=input, kill=kill)
    return graph
    
    # try:
    #     build_envisioning(v, d, model, graph, paths_dict, input=input, kill=kill)
    # finally:
    #     return graph
    #

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

        delta = values[input].delta
        if input != None and values[input].isRangeValue():
            for c in {-1, 0, 1}:  # range(values[input].delta-1, values[input].delta+2):
                if c + delta in {-1, 0, 1}:
                    for s in steps:
                        _d[model.getVariable(input).index] = c +delta
                        print('\tstep : ' + str(s) + ' ' + str(_d))
                        
                
                        new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                        label = 'Time, d' + input + ' += ' + str(c+delta)
                
                        if len(new_paths) > 0:
                            update(paths, new_paths, s, _d, label)
            
            _d[model.getVariable(input).index] = delta
        else:
    
            for c in {-1, 1}:
                if c + delta in {-1, 0, 1}:
                    _d[model.getVariable(input).index] = c + delta
                    print('\tstep : ' + str(v) + ' ' + str(_d))
            
                    new_paths = build_envisioning(v, _d, model, graph, paths_dict, input, kill=kill)
                    label = 'Time, d' + input + ' += ' + str(c + delta)
            
                    if len(new_paths) > 0:
                        update(paths, new_paths, v, _d, label)
    
            _d[model.getVariable(input).index] = delta
            
            for s in steps:
                print('\tstep : ' + str(s) + ' ' + str(_d))
                new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                label = 'Time'
    
                if len(new_paths) > 0:
                    update(paths, new_paths, s, _d, label)
        
        
            
                        
    if not valid or (kill == 0 and not state_node):
        graph.delete_node(current_node)
    
    return paths
    
    