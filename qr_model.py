#!/usr/bin/python


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
            s += '\t' + n + '\t(' + str(values[n].getValueName()) + ', ' + ['-', '0', '+'][values[n].delta] + ')\t\n'
        return s
    
    def to_string(self, v, d):
        return self.to_string_dict(self.buildValuesDict(v, d))


def expand_node(v, d, model, state_graph, kill=True, split=True):
    children = []
    
    d_original = np.array(d, copy=True, dtype=int)
    
    def build_hidden_node(d, graph, tabs=0):
        print(''.join(['\t' for i in range(tabs + 1)]) + str((v, d)))
        
        valid = model.checkValidity(v, d)
        
        if graph.has_node(('T\n' if split else '') + model.to_string(v, d)):
            return valid
        else:
            
            color = 'red' if valid else 'black'
            
            graph.add_node(('T\n' if split else '') + model.to_string(v, d), color=color)
            
            d_o = np.array(d, copy=True, dtype=int)
            
            values = model.buildValuesDict(v, d)
            
            valid = False
            
            for i, n in enumerate(model.getVariablesNames()):
                
                # print(values['V'].delta)
                for p in model.getDeltaPossibilities(n, values):
                    
                    print(''.join(['\t' for i in range(tabs + 2)]) + n + ': ' + str(p))
                    
                    dp = d[i] - p
                    
                    if math.fabs(dp) == 1:  # and math.fabs(d_original[i] - p) == 1: #
                        is_leaf = False
                        
                        d[i] = p
                        
                        # print('\t\t\t\t' + n + ': ' + str(p))
                        
                        
                        if build_hidden_node(d, graph, tabs + 1):
                            graph.add_edge(('T\n' if split else '') + model.to_string(v, d_o),
                                           ('T\n' if split else '') + model.to_string(v, d),
                                           label='d' + n + ' += ' + str(dp))
                            valid = True
                        
                        d[i] = d_o[i]
            
            if model.checkValidity(v, d_o):
                children.append(d_o)
                return True
            else:
                if not valid:
                    graph.delete_node(('T\n' if split else '') + model.to_string(v, d_o))
                
                return valid
    
    build_hidden_node(d, state_graph)  # pgv.AGraph(directed=True))
    
    return children


def buildNode(v, d, model, state_graph, input, split=True, kill=True):
    if not model.checkValuesValidity(v, d):
        return []
    
    print('Building:')
    print(model.to_string(v, d))
    
    if state_graph.has_node(model.to_string(v, d)):
        return [(v, d)]
    
    children = []
    # if not model.checkValidity(v, d):
    children = expand_node(v, d, model, state_graph, split=split, kill=kill)
    # else:
    #    children = [d]
    
    final_nodes = []
    
    for child in children:
        
        n = (v, child)

        # if model.checkValidity(v, child):
        # print('Child: ' + str(n) + ' valid')
        
        if split:
            state_graph.add_node(model.to_string(*n))
        
        if split or state_graph.has_edge(model.to_string(v, d), model.to_string(v, child)):
            final_nodes.append(n)
        
        steps = model.timeStep(v, child)
        
        for s in steps:
            print('\tstep : ' + str(s) + ' ' + str(child))
            nodes = buildNode(s, child, model, state_graph, input, split)
            
            if split:
                for node in nodes:
                    state_graph.add_edge(model.to_string(v, child), model.to_string(*node), label='time')
            elif len(nodes) > 0:
                state_graph.add_edge(model.to_string(v, child), model.to_string(s, child), label='time')
    
    if not split and not any(np.array_equal(v, t1) and np.array_equal(d, t2) for (t1, t2) in final_nodes):
        final_nodes.append((v, d))
    return final_nodes



def buildNode2(v, d, model, graph, input, kill=True):
    if not model.checkValuesValidity(v, d):
        return False

    if graph.has_node(model.to_string(v, d)):
        return True #model.checkValuesValidity(v, d)
    
    current_node = model.to_string(v, d)
    
    print('Building:')
    print(current_node)
    
    valid = state_node = model.checkValidity(v, d)

    color = 'red' if valid else 'gray'
    style = 'filled, bold' #if valid else ''
    fontcolor = 'black'# if valid else 'red'
    
    graph.add_node(current_node, color=color, style=style, fontcolor=fontcolor, shape='rectangle')

    _d = np.array(d, copy=True, dtype=int)

    values = model.buildValuesDict(v, d)

    for i, n in enumerate(model.getVariablesNames()):
        for p in model.getDeltaPossibilities(n, values):
            dp = p - _d[i]
        
            if math.fabs(dp) == 1:
                _d[i] = p
                if buildNode2(v, _d, model, graph, input, kill=kill) or not kill:
                    graph.add_edge(current_node,
                                   model.to_string(v, _d),
                                   label='d' + n + ' += ' + str(dp))
                    valid = True

                _d[i] = d[i]
    
    steps = model.timeStep(v, d)
    
    changes = [-1, 0, 1] if state_node else [0]
    
    for s in steps:
        print('\tstep : ' + str(s) + ' ' + str(d))
        for c in changes:
            if values[input].delta + c in {-1, 0, 1}:
                _d[model.getVariable(input).index] = values[input].delta + c
                if buildNode2(s, _d, model, graph, input, kill=kill) or not kill:
                    graph.add_edge(current_node, model.to_string(s, _d), label='Time, d' + input + ' += ' + str(c))
                    valid = True

    if not valid and kill:
        graph.delete_node(current_node)
    
    return valid
    
    