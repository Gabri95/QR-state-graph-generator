#!/usr/bin/python

import numpy as np


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
    
        return value.val >= 0 and value.val < self.getNumValues()

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
        
        if (not val is None and type(val) != int) or (not delta is None and type(delta) != int):
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
            m[var.index] = int(not var.isRangeValue(v[var.index]))
        
        return m
        
    def getRangePointValueVariables(self, v):
        r = []
        p = []
        for var in self.variables:
            if var.isRangeValue(v[var.index]):
                r.append(var.index)
            else:
                p.append(var.index)
        
        return r, p
    
    def timeStep(self, v, d):
        
        range_values, point_values = self.getRangePointValueVariables(v)
        
        vals = np.array(v, copy=True, dtype=int)
        
        for i in point_values:
            vals[i] += d[i]
        
        def generate(i, v, combinations):
            if i < 0:
                combinations.append(np.array(v, copy=True, dtype=int))
            else:
                generate(i-1, v, combinations)
                
                if d[range_values[i]] != 0:
                    v[range_values[i]] += d[range_values[i]]
                    generate(i-1, v, combinations)
                    v[range_values[i]] -= d[range_values[i]]

        combinations = []
        generate(len(range_values)-1, vals, combinations)
        
        return combinations
    
    def buildValuesDict(self, values, deltas):
        if len(values) != len(deltas) or len(deltas) != len(self.variables):
            raise ValueError("Error! the length of values and delta has to be the same of the number of variables!")
        
        dictionary = {}
        
        for i, (v, d) in enumerate(zip(values, deltas)):
            dictionary[self.variables[i].name] = VariableValue(self.variables[i], int(v), int(d))
        
        return dictionary
    
        
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
        s = {incremental * values[v1].delta}
        if values[v2].delta == 0:
            s |= {0}
        return incremental * min(1, values[v1].val), s
    
    def p_constraint(self, v1, v2, incremental, values):
        return incremental * values[v1].delta

    def getDeltaPossibilities(self, variable, values, use_second_order=True):
        var = self.getVariable(variable)

        incr, decr = False, False
        second_order = set()

        for c, i in var.i_target.items():
            sign, so = self.i_constraint(c, variable, i, values)
    
            if use_second_order:
                second_order |= so
    
            incr = incr or sign > 0
            decr = decr or sign < 0
        
        for c, i in var.p_target.items():
            sign = self.p_constraint(c, variable, i, values)
    
            incr = incr or sign > 0
            decr = decr or sign < 0
            
    
        if len(var.i_target) + len(var.p_target) == 0:
            return []
        elif incr and not decr:
            return [1]
        elif decr and not incr:
            return [-1]
        elif not decr and not incr:
            return [0]
        else:
            if use_second_order:
                if -1 in second_order and 1 in second_order:
                    second_order |= {0}
                return set([values[var.name].delta + d for d in second_order]) & {-1, 0, 1}
            else:
                return [-1, 0, 1]


    def checkValuesValidity(self, values):
    
        for k, v in values.items():
            if not v.isValidValue():
                return False
    
        for c in self.v_constraints:
            if not c(values):
                return False
            
        return True

    def checkValidity(self, values):
    
        for k, v in values.items():
            if not v.isValid():
                return False
            
            deltas = self.getDeltaPossibilities(v.variable.name, values, use_second_order=False)
            if len(deltas) > 0 and v.delta not in deltas:
                return False
    
        for c in self.v_constraints:
            if not c(values):
                return False
    
        return True

    def to_string_dict(self, values):
        s=''
        for i, n in enumerate(self.getVariablesNames()):
            #s += '\t' + n + '\t(' + str(values[n].getValueName()) + ', ' + ['-', '0', '+'][values[n].delta +1] + ')\t\n'
            s += n + '(' + str(values[n].getValueName()) + ', ' + ['-', '0', '+'][values[n].delta + 1] + ')'
            if i < len(self.variables)-1:
                s += '\n'
        return s
    
    def to_string(self, v, d):
        return self.to_string_dict(self.buildValuesDict(v, d))

    def variables_names_to_strings(self):
        vals =[]
        deltas = []
        
        for v in self.variables:
            vals.append('v' + v.name)
            deltas.append('d' + v.name)
        return vals + deltas

