#!/usr/bin/python

from toposort import toposort, toposort_flatten
import math
import numpy as np
import pygraphviz as pgv


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
        if i >= len(self.values_types):
            print('ERROR! ' + str(i) + ' ' + self.name)
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

        self.p_dependencies = {}
        self.i_dependencies = {}

        for i, v in enumerate(self.variables):
            self.p_dependencies[v.name] = set()
            self.i_dependencies[v.name] = set()
            v.setIndex(i)

        self.topsort = self.getVariablesNames()
        
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
        
        self.p_dependencies[v2].add((v1, incremental))

    def addIConstraint(self, incremental, v1, v2):
        
        incremental = 1 if incremental else -1
        
        var1, var2 = self.getVariable(v1), self.getVariable(v2)

        print('adding ' + v1 + ' as I source for ' + v2)
        
        var1.addIConstraintSource(incremental, v2)
        var2.addIConstraintTarget(incremental, v1)

        self.i_dependencies[v2].add((v1, incremental))
    
    def dependencies_sort(self):
        
        dependencies = {t: {s for (s, i) in l} for t, l in self.p_dependencies.items()}
        print(dependencies)
        self.topsort = toposort_flatten(dependencies)
        print(self.topsort)
        
    def addVConstraint(self, constraint):
        self.v_constraints.append(constraint)

    def i_constraint(self, v1, v2, incremental, values):
    
        sign = incremental * min(1, values[v1].val)
        
        s = incremental * values[v1].delta
        
        m = s < 0
        p = s > 0
        z = s == 0 or values[v2].delta != 0 or values[v2].val == values[v2].variable.getMaxVal()

        return sign, m, z, p
    
    def p_constraint(self, v1, v2, incremental, values):
        return incremental * values[v1].delta

    def checkValidity(self, values):
    
        for k, v in values.items():
            if not v.isValid():
                return False
    
        for c in self.v_constraints:
            if not c(values):
                return False
    
        return True

    def checkValuesValidity(self, values):
    
        for k, v in values.items():
            if not v.isValidValue():
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

    def envisioning(self, v, d, input=None, graph=None):
        if graph is None:
            graph = pgv.AGraph(directed=True, fixedsize=True)
        
        paths_dict = {}
        
        self.dependencies_sort()
        
        self.branches(v, d, graph, paths_dict, input=input)
        
        for i, n in enumerate(graph.nodes()):
            n.attr['id'] = i
        
        return graph

    def branches(self, v, d, graph, paths_dict, input):
        
        id = ''.join([str(i) for i in list(v) + list(d)])
        
        if id in paths_dict:
            return paths_dict[id]

        paths_dict[id] = []
        
        if not self.checkValuesValidity(self.buildValuesDict(v, d)):
            return []
        
        def branches_rec(current, deltas):
            values = self.buildValuesDict(v, deltas)
    
            _d = np.array(deltas, copy=True, dtype=int)
            
            if current >= len(self.variables):
                print('inferred ' + str(deltas) + ' from ' + str(d))
                if self.checkValidity(values):
                    print('VALID')
                    print('\tchanges ' + str(_d - d))
                    paths_dict[id].append((_d, _d - d))
                else:
                    print('NOT VALID')
                return
            
            variable = self.topsort[current]
            print('Inferring ' + variable)
            var_idx = self.getVariable(variable).index
            
            possibilities = set()
            incr, stat, decr = False, False, False
            so_m, so_z, so_p = False, False, False
            
            for s, i in self.i_dependencies[variable]:
                sign, m, z, p = self.i_constraint(s, variable, i, values)
                
                incr |= sign > 0
                decr |= sign < 0
                stat |= sign == 0
                
                if not(so_m or so_z or so_p):
                    so_m, so_z, so_p = m, z, p
                if (m or so_m) and (p or so_p):
                    so_m, so_z, so_p = True, True, True
                elif so_z and not so_m and not so_p:
                    so_m, so_z, so_p = m, z, p
                elif z and not m and not p:
                    so_z = so_z and z
                else:
                    so_z = so_z and z
                    so_p = p
                    so_m = m
            
            for s, i in self.p_dependencies[variable]:
                sign = self.p_constraint(s, variable, i, values)
                stat |= sign == 0
                incr |= sign > 0
                decr |= sign < 0
            
            use_second_order = False
            if incr and decr:
                if so_m or so_z or so_p:
                    possibilities = set()
                    if so_m:
                        possibilities.add(-1)
                    if so_z:
                        possibilities.add(0)
                    if so_p:
                        possibilities.add(1)
                    
                    if values[variable].delta in possibilities:
                        possibilities.add(0)
                    use_second_order = True
                else:
                    possibilities = {-1, 0, 1}
            elif not incr and not decr:
                if stat:
                    possibilities = {0}
                else:
                    possibilities = {values[variable].delta}
            elif incr:
                possibilities.add(1)
            elif decr:
                possibilities.add(-1)
                
            print('\tpossibilities' + ('(USE 2nd ORDER)' if use_second_order else '') + ' = ' + str(possibilities))
            
            if use_second_order:
                for p in possibilities:
                    if _d[var_idx] + p in {-1, 0, 1}:
                        _d[var_idx] += p
                        branches_rec(current+1, _d)
                        _d[var_idx] -= p
            else:
                for p in possibilities:
                    if math.fabs(deltas[var_idx] - p) <=1:
                        _d[var_idx] = p
                        branches_rec(current+1, _d)
                        _d[var_idx] = values[variable].delta
        
        branches_rec(0, d)
        
        valid = False
        for _d, dp in paths_dict[id]:
            if not dp.any():
                valid = True
                children = paths_dict[id]
                paths_dict[id] = [(_d, dp)]
        
        if valid:
            node = self.build_envisioning(v, d, graph, paths_dict, input)
            
            for _d, dp in children:
                if dp.any():
                    child_node = self.build_envisioning(v, _d, graph, paths_dict, input)
                    label = '\n'.join(
                        ['d' + self.variables[i].name + ' += ' + str(dp[i]) for i in range(len(self.variables)) if dp[i] != 0]
                    )
                    graph.add_edge(node,
                                   child_node,
                                   label = label
                                   )
            
        else:
            for _d, dp in paths_dict[id]:
                self.build_envisioning(v, _d, graph, paths_dict, input)

        return paths_dict[id]
        
    def build_envisioning(self, v, d, graph, paths_dict, input=None):
        values = self.buildValuesDict(v, d)
        
        current_node = self.to_string_dict(values)
        
        if graph.has_node(current_node):
            return current_node
        
        print('building')
        print(current_node)
        
        graph.add_node(current_node, color='red', style = 'filled, bold')

        graph.get_node(current_node).attr.update({k: v for k, v in zip(self.variables_names_to_strings(), np.concatenate((v, d)))})
        
        _d = np.array(d, copy=True, dtype=int)
        
        def perform_step(s, _d, label):
            print('STEP: ' + ''.join([str(i) for i in list(s) + list(_d)]))
            
            ds = s -v
            label += '\n'.join(
                [self.variables[i].name + ' += ' + str(ds[i]) for i in range(len(self.variables)) if ds[i] != 0]
            )
            label += '\n'
            nodes = self.branches(s, _d, graph, paths_dict, input)
            for n, dp in nodes:
                l = '\n'.join(
                    ['d' + self.variables[i].name + ' += ' + str(dp[i]) for i in range(len(self.variables)) if dp[i] != 0]
                )
                graph.add_edge(current_node, self.to_string(s, n), label= label + '\n\n' + l, timestep=True)

        steps = self.timeStep(v, _d)
        
        stationary = False
        for s in steps:
            if not (v - s).any():
                stationary = True
        
        if input is None or not stationary:
            for s in steps:
                perform_step(s, _d, 'TimeStep\n')
        else:
            input_idx = self.getVariable(input).index
            for c in {-1, 0, 1}:
                if c + _d[input_idx] in {-1, 0, 1}:
                    _d[input_idx] += c
                    
                    for s in steps:
                        perform_step(s, _d, 'TimeStep\n' + ('change d' + input + ' += ' + str(c) +'\n' if c!= 0 else ''))
                    
                    _d[input_idx] -= c
        
        return current_node