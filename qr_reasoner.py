import math
import numpy as np
import pygraphviz as pgv


def envisioning(v, d, model, input=None, graph=None, kill=0):
    
    if graph is None:
        graph = pgv.AGraph(directed=True, fixedsize=True)
    
    paths_dict = {}

    build_envisioning(v, d, model, graph, paths_dict, input=input, kill=kill)
    
    for i, n in enumerate(graph.nodes()):
        n.attr['id'] = i
    
    return graph


def build_envisioning(v, d, model, graph, paths_dict, input=None, kill=0):
    
    values = model.buildValuesDict(v, d)
    
    if not model.checkValuesValidity(values):
        return {}

    current_node = model.to_string_dict(values)
    
    if current_node in paths_dict:
        return paths_dict[current_node]
    
    #print('Building:')
    #print(current_node)
    
    valid = state_node = model.checkValidity(values)

    color = 'red' if valid else 'gray'
    style = 'filled, bold' #if valid else ''
    fontcolor = 'black'# if valid else 'red'
    

    
    graph.add_node(current_node, color=color, style=style, fontcolor=fontcolor, shape='record', validity=state_node)

    graph.get_node(current_node).attr.update({k : v for k,v in zip(model.variables_names_to_strings(), np.concatenate((v, d)))})
    
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
        if not input is None and values[input].isRangeValue():
            for c in {-1, 0, 1}:  # range(values[input].delta-1, values[input].delta+2):
                if c + delta in {-1, 0, 1}:
                    for s in steps:
                        _d[model.getVariable(input).index] = c +delta
                
                        new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                        label = 'Time, d' + input + ' += ' + str(c+delta)
                
                        if len(new_paths) > 0:
                            update(paths, new_paths, s, _d, label)
            
            _d[model.getVariable(input).index] = delta
        else:
    
            for c in {-1, 1}:
                if c + delta in {-1, 0, 1}:
                    _d[model.getVariable(input).index] = c + delta
            
                    new_paths = build_envisioning(v, _d, model, graph, paths_dict, input, kill=kill)
                    label = 'Time, d' + input + ' += ' + str(c + delta)
            
                    if len(new_paths) > 0:
                        update(paths, new_paths, v, _d, label)
    
            _d[model.getVariable(input).index] = delta
            
            for s in steps:
                new_paths = build_envisioning(s, _d, model, graph, paths_dict, input, kill=kill)
                label = 'Time'
                if len(new_paths) > 0:
                    update(paths, new_paths, s, _d, label)
        
    if not valid or (kill == 0 and not state_node):
        graph.delete_node(current_node)
    
    return paths