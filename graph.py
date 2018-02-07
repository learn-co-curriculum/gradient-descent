import plotly
from plotly.offline import iplot

def trace(data, mode = 'markers', name="data"):
    x_values = list(map(lambda point: point['x'],data))
    y_values = list(map(lambda point: point['y'],data))
    return {'x': x_values, 'y': y_values, 'mode': mode, 'name': name}

def line_function_data(line_function, x_values):
    y_values = list(map(lambda x: line_function(x), x_values))
    return {'x': x_values, 'y': y_values}

def line_function_trace(line_function, x_values, mode = 'line', name = 'line function'):
    values = line_function_data(line_function, x_values)
    return values.update({'mode': mode, 'name': name})

def error_line(regression_line, point):
    y_hat = regression_line(point['x'])
    x_value = point['x']
    name = 'error at ' + str(x_value)
    return {'x': [x_value, x_value], 'y': [point['y'], y_hat], 'mode': 'line', 'marker': {'color': 'red'}, 'name': name}

def error_lines(regression_line, points):
    return list(map(lambda point: error_line(regression_line, point), points))

def plot(traces):
    plotly.offline.iplot(traces)
