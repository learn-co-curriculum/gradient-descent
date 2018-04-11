from plotly import tools

def y_actual(x, x_values, y_values):
    combined_values = list(zip(x_values, y_values))
    point_at_x = list(filter(lambda point: point[0] == x,combined_values))[0]
    return point_at_x[1]

def error(x_values, y_values, m, b, x):
    expected = (m*x + b)
    return (y_actual(x, x_values, y_values) - expected)

def error_line_trace(x_values, y_values, m, b, x):
    y_hat = m*x + b
    y = y_actual(x, x_values, y_values)
    name = 'error at ' + str(x)
    return {'x': [x, x], 'y': [y, y_hat], 'mode': 'line', 'marker': {'color': 'red'}, 'name': name}

def error_line_traces(x_values, y_values, m, b):
    return list(map(lambda x_value: error_line_trace(x_values, y_values, m, b, x_value), x_values))

def squared_error(x_values, y_values, m, b, x):
    return error(x_values, y_values, m, b, x)**2

def squared_errors(x_values, y_values, m, b):
    return list(map(lambda x: squared_error(x_values, y_values, m, b, x), x_values))

def residual_sum_squares(x_values, y_values, m, b):
    return sum(squared_errors(x_values, y_values, m, b))

def trace_rss(x_values, y_values, m, b):
    rss_calc = residual_sum_squares(x_values, y_values, m, b)
    return dict(
        x=['RSS'],
        y=[rss_calc],
        type='bar'
    )

def plot_regression_and_rss(scatter_trace, regression_trace, rss_calc_trace):
    fig = tools.make_subplots(rows=1, cols=2)
    fig.append_trace(scatter_trace, 1, 1)
    fig.append_trace(regression_trace, 1, 1)
    fig.append_trace(rss_calc_trace, 1, 2)
    plotly.offline.iplot(fig)

def trace_rmse(x_values, y_values, regression_lines):
    errors = root_mean_squared_errors(x_values, y_values, regression_lines)
    x_values_bar = list(map(lambda error: 'm: ' + str(error[0]) + ' b: ' + str(error[1]),errors))
    y_values_bar = list(map(lambda error: error[-1], errors))
    return dict(
        x=x_values_bar,
        y=y_values_bar,
        type='bar'
    )
    
def regression_and_rmse(scatter_trace, regression_traces, rss_calc_trace):
    fig = tools.make_subplots(rows=1, cols=2)
    for reg_trace in regression_traces:
        fig.append_trace(reg_trace, 1, 1)
    fig.append_trace(scatter_trace, 1, 1)
    fig.append_trace(rss_calc_trace, 1, 2)
    iplot(fig)
