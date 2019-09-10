
# Gradient Descent

### Learning Objectives 

* Understand how to go from RSS to finding a "best fit" line
* Understand a cost curve and what it displays

### Introduction

In the previous section we saw how after choosing the slope and y-intercept values of a regression line, we can calculate the residual sum of squares (RSS) and related root mean squared error.  We can use either the RSS or RMSE to calculate the accuracy of a line.  In this lesson we'll proceed with RSS as it's the simpler of the two.

Once calculating the accuracy of a line, we are pretty close to improving upon a line by minimizing the RSS.  This is the task of the gradient descent technique.  But before learning about gradient descent, let's review and ensure that we understand how to evaluate how our line fits our data.  

### Review of plotting our data and a regression line

For this example, let's imagine that our data looks like the following:


```python
first_movie = {'budget': 100, 'revenue': 275}
second_movie = {'budget': 200, 'revenue': 300}
third_movie = {'budget': 250, 'revenue': 550}
fourth_movie = {'budget': 325, 'revenue': 525}
fifth_movie = {'budget': 400, 'revenue': 700}

shows = [first_movie, second_movie, third_movie, fourth_movie, fifth_movie]
```

> Press shift + enter

Let's again come up with some numbers for a slope and a y-intercept.  

>Remember that our technique so far is to get at the slope by drawing a line between the first and last points.  And from there, we calculate the value of $b$.  Our `build_regression_line` function, defined in our [linear_equations library](https://github.com/learn-co-curriculum/gradient-descent/blob/master/linear_equations.py), quickly does this for us.

So let's convert our data above into a list of `x_values`, budgets, and `y_values`, revenues, and pass them into our `build_regression_line` function. 


```python
from linear_equations import build_regression_line

budgets = list(map(lambda show: show['budget'], shows))
revenues = list(map(lambda show: show['revenue'], shows))

build_regression_line(budgets, revenues)
```




    {'m': 1.4166666666666667, 'b': 133.33333333333326}



Turning this into a regression formula, we have the following.


```python
def regression_formula(x):
    return 1.417*x + 133.33
```

Let's plot this regression formula with our data to get a sense of what it looks like. 


```python
# First import the `plotly` libraries and functions in our notebook. 
import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# then import our graph functions
from graph import m_b_trace, trace_values, plot

regression_trace = m_b_trace(1.417, 133.33, budgets)
scatter_trace = trace_values(budgets, revenues)
plot([regression_trace, scatter_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="850ecb7c-e468-4c29-8fc8-e68c839852cd" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("850ecb7c-e468-4c29-8fc8-e68c839852cd", [{"mode": "lines", "name": "line function", "x": [100, 200, 250, 325, 400], "y": [275.03000000000003, 416.73, 487.58000000000004, 593.855, 700.1300000000001], "type": "scatter", "uid": "3381e812-d3e6-11e9-84da-3af9d3ad3e0b"}, {"mode": "markers", "name": "data", "text": [], "x": [100, 200, 250, 325, 400], "y": [275, 300, 550, 525, 700], "type": "scatter", "uid": "3381e928-d3e6-11e9-a458-3af9d3ad3e0b"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


### Evaluating the regression line

Ok, now we add in our functions for displaying the errors for our graph.


```python
from graph import trace, plot, line_function_trace

def y_actual(x, x_values, y_values):
    combined_values = list(zip(x_values, y_values))
    point_at_x = list(filter(lambda point: point[0] == x,combined_values))[0]
    return point_at_x[1]

def error_line_trace(x_values, y_values, m, b, x):
    y_hat = m*x + b
    y = y_actual(x, x_values, y_values)
    name = 'error at ' + str(x)
    error_value = y - y_hat
    return {'x': [x, x], 'y': [y, y_hat], 'mode': 'lines', 'marker': {'color': 'red'}, 'name': name, 'text': [error_value], 'textposition':'top right'}

def error_line_traces(x_values, y_values, m, b):
    return list(map(lambda x_value: error_line_trace(x_values, y_values, m, b, x_value), x_values))

errors = error_line_traces(budgets, revenues, 1.417, 133.33)
plot([scatter_trace, regression_trace, *errors])
```


<div id="ca91bd31-22f2-4b4d-9060-e54dfb4105f8" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ca91bd31-22f2-4b4d-9060-e54dfb4105f8", [{"mode": "markers", "name": "data", "text": [], "x": [100, 200, 250, 325, 400], "y": [275, 300, 550, 525, 700], "type": "scatter", "uid": "34e73cf4-d3e6-11e9-a72a-3af9d3ad3e0b"}, {"mode": "lines", "name": "line function", "x": [100, 200, 250, 325, 400], "y": [275.03000000000003, 416.73, 487.58000000000004, 593.855, 700.1300000000001], "type": "scatter", "uid": "34e73e1e-d3e6-11e9-bc64-3af9d3ad3e0b"}, {"marker": {"color": "red"}, "mode": "lines", "name": "error at 100", "text": ["-0.03000000000002956"], "textposition": "top right", "x": [100, 100], "y": [275, 275.03000000000003], "type": "scatter", "uid": "34e73eb4-d3e6-11e9-843c-3af9d3ad3e0b"}, {"marker": {"color": "red"}, "mode": "lines", "name": "error at 200", "text": ["-116.73000000000002"], "textposition": "top right", "x": [200, 200], "y": [300, 416.73], "type": "scatter", "uid": "34e73f36-d3e6-11e9-8385-3af9d3ad3e0b"}, {"marker": {"color": "red"}, "mode": "lines", "name": "error at 250", "text": ["62.41999999999996"], "textposition": "top right", "x": [250, 250], "y": [550, 487.58000000000004], "type": "scatter", "uid": "34e73fae-d3e6-11e9-823a-3af9d3ad3e0b"}, {"marker": {"color": "red"}, "mode": "lines", "name": "error at 325", "text": ["-68.85500000000002"], "textposition": "top right", "x": [325, 325], "y": [525, 593.855], "type": "scatter", "uid": "34e7401c-d3e6-11e9-9800-3af9d3ad3e0b"}, {"marker": {"color": "red"}, "mode": "lines", "name": "error at 400", "text": ["-0.13000000000010914"], "textposition": "top right", "x": [400, 400], "y": [700, 700.1300000000001], "type": "scatter", "uid": "34e7408a-d3e6-11e9-9dc8-3af9d3ad3e0b"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


From there, we calculate the `residual sum of squared errors` and the `root mean squared error`.


```python
import math
def error(x_values, y_values, m, b, x):
    expected = (m*x + b)
    return (y_actual(x, x_values, y_values) - expected)

def squared_error(x_values, y_values, m, b, x):
    return round(error(x_values, y_values, m, b, x)**2, 2)

def squared_errors(x_values, y_values, m, b):
    return list(map(lambda x: squared_error(x_values, y_values, m, b, x), x_values))

def residual_sum_squares(x_values, y_values, m, b):
    return round(sum(squared_errors(x_values, y_values, m, b)), 2)

def root_mean_squared_error(x_values, y_values, m, b):
    return round(math.sqrt(sum(squared_errors(x_values, y_values, m, b))/len(x_values)), 2)

squared_errors(budgets, revenues, 1.417, 133.33) #[0.0, 13625.89, 3896.26, 4741.01, 0.02]
residual_sum_squares(budgets, revenues, 1.417, 133.33) # 22263.18
root_mean_squared_error(budgets, revenues, 1.417, 133.33) # 66.73
```




    66.73



### Moving towards gradient descent

Now that we have the residual sum of squares function to evaluate the accuracy of our regression line, we can simply try out different regression lines and use the regression line that has the lowest RSS.  The regression line that produces the lowest RSS for a given dataset is called the "best fit" line for that dataset.  

So this will be our technique for finding our "best fit" line:

> * Choose a regression line with a guess of values for $m$ and $b$
> * Calculate the RSS
> * Adjust $m$ and $b$, as these are the only things that can vary in a single-variable regression line.
> * Again calculate the RSS 
> * Repeat this process
> * The regression line (that is, the values of $b$ and $m$) with the smallest RSS is our **best fit line**

We'll eventually tweak and improve upon that process, but for now it will do.  In fact, we will make things even easier at first by holding $m$ fixed to a constant value while we experiment with different $b$ values.  In later lessons, we will change both variables.

#### Updating the regression line to improve accuracy

Ok, so we have a regression line of $\hat{y} = mx + b $, and we started with values of $m = 1.417 $ and $b = 133.33 $.  Then seeing how well this regression line matched our dataset, we calculated that $ RSS = 22,263.18 $.  Our next step is to plug in different values of $b$ and see how RSS changes.  Let's try $b$ = 140 instead of $133.33$.


```python
residual_sum_squares(budgets, revenues, 1.417, 140)
```




    24130.78



Now let's the RSS for a variety of $b$ values.


```python
def residual_sum_squares_errors(x_values, y_values, regression_lines):
    errors = []
    for regression_line in regression_lines:
        error = residual_sum_squares(x_values, y_values, regression_line[0], regression_line[1])
        errors.append([regression_line[0], regression_line[1], round(error, 0)])
    return errors
```


```python
b_values = list(range(70, 150, 10))

m_values = [1.417]*8
regression_lines = list(zip(m_values, b_values))
regression_lines
```




    [(1.417, 70),
     (1.417, 80),
     (1.417, 90),
     (1.417, 100),
     (1.417, 110),
     (1.417, 120),
     (1.417, 130),
     (1.417, 140)]




```python
rss_lines = residual_sum_squares_errors(budgets, revenues, regression_lines)
rss_lines
```




    [[1.417, 70, 26696.0],
     [1.417, 80, 23330.0],
     [1.417, 90, 20963.0],
     [1.417, 100, 19597.0],
     [1.417, 110, 19230.0],
     [1.417, 120, 19864.0],
     [1.417, 130, 21497.0],
     [1.417, 140, 24131.0]]



| b        | residual sum of squares           | 
| ------------- |:-------------:| 
| 140| 24131
| 130      |21497| 
| 120      |19864 | 
| 110      |19230| 
|100 | 19597
|90 | 20963
|80 | 23330
|70| 26696

Notice what the above chart represents.  While keeping our value of $m$ fixed at 1.417, we moved towards a smaller residual sum of squares (RSS) by changing our value of $b$, our y-intercept. 

Setting $b$ to 130 produced a lower error than at 140.  We kept moving our $b$ value lower until we set $b$ = 100, at which point our error began to increase.  Therefore, we know that a value of $b$ between 110 and 100 produces the smallest RSS for our data while $m = 1.417 $. 

This changing output of RSS based on a changing input of different regression lines is called our **cost function**.  Let's plot this chart to see it better.

We set:

* `b_values` as the input values (x values), and
* `rss_errors` as the output values (y values)


```python
b_values = list(range(70, 150, 10))

# remember that each element in rss_lines has the m value, b value, and related rss error
# rss_lines[0] => [1.417, 70, 26696.0]
# so we collect the rss errors for each regression line  
rss_errors = list(map(lambda line: line[-1], rss_lines))
```


```python
import plotly
from plotly.offline import init_notebook_mode, iplot
from graph import m_b_trace, trace_values, plot
init_notebook_mode(connected=True)


cost_curve_trace = trace_values(b_values, rss_errors, mode="lines")
plot([cost_curve_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="865c0b6f-2aa1-4fbc-a85b-ff01503d49a8" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("865c0b6f-2aa1-4fbc-a85b-ff01503d49a8", [{"mode": "lines", "name": "data", "text": [], "x": [70, 80, 90, 100, 110, 120, 130, 140], "y": [26696.0, 23330.0, 20963.0, 19597.0, 19230.0, 19864.0, 21497.0, 24131.0], "type": "scatter", "uid": "5583ae82-d3e6-11e9-b1e0-3af9d3ad3e0b"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


The graph above is called the **cost curve**.  It is a plot of the RSS for different values of $b$.    The curve demonstrates that when $b$ is between 100 and 120, the RSS is lowest.  This technique of optimizing towards a minimum value is called *gradient descent*.  Here, we *descend* along a cost curve.  As we change our variable, we need to stop when the value of our RSS no longer decreases.

### Summary

In this section we saw the path from going from calculating the RSS for a given regression line, to finding a line that minimizes our RSS - a best fit line.  We learned that we can move to a better regression line by descending along our cost curve.  Going forward, we will learn how to move towards our best fit line in an efficient manner. 
