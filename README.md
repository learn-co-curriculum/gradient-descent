
# Improving Regression Lines

### Learning Objectives 

* Understand how to go from RSS to finding a "best fit" line
* Understand a cost curve and what it displays

### Introduction

In the previous section we saw how after choosing the slope and y-intercept values of a regression line, we use the residual sum of squares (RSS) to distill the goodness of fit into one number.  Once doing so, we are pretty close to understanding the gradient descent technique.  But before doing so, let's review and ensure that we understand how to evaluate the accuracy of our line to our data.  

### Review of plotting our data and a regression line

For this example, let's imagine that our data looks like the following:


```python
first_show = {'budget': 100, 'revenue': 275}
second_show = {'budget': 200, 'revenue': 300}
third_show = {'budget': 400, 'revenue': 700}

shows = [first_show, second_show, third_show]
```

> Press shift + enter

Let's again come up with some numbers for a slope and a y-intercept.  Remember that our technique so far is to get at the slope by drawing a line between the first and last points.  And from there, we calculate the value of $b$.  Our `build_regression_line` function, defined in our linear_equations [library](https://github.com/learn-co-curriculum/gradient-descent/blob/master/linear_equations.py), quickly does this for us.

So let's do this with our data above by getting a list of `x_values`, budgets, and `y_values`, revenues, and pass them into our `build_regression_line` function. 


```python
from linear_equations import build_regression_line

budgets = list(map(lambda show: show['budget'], shows))
revenues = list(map(lambda show: show['revenue'], shows))

build_regression_line(budgets, revenues)
```




    {'b': 133.33333333333326, 'm': 1.4166666666666667}



Turning this into a regression formula, it looks like the following.


```python
def regression_formula(x):
    return 1.417*x + 133.33
```

Let's plot this regression formula with our data to get a sense of what this looks like.  First import the necessary libraries to allow us to use `plotly` in our notebook. 


```python
import plotly
from plotly.offline import init_notebook_mode, iplot
from graph import m_b_trace, trace_values, plot
init_notebook_mode(connected=True)

regression_trace = m_b_trace(1.417, 133.33, budgets)
scatter_trace = trace_values(budgets, revenues)
plot([regression_trace, scatter_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="40b83cfb-eaed-47e6-b162-8001328ae6f0" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("40b83cfb-eaed-47e6-b162-8001328ae6f0", [{"x": [100, 200, 400], "y": [275.03000000000003, 416.73, 700.1300000000001], "mode": "line", "name": "line function"}, {"x": [100, 200, 400], "y": [275, 300, 700], "mode": "markers", "name": "data", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


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
    return {'x': [x, x], 'y': [y, y_hat], 'mode': 'line', 'marker': {'color': 'red'}, 'name': name, 'text': [error_value], 'textposition':'right'}

def error_line_traces(x_values, y_values, m, b):
    return list(map(lambda x_value: error_line_trace(x_values, y_values, m, b, x_value), x_values))

errors = error_line_traces(budgets, revenues, 1.417, 133.33)
plot([scatter_trace, regression_trace, *errors])
```


<div id="0f0b7ce3-1232-47e7-9fc5-2134a0792f2e" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("0f0b7ce3-1232-47e7-9fc5-2134a0792f2e", [{"x": [100, 200, 400], "y": [275, 300, 700], "mode": "markers", "name": "data", "text": []}, {"x": [100, 200, 400], "y": [275.03000000000003, 416.73, 700.1300000000001], "mode": "line", "name": "line function"}, {"x": [100, 100], "y": [275, 275.03000000000003], "mode": "line", "marker": {"color": "red"}, "name": "error at 100", "text": [-0.03000000000002956], "textposition": "right"}, {"x": [200, 200], "y": [300, 416.73], "mode": "line", "marker": {"color": "red"}, "name": "error at 200", "text": [-116.73000000000002], "textposition": "right"}, {"x": [400, 400], "y": [700, 700.1300000000001], "mode": "line", "marker": {"color": "red"}, "name": "error at 400", "text": [-0.13000000000010914], "textposition": "right"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


From there, we calculate the `residual sum of squared errors`.


```python
def error(x_values, y_values, m, b, x):
    expected = (m*x + b)
    return (y_actual(x, x_values, y_values) - expected)

def squared_error(x_values, y_values, m, b, x):
    return error(x_values, y_values, m, b, x)**2

def squared_errors(x_values, y_values, m, b):
    return list(map(lambda x: squared_error(x_values, y_values, m, b, x), x_values))

def residual_sum_squares(x_values, y_values, m, b):
    return sum(squared_errors(x_values, y_values, m, b))

squared_errors(budgets, revenues, 1.417, 133.33)
residual_sum_squares(budgets, revenues, 1.417, 133.33)

```




    13625.910700000006



### Moving towards gradient descent

Now that we have the residual sum of squares function to evaluate the accuracy of our regression line, we can simply try out different regression lines and use the regression line that has the lowest RSS.  The regression line that produces the lowest RSS for a given dataset is called the "best fit" line for that dataset.  

So this will be our technique for finding our "best fit" line:

* Choose a regression line with a guess of values for $m$ and $b$
* Calculate the RSS
* Adjust $m$ and $b$, as these are the only things that can vary in a single-variable regression line.
* Again calculate the RSS 
* Repeat this process
* The regression line (that is, the values of $b$ and $m$) with the smallest RSS is our **best fit line**

We'll eventually tweak and improve upon that process, but for now it can get us pretty far.  In fact, let's make things even easier by keeping our value of $m$ fixed, and only changing our value of $b$.  In later lessons, we will change both variables.

Ok, so we have a regression line of $\overline{y} = \overline{m}x + \overline{b} $, and we started with values of $m = 1.41 $ and $b = 133.33 $.  Then seeing how well this regression line matched our dataset, we calculated that $ RSS = 13625.9 $.  Our next step is to plug in different values of $b$ and see how RSS changes.


```python
residual_sum_squares(budgets, revenues, 1.417, 70)
```




    10852.689999999993



| b        | residual sum of squared           | 
| ------------- |:-------------:| 
| 140 | 15318 | 
| 130      |12880| 
| 120      |11042 | 
| 110      |9804| 
|100 | 9166
|90 | 9128
|80 | 9690
|70| 10852

Now notice that while keeping our value of $m$ fixed at 1.417, we can move towards a smaller residual sum of squares (RSS) by changing our value of $b$.  Setting $b$ to 140 produced a higher error than at 130, so we tried moving in the other direction.  We kept moving our $b$ value lower until we set $b$ = 80, at which point our error again increased from the value at 90.  So, we know that a value of $b$ between 80 and 90 produces the smallest RSS, for when $m$ = 1.417. 

This changing output of RSS based on a changing input of different regression lines is called our cost function.  You can see that if we plot our cost function as RSS with changing values of $b$, we get the following:


```python
import plotly
from plotly.offline import init_notebook_mode, iplot
from graph import m_b_trace, trace_values, plot
init_notebook_mode(connected=True)
b_values = list(range(70, 150, 10))
rss = [10852, 9690, 9128, 9166, 9804, 11042, 12880, 15318]
cost_curve_trace = trace_values(b_values, rss, mode="line")
plot([cost_curve_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="c44befe5-53ec-4a57-9fb8-e7f78bca148b" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("c44befe5-53ec-4a57-9fb8-e7f78bca148b", [{"x": [70, 80, 90, 100, 110, 120, 130, 140], "y": [10852, 9690, 9128, 9166, 9804, 11042, 12880, 15318], "mode": "line", "name": "data", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


The graph above is called the cost curve.  It is a plot of the RSS as values of $y$ for different values of $b$.    The curve shows see visually that when $b$ is between 90 and 100 RSS is the lowest.  This technique of adjusting our values to minimize move towards a minimum value is called *gradient descent*.  Here, we *descend* along a cost curve.  When the value of our RSS no longer decreases as we change our variable, we stop.

### Summary

In this section we saw the path from going from calculating the RSS for a given regression line, to finding a best fit line.  Already we see how to move to a better regression line by moving down along our cost curve.  Going forward, we will see ensure that we can move towards our "best fit" line in an efficient manner. 
