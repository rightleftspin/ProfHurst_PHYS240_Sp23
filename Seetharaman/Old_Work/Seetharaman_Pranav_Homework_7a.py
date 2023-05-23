import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import lmfit

def general_fit_function(function_string, x_data, y_data, guess_parameters):
    """
    This function takes in a function string to be parsed by Sympy's
    expression parser and tries to fit the data given along the 'x'
    independent variables. 
    """
    # Use sympy's parser to find the expression from the input string
    parsed_expression = sy.parsing.sympy_parser.parse_expr(function_string)
    parsed_function = sy.lambdify(list(parsed_expression.free_symbols), parsed_expression)
    
    # Create a model based on this input string with the independent variable x
    parsed_lm_model = lmfit.Model(parsed_function, independent_vars = ['x'])
    # Run the fit of the model to the data starting with the given parameters
    fit_to_model = parsed_lm_model.fit(data = y_data, x = x_data, **guess_parameters)

    return(fit_to_model, parsed_function)

def poly_order_n(order):
    """
    Returns a string containing a polynomial of the appropriate order
    """
    # constant element of poly function
    poly_string = "c_0"
    poly_dict = {"c_0": 0}

    # add all nonconstant elements
    for i in range(1, order + 1):
        poly_dict[f"c_{i}"] = 0
        poly_string += f" + c_{i} * (x ** {i})"
    
    return(poly_string, poly_dict)


def fit_general_poly(order, x_data, y_data, guess_parameters):
    """
    Takes in a specific order for polynomial and fits the data to
    that order polynomial
    """
    # Create polynomial string
    poly_string, poly_dict = poly_order_n(order)
    # fit data to polynomial string
    fit_model, poly_function = general_fit_function(poly_string, x_data, y_data, guess_parameters)

    return(fit_model, poly_function)

def fit_sine(x_data, y_data, guess_parameters):
    """
    Takes data that can be fit to a sine function, and 
    attemts the fit to that data, it first casts the data
    into a sin function, then attempts a linear fit
    """
    # Create linear string
    line_string = "c_0 + c_1 * x" 
    # fit data to sine string
    fit_model, line_function = general_fit_function(line_string, np.sin(x_data), y_data, guess_parameters)

    return(fit_model, line_function)

if __name__ == "__main__":
    """
    Starting the first question here
    """
    # Load data from text file 
    sin_poly_data = np.loadtxt('curve_data.txt', skiprows = 1)
    # Split data into individual variables
    x_data, y_data, error_bar = sin_poly_data.transpose()

    fig = plt.figure(0)
    ax = plt.subplot(111)
    
    # Plot the input data with errorbars
    ax.errorbar(x_data, y_data, error_bar, marker='.', linestyle='')

    # fit to the highest order polynomial possible
    # for this number of data points
    for poly_order in range(3, len(x_data) - 1):
        # get all the nessecary starting parmeters
        _, poly_dict = poly_order_n(poly_order)
        # find the fit of the polynomial
        fit_model, poly_function = fit_general_poly(poly_order, x_data, y_data, poly_dict)
        # plot each polynomial fit and write down the chi squared
        ax.plot(x_data, poly_function(x = x_data, **fit_model.values), label = f"Polynomial of Order {poly_order}, Chi-Sqr = {fit_model.chisqr:.3f}")

    # Fit the sine function
    fit_model, sine_function = fit_sine(x_data, y_data, {"c_0": 0, "c_1": 0})

    # Refit the model to a sine function
    y_sine = fit_model.values["c_0"] + fit_model.values["c_1"] * np.sin(x_data)
    ax.plot(x_data, y_sine, label = f"Sine Function, Chi-Sqr = {fit_model.chisqr:.2f}")

    # Resize the box to include the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.35, box.width, box.height * 0.65])

    # Important information about the data
    plt.title("Fits for Curve Data")
    plt.xlabel("Unknown X Data")
    plt.ylabel("Unknown Y Data")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f"./output/HW_7/curve_data.pdf")
    plt.close()


    """
        Starting the second question here
    """

    # Write down data, assuming constant error bar of about 10 points
    x_data, y_data, error_bars = np.array([1, 2, 3, 4, 5]), np.array([2470, 2510, 2410, 2350, 2240]), np.array([30] * 5)

    fig = plt.figure(1)
    ax = plt.subplot(111)
    
    # Plot the input data with errorbars, but add the extra data point for later vizualization
    # this extra data point won't be used in fitting, but will be extrapolated to later
    new_x_data, new_y_data, new_error_bars = np.append(x_data, 6), np.append(y_data, 1740), np.append(error_bars, 30)
    ax.errorbar(new_x_data, new_y_data, new_error_bars, marker='.', linestyle='')

    # fit to the highest order polynomial possible
    # for this number of data points
    for poly_order in range(1, len(x_data)):
        # get all the nessecary starting parmeters
        _, poly_dict = poly_order_n(poly_order)
        # find the fit of the polynomial
        fit_model, poly_function = fit_general_poly(poly_order, x_data, y_data, poly_dict)
        # plot each polynomial fit and write down the chi squared
        ax.plot(new_x_data, poly_function(x = new_x_data, **fit_model.values), label = f"Polynomial of Order {poly_order}")
    

    # Resize the box to include the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.75])

    plt.title("Fits for Stock Market Data")
    plt.xlabel("Day (Days)")
    plt.ylabel("Dow Jones Average (Points)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f"./output/HW_7/stock_analysis.pdf")
    plt.close()









