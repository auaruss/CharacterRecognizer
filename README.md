# B351_Final
B351 Final Project

This is a project to automatically generate, train, validate, and test a CNN on the EMNIST data set.

To get the dependency for this code, use "pip install tensorflow" or "python -m pip install tensorflow" on your command line (install python if you don't have it).
You might also need to install numpy. Use the above method with numpy if so.

To run this code, clone it with git and modify the bottom of the project.py file. Give the EmnistModelGenerator class the parameters you want to change.

Check the documentation of project.py for a list of attributes you can modify to get a different convolutional neural network setup.
You can change any of this data, although some things such as input_shape should not be changed if you are planning to use the EMNIST data set.

Use the save_model method to automatically save your model.
NOTE: you must manually make a models folder if you don't have one where you run the code or your model will not save correctly.

Use the print methods in DataPrinter to print a pass through the respective data corresponding to the specific print method.

